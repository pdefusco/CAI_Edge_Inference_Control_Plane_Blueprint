"""Device enrollment, credentials and the operator-facing device views.

The token format is `lhd_<token_id>.<secret>`, and the split is not decoration.
With only an opaque secret, verification degrades to scanning every token row and
comparing; with a public id, it is one indexed lookup plus a single constant-time
compare. The device_id is deliberately *not* in the token -- identity comes from
the stored row, so a device cannot assert who it is.

**SHA-256 is the right hash here, and not an oversight.** These secrets are
256-bit CSPRNG output, not passwords. bcrypt and argon2 exist to make guessing
*low*-entropy inputs expensive; against full entropy they buy nothing, while
adding per-request CPU to a loop that runs every ten seconds per device. Please
don't "fix" this.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from dataclasses import dataclass

from lighthouse_contracts import (
    DeviceRegistrationRequest,
    DeviceTokenIssued,
    DeviceView,
    EventType,
)

from ..config import Settings
from ..repositories import DeviceExists, DeviceRow, DeviceTokenRow, Store
from ..util import now_utc
from .artifact_service import ArtifactService
from .audit import AuditService
from .errors import DeviceAlreadyExists, DeviceNotFound
from .governance import build_device_view

log = logging.getLogger(__name__)

_TOKEN_PREFIX = "lhd_"
# Write last_used_at at most this often per token. A heartbeat every 10s would
# otherwise mean a write per heartbeat per device purely for bookkeeping.
_TOKEN_USE_THROTTLE_SECONDS = 60


@dataclass(frozen=True, slots=True)
class DevicePrincipal:
    """An authenticated device."""

    device_id: str
    token_id: str


class DeviceService:
    def __init__(
        self,
        store: Store,
        audit: AuditService,
        settings: Settings,
        artifacts: ArtifactService | None = None,
    ) -> None:
        self._store = store
        self._audit = audit
        self._settings = settings
        self._artifacts = artifacts
        self._last_use_write: dict[str, float] = {}

    # -- enrollment ------------------------------------------------------

    def register(
        self, request: DeviceRegistrationRequest
    ) -> tuple[DeviceRow, DeviceTokenIssued]:
        """Enrol a device and issue its first token.

        The token is returned here and nowhere else -- only `sha256(secret)` is
        stored, so there is no recovery path, only reissue.
        """
        try:
            device = self._store.create_device(
                request.device_id, request.display_name, request.platform
            )
        except DeviceExists as exc:
            raise DeviceAlreadyExists(request.device_id) from exc

        self._audit.record(
            EventType.DEVICE_REGISTERED,
            device_id=device.device_id,
            display_name=request.display_name,
            platform=request.platform,
        )
        issued = self.issue_token(device.device_id, label="initial")
        return device, issued

    def issue_token(self, device_id: str, label: str | None = None) -> DeviceTokenIssued:
        """Mint an additional token.

        Several tokens can be valid at once, which is what makes rotation
        downtime-free: issue, deploy to the device, then revoke the old one.
        """
        if self._store.get_device(device_id) is None:
            raise DeviceNotFound(device_id)

        token_id = secrets.token_hex(8)
        secret = secrets.token_urlsafe(32)
        self._store.create_token(
            DeviceTokenRow(
                token_id=token_id,
                device_id=device_id,
                token_sha256=_hash_secret(secret),
                created_at=now_utc(),
                label=label,
            )
        )
        self._audit.record(
            EventType.DEVICE_TOKEN_ISSUED,
            device_id=device_id,
            token_id=token_id,
            label=label,
        )
        return DeviceTokenIssued(
            device_id=device_id,
            token_id=token_id,
            token=f"{_TOKEN_PREFIX}{token_id}.{secret}",
        )

    def revoke_token(self, token_id: str) -> bool:
        row = self._store.get_token(token_id)
        revoked = self._store.revoke_token(token_id)
        if revoked:
            self._audit.record(
                EventType.DEVICE_TOKEN_REVOKED,
                device_id=row.device_id if row else None,
                token_id=token_id,
            )
        return revoked

    # -- authentication --------------------------------------------------

    def authenticate(self, presented: str) -> DevicePrincipal | None:
        """Verify a device bearer token.

        Returns None for every failure mode -- malformed, unknown, revoked, wrong
        secret -- so a caller cannot probe which token ids exist by comparing
        error responses.
        """
        parsed = _parse_token(presented)
        if parsed is None:
            return None
        token_id, secret = parsed

        row = self._store.get_token(token_id)
        if row is None:
            # Still spend a compare against a dummy value so an unknown token id
            # is not measurably faster to reject than a known one.
            hmac.compare_digest(_hash_secret(secret), _hash_secret("dummy"))
            return None
        if not row.active:
            return None
        if not hmac.compare_digest(_hash_secret(secret), row.token_sha256):
            return None

        self._touch_token(token_id)
        return DevicePrincipal(device_id=row.device_id, token_id=token_id)

    def _touch_token(self, token_id: str) -> None:
        now = now_utc().timestamp()
        last = self._last_use_write.get(token_id, 0.0)
        if now - last < _TOKEN_USE_THROTTLE_SECONDS:
            return
        self._last_use_write[token_id] = now
        try:
            self._store.mark_token_used(token_id)
        except Exception:  # pragma: no cover - bookkeeping only
            log.debug("could not update last_used_at for %s", token_id, exc_info=True)

    # -- views -----------------------------------------------------------

    def list_views(self) -> list[DeviceView]:
        now = now_utc()
        return [self._view(d, now=now) for d in self._store.list_devices()]

    def get_view(self, device_id: str) -> DeviceView:
        device = self._store.get_device(device_id)
        if device is None:
            raise DeviceNotFound(device_id)
        return self._view(device)

    def _view(self, device: DeviceRow, now=None) -> DeviceView:
        desired = self._store.get_desired(device.device_id)
        actual = self._store.get_actual(device.device_id)
        artifact_ready = True
        if desired is not None and desired.cache_key is not None and self._artifacts is not None:
            artifact_ready = self._artifacts.get_ready(desired.cache_key) is not None
        return build_device_view(
            device,
            desired,
            actual,
            self._settings,
            artifact_ready=artifact_ready,
            now=now,
        )

    def delete(self, device_id: str) -> None:
        if self._store.get_device(device_id) is None:
            raise DeviceNotFound(device_id)
        self._store.delete_device(device_id)


def _hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _parse_token(presented: str) -> tuple[str, str] | None:
    """Split `lhd_<token_id>.<secret>` into its parts, or None if malformed."""
    if not presented or not presented.startswith(_TOKEN_PREFIX):
        return None
    body = presented[len(_TOKEN_PREFIX) :]
    token_id, _, secret = body.partition(".")
    if not token_id or not secret:
        return None
    return token_id, secret
