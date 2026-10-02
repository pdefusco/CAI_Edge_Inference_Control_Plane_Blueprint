"""The agent's only outbound surface.

Every request in this file is initiated *by the device*. There is no server, no
listening socket, and no inbound path into the network the device sits on -- which
is the whole reason a box behind a home router can be governed from CAI at all.

Transport failures are normalised into `TransientError` (retry, say nothing
alarming) versus `AuthError` and `ProtocolError` (a human has to fix something).
The distinction is load-bearing: the reconciler must keep calm through a flaky
uplink and must *not* mark a model FAILED because a DNS lookup blipped.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx
from lighthouse_contracts import (
    DesiredStateResponse,
    HeartbeatRequest,
    HeartbeatResponse,
)
from pydantic import ValidationError

from .config import AgentSettings

log = logging.getLogger(__name__)


class ClientError(RuntimeError):
    """Base for everything this module raises."""


class TransientError(ClientError):
    """Network trouble or a 5xx. Retry; this is not a deployment failure."""


class AuthError(ClientError):
    """401/403. The device token is wrong, revoked, or aimed at another device.

    Never retried quickly: hammering a rejected credential is how an agent gets
    itself rate-limited or noticed as an attack.
    """


class ProtocolError(ClientError):
    """The server answered, but not with something this agent understands.

    Usually a version skew between control plane and agent. Surfaced loudly
    rather than coerced, because guessing at a changed contract is how an agent
    deploys the wrong bytes.
    """


class NotFoundError(ClientError):
    """404. The device is not enrolled (or was deleted) on this control plane."""


class ArtifactNotReadyError(ClientError):
    """503 from the artifact route: the control plane is still materializing.

    Distinct from `TransientError` only so the reconciler can log it as
    "waiting", not "network problem".
    """


class ArtifactForbiddenError(ClientError):
    """403 from the artifact route: this generation is revoked.

    The bytes are gone and will not come back for this generation. The reconciler
    must not treat it as retryable.
    """


class GenerationStaleError(ClientError):
    """409 from the artifact route: a newer generation superseded this one.

    Abandon the download; the next poll carries the current instruction.
    """


class ControlPlaneClient:
    """Typed wrapper over the Lighthouse device API."""

    def __init__(self, settings: AgentSettings, *, client: httpx.Client | None = None) -> None:
        self._settings = settings
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(settings.request_timeout_seconds),
            verify=settings.verify_tls,
            headers={"User-Agent": f"keeper/{settings.device_id}"},
            follow_redirects=False,
        )

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> ControlPlaneClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- device API --------------------------------------------------------

    def fetch_desired_state(self) -> DesiredStateResponse:
        """GET /api/v1/devices/{id}/desired-state."""
        path = f"/api/v1/devices/{self._settings.device_id}/desired-state"
        response = self._request("GET", self._settings.url(path))
        try:
            return DesiredStateResponse.model_validate(response.json())
        except (ValidationError, ValueError) as exc:
            raise ProtocolError(f"unreadable desired-state payload: {exc}") from exc

    def send_heartbeat(self, heartbeat: HeartbeatRequest) -> HeartbeatResponse:
        """POST /api/v1/devices/{id}/heartbeat."""
        path = f"/api/v1/devices/{self._settings.device_id}/heartbeat"
        response = self._request(
            "POST",
            self._settings.url(path),
            json=heartbeat.model_dump(mode="json", exclude_none=True),
        )
        try:
            return HeartbeatResponse.model_validate(response.json())
        except (ValidationError, ValueError) as exc:
            raise ProtocolError(f"unreadable heartbeat response: {exc}") from exc

    @contextmanager
    def stream_artifact(
        self,
        artifact_uri: str,
        *,
        offset: int = 0,
        if_match: str | None = None,
    ) -> Iterator[httpx.Response]:
        """Open a streaming GET for artifact bytes, optionally resuming.

        `offset > 0` sends `Range: bytes=<offset>-` *together with* `If-Match`, so
        a server whose bytes changed under us answers 412 instead of splicing two
        different artifacts into one file. That pairing is the entire safety
        argument for resume.
        """
        headers: dict[str, str] = dict(self._auth_header())
        if offset > 0:
            headers["Range"] = f"bytes={offset}-"
        if if_match:
            headers["If-Match"] = f'"{if_match}"' if not if_match.startswith('"') else if_match

        url = artifact_uri if artifact_uri.startswith("http") else self._settings.url(artifact_uri)
        try:
            with self._client.stream(
                "GET",
                url,
                headers=headers,
                timeout=httpx.Timeout(
                    # Generous read timeout for the body, normal timeout to connect:
                    # a large artifact over a home uplink is slow, not broken.
                    self._settings.download_timeout_seconds,
                    connect=self._settings.request_timeout_seconds,
                ),
            ) as response:
                _raise_for_artifact_status(response)
                yield response
        except httpx.HTTPError as exc:
            raise TransientError(f"artifact stream failed: {exc}") from exc

    # -- plumbing ----------------------------------------------------------

    def _auth_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._settings.token}"}

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        headers = dict(kwargs.pop("headers", {}) or {})
        headers.update(self._auth_header())
        try:
            response = self._client.request(method, url, headers=headers, **kwargs)
        except httpx.HTTPError as exc:
            # One normalisation point for every connect/DNS/TLS/timeout failure,
            # so callers never import httpx exception types.
            raise TransientError(f"{method} {_safe_url(url)} failed: {exc}") from exc
        _raise_for_status(response, url)
        return response


def _raise_for_status(response: httpx.Response, url: str) -> None:
    code = response.status_code
    if code < 400:
        return
    detail = _error_detail(response)
    if code in (401, 403):
        raise AuthError(f"{code} from {_safe_url(url)}: {detail}")
    if code == 404:
        raise NotFoundError(f"404 from {_safe_url(url)}: {detail}")
    if code == 422:
        raise ProtocolError(f"422 from {_safe_url(url)}: {detail}")
    if code >= 500 or code == 429:
        raise TransientError(f"{code} from {_safe_url(url)}: {detail}")
    raise ProtocolError(f"{code} from {_safe_url(url)}: {detail}")


def _raise_for_artifact_status(response: httpx.Response) -> None:
    """Artifact-route statuses carry meanings the generic mapper would flatten."""
    code = response.status_code
    if code in (200, 206):
        return
    # The body of a streamed error response has not been read yet.
    try:
        response.read()
    except httpx.HTTPError:  # pragma: no cover - already failing
        pass
    detail = _error_detail(response)
    if code == 403:
        raise ArtifactForbiddenError(f"artifact revoked: {detail}")
    if code == 409:
        raise GenerationStaleError(f"generation superseded: {detail}")
    if code == 412:
        # The cached partial file is from different bytes than the server now
        # holds. Caller discards the .part and restarts from zero.
        raise GenerationStaleError(f"artifact digest changed mid-download: {detail}")
    if code == 503:
        raise ArtifactNotReadyError(f"artifact not ready: {detail}")
    if code == 416:
        raise GenerationStaleError(f"range not satisfiable: {detail}")
    if code == 401:
        raise AuthError(f"artifact request rejected: {detail}")
    if code == 404:
        raise NotFoundError(f"no artifact for this device: {detail}")
    if code >= 500:
        raise TransientError(f"{code} fetching artifact: {detail}")
    raise ProtocolError(f"{code} fetching artifact: {detail}")


def _error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return (response.text or "").strip()[:200]
    if isinstance(body, dict):
        return str(body.get("message") or body.get("detail") or body)[:200]
    return str(body)[:200]


def _safe_url(url: str) -> str:
    """Drop any query string before a URL reaches a log line.

    Nothing in this agent's own requests puts a credential in a query string, but
    a presigned URL would arrive with one, and logs outlive the signature.
    """
    return url.split("?", 1)[0]
