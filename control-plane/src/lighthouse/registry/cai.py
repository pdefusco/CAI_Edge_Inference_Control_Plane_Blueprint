"""The real Cloudera AI Registry adapter (spec SS11 / SS18, M2).

Everything in here exists to make `CAIModelRegistry` substitutable for
`FakeModelRegistry` on the far side of `ModelRegistry` -- which means every
quirk of the live registry's wire shape, its auth chain, and its own HTTP
failure modes gets absorbed in this one file instead of leaking into
`services/` or `api/`. Nothing above this module may see an `httpx`
exception, a `cdp` subprocess, or a UMS JWT.

Three things about the real registry are easy to get wrong and fail
*silently* rather than loudly, which is why they are called out verbatim here
rather than left to be rediscovered:

  * the model listing's id field is `id`, not `model_id` -- only
    `ModelVersion` carries a `model_id`;
  * versions live at `Model.model_versions`, not `versions`;
  * an empty registry answers `{"models": null}`, not `{"models": []}`.

`boto3` is deliberately never imported here, at any scope, even though it
rides along in the `cai` extra -- whether the control plane ever needs
object-store identity is an M3 finding, not an M2 assumption. `httpx` is fine
at module scope: `api/deps.py` imports this module lazily, specifically so a
local/fake run never pays for it.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Callable, Protocol

import httpx
from lighthouse_contracts import ArtifactFormat, Packaging

from ..config import ConfigError, Settings
from .base import (
    ArtifactStream,
    ModelNotFound,
    RegistryAuthError,
    RegistryError,
    RegistryModelVersion,
    RegistryUnavailable,
    UnsupportedFlavor,
    VersionFailed,
    VersionNotReady,
)

log = logging.getLogger(__name__)

# httpx content types that mean "the artifact body, uninterpreted" on the
# `/artifact` route. Anything else (multipart, or nothing at all because the
# header is missing) gets its own handling in `_open_artifact_stream`.
_RAW_ARTIFACT_CONTENT_TYPES = frozenset(
    {
        "application/octet-stream",
        "application/gzip",
        "application/x-tar",
        "application/x-gzip",
    }
)

# A JWT's own expiry minus this skew is used as the cache TTL, so a token is
# never handed out when it is seconds from rejection anyway.
_TOKEN_EXPIRY_SKEW_SECONDS = 120

# Fallback TTL when a token's `exp` claim cannot be read. Conservative on
# purpose: better to refresh an unexpired token than to serve a dead one.
_TOKEN_FALLBACK_TTL_SECONDS = 600

# How long `ping()`'s own result is cached. Short -- it exists to answer "is
# the chain healthy right now", not to save a request every few seconds.
_PING_CACHE_TTL_SECONDS = 5


# == token providers =========================================================


class TokenProvider(Protocol):
    """Something that can hand back a bearer token and be told it was wrong."""

    def token(self) -> str:
        ...

    def invalidate(self) -> None:
        """Discard any cached value; the next `token()` call must refresh."""
        ...


def _decode_jwt_exp(token: str) -> float | None:
    """Read the `exp` claim out of a JWT, with no signature check.

    This is reading our *own* token's self-reported expiry to size a cache
    TTL, not trusting a third party's claims -- so no verification, and
    deliberately no new crypto dependency for it.
    """
    parts = token.split(".")
    if len(parts) != 3:
        return None
    payload_segment = parts[1]
    try:
        padded = payload_segment + "=" * (-len(payload_segment) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded))
    except (binascii.Error, ValueError, TypeError):
        return None
    exp = payload.get("exp") if isinstance(payload, dict) else None
    if not isinstance(exp, (int, float)):
        return None
    return float(exp)


class _CachingTokenProviderMixin:
    """Shared TTL-cache behaviour for every `TokenProvider` implementation.

    The TTL is derived from the token's own `exp` claim (minus a skew) when
    readable, so a long-lived token is not needlessly re-minted and a
    short-lived one is not held past its welcome. `clock` is injectable so
    tests can move time without sleeping.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._cached: str | None = None
        self._expires_at: float = 0.0

    def _fetch(self) -> str:
        raise NotImplementedError

    def token(self) -> str:
        now = self._clock()
        if self._cached is not None and now < self._expires_at:
            return self._cached
        value = self._fetch()
        exp = _decode_jwt_exp(value)
        if exp is not None:
            # `exp` is wall-clock (epoch seconds) but our clock is monotonic;
            # we only need the *duration*, so measure it against wall time
            # once and apply that duration to the monotonic clock.
            ttl = max(0.0, exp - time.time() - _TOKEN_EXPIRY_SKEW_SECONDS)
        else:
            ttl = float(_TOKEN_FALLBACK_TTL_SECONDS)
        self._cached = value
        self._expires_at = now + ttl
        return value

    def invalidate(self) -> None:
        self._cached = None
        self._expires_at = 0.0


class CdpCliTokenProvider(_CachingTokenProviderMixin):
    """Default provider: shells out to `cdp iam generate-workload-auth-token`.

    This is the auth chain a CAI Session/Application actually has: a
    workload-scoped UMS JWT, not the workbench API key (which the gateway in
    front of the registry always 401s). `--workload-name` only accepts
    DE/DF/OPDB; any of them mints the same general-purpose JWT, so the
    default of "DE" is not a guess about which service is being called.
    """

    def __init__(
        self,
        workload_name: str = "DE",
        *,
        runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(clock=clock)
        self._workload_name = workload_name
        # Whether we will actually spawn a subprocess. The PATH probe below is
        # a precondition of doing that, not of this class existing, so an
        # injected runner must skip it -- otherwise the seam is only half a
        # seam and every test of this provider has to be gated on the real CLI
        # being installed, which silently skips them on CI.
        self._probe_path = runner is None
        self._runner = runner or _run_cdp

    def _fetch(self) -> str:
        if self._probe_path and shutil.which("cdp") is None:
            raise RegistryAuthError(
                "the `cdp` CLI is not on PATH; install it with `pip install cdpcli` "
                "and run `cdp configure` before using the CAI registry"
            )
        result = self._runner(
            ["cdp", "iam", "generate-workload-auth-token", "--workload-name", self._workload_name]
        )
        if result.returncode != 0:
            # stdout can carry a freshly minted JWT on some CLI versions even
            # on failure paths; only stderr is safe to surface, and even that
            # is truncated in case it echoes an argument back.
            stderr = (result.stderr or "").strip()[:400]
            raise RegistryAuthError(f"cdp iam generate-workload-auth-token failed: {stderr}")
        try:
            payload = json.loads(result.stdout)
        except ValueError as exc:
            raise RegistryAuthError(
                "cdp iam generate-workload-auth-token returned unparseable output"
            ) from exc
        token = payload.get("token") if isinstance(payload, dict) else None
        if not token:
            raise RegistryAuthError(
                "cdp iam generate-workload-auth-token response had no 'token' field"
            )
        return str(token)


class EnvTokenProvider(_CachingTokenProviderMixin):
    """Reads the token from an environment variable, for anywhere the `cdp`
    CLI is not available (e.g. a non-Session deployment of the control
    plane). There is no way to refresh an env var at runtime, so a second
    401 after `invalidate()` is treated as fatal -- `invalidate()` is a
    no-op here on purpose.
    """

    def __init__(self, env_var: str, *, clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__(clock=clock)
        self._env_var = env_var

    def _fetch(self) -> str:
        import os

        value = os.environ.get(self._env_var, "").strip()
        if not value:
            raise RegistryAuthError(f"environment variable {self._env_var} is unset or empty")
        return value

    def invalidate(self) -> None:  # noqa: D102 - documented on the class
        pass


class FileTokenProvider(_CachingTokenProviderMixin):
    """Reads the token from a file, re-read on `invalidate()`.

    Accepts either a raw token as the file's whole content, or a JSON object
    carrying it under `token` or `access_token` -- covers both "operator
    drops a bearer string in a file" and "something else writes CAI's own
    response shape there".
    """

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__(clock=clock)
        self._path = path

    def _fetch(self) -> str:
        try:
            raw = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RegistryAuthError(f"could not read token file {self._path}: {exc}") from exc
        stripped = raw.strip()
        if not stripped:
            raise RegistryAuthError(f"token file {self._path} is empty")
        try:
            parsed = json.loads(stripped)
        except ValueError:
            return stripped
        if isinstance(parsed, dict):
            token = parsed.get("token") or parsed.get("access_token")
            if token:
                return str(token)
        return stripped

    def invalidate(self) -> None:
        # Force the next token() call to re-read the file rather than reuse
        # whatever TTL the previous content's exp claim implied.
        self._cached = None
        self._expires_at = 0.0


def _run_cdp(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, timeout=30)


def discover_domain(
    environment: str,
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> str:
    """Resolve a CDP environment name to its registry's domain.

    `cdp ml list-model-registries` is tenant-wide -- it answers with every
    registry the caller's credentials can see, which in practice is mostly
    other people's. There is deliberately no "pick the first one" fallback:
    a match on `environmentName` is required, or this raises.
    """
    run = runner or _run_cdp
    # Probed only when we are the ones spawning it -- see CdpCliTokenProvider.
    if runner is None and shutil.which("cdp") is None:
        raise ConfigError(
            "the `cdp` CLI is not on PATH; install it with `pip install cdpcli` "
            "and run `cdp configure`, or set LIGHTHOUSE_REGISTRY_DOMAIN directly"
        )
    result = run(["cdp", "ml", "list-model-registries"])
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()[:400]
        raise ConfigError(f"cdp ml list-model-registries failed: {stderr}")
    try:
        payload = json.loads(result.stdout)
    except ValueError as exc:
        raise ConfigError("cdp ml list-model-registries returned unparseable output") from exc
    registries = payload.get("modelRegistries") if isinstance(payload, dict) else None
    registries = registries or []
    matches = [r for r in registries if isinstance(r, dict) and r.get("environmentName") == environment]
    if not matches:
        # Never name the other registries here -- this repo is public and
        # their domains/names are tenant identifiers.
        raise ConfigError(
            f"no model registry found for CDP environment {environment!r} "
            f"({len(registries)} registries visible to this credential, none matching)"
        )
    entry = matches[0]
    status = str(entry.get("status") or "")
    if not status.endswith(":finished"):
        log.warning(
            "model registry for environment %r has status %r, not a '...:finished' state",
            environment,
            status,
        )
    domain = entry.get("domain")
    if not domain:
        raise ConfigError(f"model registry for environment {environment!r} has no domain")
    return str(domain)


def _build_token_provider(settings: Settings) -> TokenProvider:
    source = settings.registry_token_source
    if source == "cli":
        return CdpCliTokenProvider(settings.registry_workload_name)
    if source == "env":
        if not settings.registry_token_env:
            raise ConfigError("registry_token_source is 'env' but registry_token_env is unset")
        return EnvTokenProvider(settings.registry_token_env)
    if source == "file":
        if not settings.registry_token_file:
            raise ConfigError("registry_token_source is 'file' but registry_token_file is unset")
        return FileTokenProvider(settings.registry_token_file)
    raise ConfigError(f"unknown registry_token_source: {source!r}")


# == transport ================================================================


class CAIRegistryClient:
    """Owns the one `httpx.Client` and is the sole place transport failures
    get normalized into the `RegistryError` hierarchy.

    Nothing above `_send` may see an `httpx` exception or a raw status code;
    everything else in this module calls `get_json` / `_send` and only ever
    catches `RegistryError` subclasses.
    """

    def __init__(
        self,
        settings: Settings,
        domain: str,
        token_provider: TokenProvider,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings
        self._token_provider = token_provider
        self._owns_client = client is None
        # Operators and `cdp ml list-model-registries` both sometimes include
        # a scheme; strip it so we never end up with "https://https://host".
        bare_domain = domain.split("://", 1)[-1].rstrip("/")
        self._base_url = f"https://{bare_domain}"
        self._api_prefix = settings.registry_api_prefix
        verify: bool | str = settings.registry_verify_tls
        if settings.registry_ca_bundle is not None:
            verify = str(settings.registry_ca_bundle)
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(settings.registry_request_timeout),
            verify=verify,
            headers={"User-Agent": "lighthouse-registry/0.1.0"},
            follow_redirects=False,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _url(self, path: str) -> str:
        return f"{self._base_url}{self._api_prefix}{path}"

    def _auth_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token_provider.token()}"}

    def _send(self, method: str, path: str, *, params: dict[str, Any] | None = None) -> httpx.Response:
        """The one normalization point for every CAI HTTP call.

        401/403 trigger exactly one retry after `invalidate()`; everything
        else maps straight through. The request path is safe to log -- the
        Authorization header and the token value never are.
        """
        url = self._url(path)
        attempted_retry = False
        while True:
            headers = self._auth_header()
            try:
                response = self._client.request(method, url, headers=headers, params=params)
            except httpx.TimeoutException as exc:
                raise RegistryUnavailable(f"timeout calling {path}: {exc}") from exc
            except httpx.ConnectError as exc:
                raise RegistryUnavailable(f"connection failed calling {path}: {exc}") from exc
            except httpx.TransportError as exc:
                raise RegistryUnavailable(f"transport error calling {path}: {exc}") from exc
            except httpx.HTTPError as exc:
                raise RegistryUnavailable(f"request failed calling {path}: {exc}") from exc

            if response.status_code in (401, 403):
                if attempted_retry:
                    raise RegistryAuthError(f"authentication rejected for {path}")
                attempted_retry = True
                self._token_provider.invalidate()
                continue
            return response

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self._send("GET", path, params=params)
        _raise_for_status(response, path)
        try:
            body = response.json()
        except ValueError as exc:
            raise RegistryError(f"unparseable JSON body from {path}: {exc}") from exc
        if not isinstance(body, dict):
            raise RegistryError(f"expected a JSON object from {path}, got {type(body).__name__}")
        return body

    def send_stream(self, method: str, path: str) -> httpx.Response:
        """Open a streaming request for the artifact route.

        Separate from `_send` because the caller must keep the response open
        (`stream=True` semantics via `client.send`) and apply the longer
        `registry_stream_timeout` to the read, not just the connect.
        """
        url = self._url(path)
        attempted_retry = False
        while True:
            headers = self._auth_header()
            request = self._client.build_request(
                method,
                url,
                headers=headers,
                timeout=httpx.Timeout(self._settings.registry_stream_timeout),
            )
            try:
                response = self._client.send(request, stream=True, follow_redirects=False)
            except httpx.TimeoutException as exc:
                raise RegistryUnavailable(f"timeout calling {path}: {exc}") from exc
            except httpx.ConnectError as exc:
                raise RegistryUnavailable(f"connection failed calling {path}: {exc}") from exc
            except httpx.TransportError as exc:
                raise RegistryUnavailable(f"transport error calling {path}: {exc}") from exc
            except httpx.HTTPError as exc:
                raise RegistryUnavailable(f"request failed calling {path}: {exc}") from exc

            if response.status_code in (401, 403):
                response.close()
                if attempted_retry:
                    raise RegistryAuthError(f"authentication rejected for {path}")
                attempted_retry = True
                self._token_provider.invalidate()
                continue
            return response

    def unauthenticated_get(self, url: str) -> httpx.Response:
        """A GET that carries no Authorization header at all.

        The only caller is the artifact-redirect follower: the workload JWT
        must never reach an object store that a redirect might point at.
        """
        try:
            return self._client.send(
                self._client.build_request(
                    "GET", url, timeout=httpx.Timeout(self._settings.registry_stream_timeout)
                ),
                stream=True,
                follow_redirects=False,
            )
        except httpx.HTTPError as exc:
            raise RegistryUnavailable(f"redirect fetch failed: {exc}") from exc


def _raise_for_status(response: httpx.Response, path: str) -> None:
    code = response.status_code
    if code < 400:
        return
    if code == 404:
        raise ModelNotFound(f"404 from {path}")
    if code in (408, 429) or code >= 500:
        raise RegistryUnavailable(f"{code} from {path}")
    raise RegistryError(f"{code} from {path}")


# == version parsing ==========================================================


def _parse_created_at(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None


def _tags_from_wire(raw: Any) -> dict[str, str]:
    tags: dict[str, str] = {}
    if not isinstance(raw, list):
        return tags
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        if not key:
            continue
        tags[str(key)] = str(entry.get("value", ""))
    return tags


def _repo_type_tokens(metadata: dict[str, Any]) -> set[str]:
    """Split `model_repo_type` into lowercase alphanumeric tokens.

    Matching on tokens rather than the whole string, because an API that
    returns `"mlflow"` in one place is apt to return `"MLFLOW_MODEL"` or
    `"MODEL_REPO_TYPE_MLFLOW"` in another, and exact equality against
    `"mlflow"` silently answers UNKNOWN for all of them -- which would make
    every real model undeployable. Tokens rather than a substring test,
    because `"hf"` inside a longer word is a false positive waiting to happen
    while `"hf"` as a token is unambiguous.
    """
    raw = str(metadata.get("model_repo_type") or "").strip().lower()
    return {t for t in re.split(r"[^a-z0-9]+", raw) if t}


def _format_and_packaging(metadata: dict[str, Any]) -> tuple[ArtifactFormat, Packaging]:
    """MLflow-sourced versions are *presumed* ONNX at the metadata layer only
    -- `MLFlowMetadata` carries no flavor information at all. The real check
    is `flavors.onnx.data` in the unpacked `MLmodel`, done by the artifact
    service at unpack time; this is just the best guess available before any
    bytes have been read.
    """
    tokens = _repo_type_tokens(metadata)
    has_mlflow = "mlflowMetadata" in metadata or "mlflow_metadata" in metadata
    has_hf = "huggingface_metadata" in metadata
    has_ngc = "ngc_metadata" in metadata

    if "mlflow" in tokens or (has_mlflow and not has_hf and not has_ngc):
        return ArtifactFormat.ONNX, Packaging.MLFLOW_TAR_GZ
    if tokens & {"hf", "huggingface", "ngc"} or has_hf or has_ngc:
        return ArtifactFormat.UNKNOWN, Packaging.RAW_FILE
    # Nothing in the metadata identifies a source -- including the case where
    # `metadata` is absent or `{}` entirely. Deliberately UNKNOWN rather than
    # presuming MLflow: UNKNOWN makes `get_version` refuse with
    # `UnsupportedFlavor` at the deploy gate.
    #
    # M3 SETTLED (2026-10-04), against the registry's own `/swagger.json` and
    # a real registered version. This line used to carry two candidate
    # realities needing opposite fixes; the spec rules one of them out
    # outright:
    #
    #   ModelVersionMetadata:
    #     *model_repo_type   string  enum=MLFLOW,HF,NGC      <- required
    #
    # `model_repo_type` is REQUIRED and its domain is exactly those three
    # values, so the feared case -- `metadata` genuinely `{}` for a normally
    # registered MLflow model, which would have made every real model
    # undeployable and forced this line to return ONNX -- cannot occur for a
    # version the registry itself produced. A real version was observed
    # reporting `model_repo_type: "MLFLOW"` with a fully populated
    # `mlflowMetadata`. All three enum values are handled above: "MLFLOW"
    # tokenizes to {"mlflow"}, "HF" to {"hf"}, "NGC" to {"ngc"}.
    #
    # So this fall-through is now unreachable for well-formed registry output
    # and exists purely to fail closed on a malformed or truncated response.
    # UNKNOWN makes `get_version` refuse with `UnsupportedFlavor` at the
    # deploy gate, which is the right answer for a payload this code cannot
    # identify -- do not "simplify" it into an optimistic ONNX guess. A
    # tarball that cannot be opened at all is deliberately not refused by
    # `artifact_service` (that is a transport symptom, not a flavor one), so
    # an optimistic guess here would still reach a device for a non-tar
    # artifact.
    return ArtifactFormat.UNKNOWN, Packaging.RAW_FILE


def _parse_version(raw: dict[str, Any], *, model_name: str, model_id: str) -> RegistryModelVersion:
    """Build a `RegistryModelVersion` from one wire `ModelVersion` object.

    Every field is read defensively -- a missing `artifact_uri` becomes `""`
    rather than a `KeyError`, because a malformed entry should degrade to an
    undeployable row the operator can see, not a 500 that hides the whole
    model's version list.
    """
    version = str(raw.get("version", ""))
    created_at = _parse_created_at(raw.get("created_at"))

    # The most important line in this file. CAI has no stable per-version
    # content id: a version is just an integer, and `DELETE
    # /models/{id}/versions/{version}` means that integer can be deleted and
    # reissued against entirely different bytes later. If `version_uuid` were
    # a bare `str(version)`, services/deployment_service.py's
    # `mv.cache_key != row.cache_key` drift check would PASS across a
    # reissue -- the label would look unchanged even though the bytes behind
    # it are not -- and the control plane would keep serving a device the
    # stale cached artifact. Composing the version with its creation instant
    # makes a reissue read as a new lineage instead. Deliberately
    # `created_at`, never `updated_at`: `updated_at` moves on a tag `PATCH`,
    # which would manufacture false drift (and a needless re-download) for a
    # version whose bytes never changed at all.
    if created_at is not None:
        version_uuid = f"{version}-{int(created_at.timestamp())}"
    else:
        version_uuid = version

    metadata = raw.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    fmt, packaging = _format_and_packaging(metadata)

    status = str(raw.get("status") or "UNKNOWN").upper()

    return RegistryModelVersion(
        name=model_name,
        version=version,
        model_id=model_id,
        version_uuid=version_uuid,
        artifact_uri=str(raw.get("artifact_uri") or ""),
        status=status,
        format=fmt,
        packaging=packaging,
        created_at=created_at,
        entrypoint=None,  # resolved later from MLmodel at unpack time
        size_bytes=None,  # the wire format has no size field at all
        tags=_tags_from_wire(raw.get("tags")),
    )


# == a file-like reader over a streamed httpx response ========================


class _StreamingArtifactReader:
    """Adapts an open `httpx.Response` (or a bounded multipart body within
    one) into the `read(n)` / `close()` surface `ArtifactStream.chunks()`
    needs. Nothing above `open_artifact` ever touches `httpx` directly.
    """

    def __init__(self, response: httpx.Response, chunk_iter: Any) -> None:
        self._response = response
        self._chunk_iter = chunk_iter
        self._buffer = b""
        self._exhausted = False

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            # Artifacts are multi-megabyte; `ArtifactStream.chunks()` is the
            # only sanctioned read path and always passes a bounded size.
            raise ValueError("_StreamingArtifactReader.read() requires a positive size")
        while len(self._buffer) < size and not self._exhausted:
            try:
                nxt = next(self._chunk_iter)
            except StopIteration:
                self._exhausted = True
                break
            except httpx.HTTPError as exc:
                raise RegistryUnavailable(f"artifact stream failed mid-transfer: {exc}") from exc
            self._buffer += nxt
        out, self._buffer = self._buffer[:size], self._buffer[size:]
        return out

    def close(self) -> None:
        try:
            self._response.close()
        except Exception:  # pragma: no cover - best-effort cleanup
            pass


def _multipart_boundary(content_type: str) -> str | None:
    for part in content_type.split(";"):
        part = part.strip()
        if part.lower().startswith("boundary="):
            value = part.split("=", 1)[1].strip()
            return value.strip('"')
    return None


def _multipart_body_chunks(raw_chunks: Any, boundary: str) -> Any:
    """Yield only the first part's body out of a `multipart/*` response,
    streaming rather than buffering the whole artifact in memory.

    Strategy: accumulate until the part headers end at the first `\\r\\n\\r\\n`,
    then emit body bytes while holding back enough tail that the closing
    delimiter can be recognized and trimmed before it would otherwise leak
    into the artifact bytes. The delimiter this adapter ever actually sees is
    the *terminal* one, `\\r\\n--boundary--\\r\\n` (there is only ever one
    part) -- `len(boundary)+6` is only enough to retain `\\r\\n--boundary`
    plus a 2-byte trailer, which covers an *intermediate* boundary's
    `\\r\\n--boundary\\r\\n` but is 2 bytes short of the terminal one's
    trailing `--\\r\\n`. Short by exactly that much, the last 2 bytes of
    `\\r\\n` get flushed as artifact body before the match is attempted, the
    search finds nothing, and the whole `--boundary--\\r\\n` leaks into the
    hashed bytes. `len(boundary)+8` covers it either way.
    """
    hold_back = len(boundary) + 8
    buf = bytearray()
    headers_done = False
    closer = f"--{boundary}".encode("ascii", "ignore")

    for chunk in raw_chunks:
        buf += chunk
        if not headers_done:
            idx = buf.find(b"\r\n\r\n")
            if idx == -1:
                continue
            del buf[: idx + 4]
            headers_done = True
        # Emit everything except a tail long enough to still contain the
        # closing boundary line, in case it straddles this chunk and the next.
        if len(buf) > hold_back:
            emit_len = len(buf) - hold_back
            to_emit = bytes(buf[:emit_len])
            del buf[:emit_len]
            cut = to_emit.find(b"\r\n" + closer)
            if cut != -1:
                if cut:
                    yield to_emit[:cut]
                return
            yield to_emit

    # Flush whatever remains, trimming a closing boundary if present.
    tail = bytes(buf)
    cut = tail.find(b"\r\n" + closer)
    if cut != -1:
        tail = tail[:cut]
    if tail:
        yield tail


# == the registry itself ======================================================


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


class CAIModelRegistry:
    """`ModelRegistry` backed by a live Cloudera AI Registry.

    `get_version` is the deploy-time gate and is never served from the TTL
    cache below -- a stale READY there would put bytes into desired state
    that the registry may already have deleted. Everything else (`list_models`,
    `list_versions`, and the name->id map) is cached for `registry_cache_ttl_seconds`,
    because every one of them exists purely for display and a few seconds of
    staleness there is harmless.
    """

    def __init__(
        self,
        settings: Settings,
        client: CAIRegistryClient,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._client = client
        self._clock = clock
        self._cache_ttl = float(settings.registry_cache_ttl_seconds)
        self._name_to_id_cache: _CacheEntry | None = None
        self._list_models_cache: _CacheEntry | None = None
        self._versions_cache: dict[str, _CacheEntry] = {}
        self._ping_cache: _CacheEntry | None = None
        self._logged_artifact_probe = False

    @classmethod
    def from_env(
        cls,
        settings: Settings,
        *,
        client: httpx.Client | None = None,
        runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> CAIModelRegistry:
        if settings.registry_domain:
            domain = settings.registry_domain
        elif settings.registry_environment:
            domain = discover_domain(settings.registry_environment, runner)
        else:
            raise ConfigError(
                "registry_impl is 'cai' but neither registry_domain nor "
                "registry_environment is set"
            )
        token_provider = _build_token_provider(settings)
        transport = CAIRegistryClient(settings, domain, token_provider, client=client)
        return cls(settings, transport, clock=clock)

    @property
    def name(self) -> str:
        return "cai"

    def close(self) -> None:
        self._client.close()

    # -- cache plumbing ----------------------------------------------------

    def _cache_get(self, entry: _CacheEntry | None) -> Any:
        if entry is None or self._clock() >= entry.expires_at:
            return None
        return entry.value

    def _cache_put(self, value: Any) -> _CacheEntry:
        return _CacheEntry(value=value, expires_at=self._clock() + self._cache_ttl)

    # -- listing / resolution ------------------------------------------------

    def _fetch_all_models(self) -> list[dict[str, Any]]:
        """Page through `GET /models`.

        `page_token` is CONFIRMED (2026-10-04) against the registry's own
        `/swagger.json`, which lists exactly four query parameters on this
        route: `page_size`, `page_token`, `search_filter`, `sort`. It used to
        be a guess, hence the non-advancing-token guard below -- which is
        kept, because it also covers a server that returns the same token
        forever, and that is a real failure mode independent of the name.
        """
        all_models: list[dict[str, Any]] = []
        seen_tokens: set[str] = set()
        page_token: str | None = None
        for _ in range(50):
            params: dict[str, Any] = {"page_size": 200}
            if page_token:
                params["page_token"] = page_token
            body = self._client.get_json("/models", params=params)
            models = body.get("models")
            if models:
                all_models.extend(m for m in models if isinstance(m, dict))
            next_token = body.get("next_page_token") or None
            if not next_token or next_token in seen_tokens:
                break
            seen_tokens.add(next_token)
            page_token = next_token
        else:
            log.warning("list_models() hit the 50-page cap; registry listing may be truncated")
        return all_models

    def _get_name_to_id(self, *, force_refresh: bool = False) -> dict[str, str]:
        if not force_refresh:
            cached = self._cache_get(self._name_to_id_cache)
            if cached is not None:
                return cached
        mapping: dict[str, str] = {}
        for entry in self._fetch_all_models():
            model_id = entry.get("id")
            name = entry.get("name")
            if not model_id or not name:
                continue
            if name in mapping:
                log.warning("duplicate model name %r across ids; keeping the first seen", name)
                continue
            mapping[name] = str(model_id)
        self._name_to_id_cache = self._cache_put(mapping)
        return mapping

    def _resolve_model_id(self, model_name: str) -> str:
        mapping = self._get_name_to_id()
        model_id = mapping.get(model_name)
        if model_id is not None:
            return model_id
        # A model registered seconds ago should be reachable without waiting
        # out the TTL, so refresh once before giving up.
        mapping = self._get_name_to_id(force_refresh=True)
        model_id = mapping.get(model_name)
        if model_id is None:
            raise ModelNotFound(f"no such model: {model_name}")
        return model_id

    def list_models(self) -> list[str]:
        cached = self._cache_get(self._list_models_cache)
        if cached is not None:
            return cached
        names = sorted(self._get_name_to_id())
        self._list_models_cache = self._cache_put(names)
        return names

    def list_versions(self, model_name: str) -> list[RegistryModelVersion]:
        cached = self._cache_get(self._versions_cache.get(model_name))
        if cached is not None:
            return cached
        model_id = self._resolve_model_id(model_name)
        # `GET /models/{model_id}`, deliberately -- NOT
        # `/models/{model_id}/versions`, which does not exist and answers 405
        # (observed 2026-10-04; the spec's 11 paths confirm no collection
        # route for versions, only `/versions/{version}` for a single one).
        # Versions arrive nested under the model as `model_versions`.
        body = self._client.get_json(f"/models/{model_id}")
        raw_versions = body.get("model_versions") or []
        parsed = [
            _parse_version(v, model_name=model_name, model_id=model_id)
            for v in raw_versions
            if isinstance(v, dict)
        ]

        def _sort_key(mv: RegistryModelVersion) -> tuple[int, int | str]:
            try:
                return (0, int(mv.version))
            except ValueError:
                return (1, mv.version)

        parsed.sort(key=_sort_key)
        self._versions_cache[model_name] = self._cache_put(parsed)
        return parsed

    def get_version(self, model_name: str, version: str) -> RegistryModelVersion:
        """The deploy-time gate. Never cached -- see the class docstring."""
        model_id = self._resolve_model_id(model_name)
        body = self._client.get_json(f"/models/{model_id}/versions/{version}")
        mv = _parse_version(body, model_name=model_name, model_id=model_id)

        status = mv.status
        if status in {"REGISTERING", "UPLOADING"}:
            raise VersionNotReady(f"{model_name} v{version} is {status.lower()}")
        if status == "UNKNOWN":
            # Not claimed terminal: the raw status is surfaced so an operator
            # can tell this apart from a genuine registry outage.
            raise VersionNotReady(f"{model_name} v{version} has registry status UNKNOWN")
        if status in {"UPLOAD_FAILED", "DELETE_FAILED"}:
            error_message = str(body.get("error_message") or "no error_message from registry")
            raise VersionFailed(f"{model_name} v{version} ({status}): {error_message}")
        if status in {"DELETED", "DELETING"}:
            raise ModelNotFound(f"{model_name} v{version} is {status.lower()}")
        if status != "READY":
            # Any future status value the registry adds: fail closed rather
            # than silently deploying something we don't understand.
            raise VersionNotReady(f"{model_name} v{version} has registry status {status}")

        if mv.format is not ArtifactFormat.ONNX:
            raise UnsupportedFlavor(
                f"{model_name} v{version} has no ONNX flavor; the edge cannot run it"
            )
        return mv

    # -- artifact bytes ------------------------------------------------------

    def open_artifact(self, mv: RegistryModelVersion) -> ArtifactStream:
        """Stream the version's bytes from the registry's own artifact route.

        OBSERVED 2026-10-04 against a real registered MLflow version, which
        settles the last of M2's three open questions:

            status           = 200
            Content-Type     = application/octet-stream
            Content-Length   = 876
            Content-Encoding = <absent>
            body             starts 1f 8b (gzip magic)

        Three consequences worth stating, because each was an open risk:

        * **No redirect.** The registry streams the bytes itself rather than
          302-ing to presigned object storage, so the control plane needs no
          object-store identity of its own. That is why there is still no
          `boto3` anywhere above this layer, and nothing here needs one.
        * **`Packaging.MLFLOW_TAR_GZ` is right**, confirmed by the gzip magic
          rather than inferred from the `.tar.gz` suffix.
        * **No transparent-decode trap**, this time. `Content-Encoding` was
          absent, so httpx hands over the gzip stream intact. The probe below
          still logs that header, because this is one tenant's behaviour and
          the failure it guards against is close to undiagnosable.

        Note `artifact_uri` on the version is an `s3a://` path into tenant
        object storage -- unreachable with the registry credential, and not a
        fetch path. This route is the only way to the bytes.

        The spec declares `produces: multipart/form-data` for this route while
        the server actually sent `application/octet-stream`. The branching in
        `_open_artifact_stream` keys off the *observed* `Content-Type` and so
        handles both; the multipart branch is simply unexercised so far. Do
        not delete it on the strength of one tenant's response.
        """
        response = self._client.send_stream(
            "GET", f"/models/{mv.model_id}/versions/{mv.version}/artifact"
        )
        return self._open_artifact_stream(response, mv)

    def _open_artifact_stream(self, response: httpx.Response, mv: RegistryModelVersion) -> ArtifactStream:
        status = response.status_code
        content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()

        if not self._logged_artifact_probe:
            self._logged_artifact_probe = True
            # `content-encoding` is here for a specific failure that is
            # otherwise close to undiagnosable. httpx decodes a
            # `Content-Encoding: gzip` body transparently, so if the registry
            # serves the tarball that way the bytes written to the cache are a
            # *bare* tar while `Packaging` still says MLFLOW_TAR_GZ. Nothing
            # upstream notices: the control plane hashes the decoded bytes and
            # the device receives and verifies those same bytes, so every
            # checksum in the system agrees. The only thing that fails is
            # `tarfile.open(..., "r:gz")` -- once in `_read_entrypoint`
            # server-side, and again on the Jetson in `_activate`.
            #
            # This is the one line anybody will read after the first real
            # registration, so the header that distinguishes that case from a
            # healthy response belongs in it.
            log.info(
                "artifact response: status=%s content-type=%r content-length=%r "
                "content-encoding=%r transfer-encoding=%r",
                status,
                response.headers.get("content-type"),
                response.headers.get("content-length"),
                response.headers.get("content-encoding"),
                response.headers.get("transfer-encoding"),
            )

        if 300 <= status < 400:
            location = response.headers.get("location")
            response.close()
            if not location:
                raise RegistryError(
                    f"{status} from artifact route for {mv.model_id}/{mv.version} had no Location"
                )
            # Follow the redirect with NO Authorization header: the object
            # store behind it must never see the workload JWT.
            redirected = self._client.unauthenticated_get(location)
            if redirected.status_code >= 400:
                redirected.close()
                raise RegistryUnavailable(
                    f"redirected artifact fetch failed with {redirected.status_code}"
                )
            fileobj: IO[bytes] = _StreamingArtifactReader(  # type: ignore[assignment]
                redirected, redirected.iter_bytes()
            )
            return ArtifactStream(
                fileobj=fileobj,
                packaging=mv.packaging,
                size_bytes=None,
                source_uri=mv.artifact_uri,
                etag=None,
            )

        if status == 400:
            response.close()
            raise UnsupportedFlavor(
                f"artifact route refused {mv.model_id}/{mv.version} "
                "(model type is HF or NGC, not downloadable here)"
            )

        if status >= 400:
            try:
                response.read()
            except httpx.HTTPError:  # pragma: no cover - already failing
                pass
            _raise_for_status(response, f"/models/{mv.model_id}/versions/{mv.version}/artifact")
            # _raise_for_status always raises for status >= 400; this is
            # unreachable but keeps type-checkers happy.
            raise RegistryError(f"{status} fetching artifact")

        if content_type.startswith("multipart/"):
            boundary = _multipart_boundary(response.headers.get("content-type") or "")
            if not boundary:
                response.close()
                raise RegistryError("multipart artifact response had no boundary")
            chunk_iter = _multipart_body_chunks(response.iter_bytes(), boundary)
            fileobj = _StreamingArtifactReader(response, chunk_iter)  # type: ignore[assignment]
            return ArtifactStream(
                fileobj=fileobj,
                packaging=mv.packaging,
                size_bytes=None,
                source_uri=mv.artifact_uri,
                etag=None,
            )

        # 200 with a raw content type, or no content type at all: stream the
        # body straight through.
        if content_type and content_type not in _RAW_ARTIFACT_CONTENT_TYPES:
            log.warning("unexpected artifact content-type %r; streaming it as raw bytes", content_type)
        fileobj = _StreamingArtifactReader(response, response.iter_bytes())  # type: ignore[assignment]
        return ArtifactStream(
            fileobj=fileobj,
            packaging=mv.packaging,
            size_bytes=None,
            source_uri=mv.artifact_uri,
            etag=None,
        )

    def get_artifact_uri(self, mv: RegistryModelVersion) -> str:
        return mv.artifact_uri

    def ping(self) -> bool:
        cached = self._cache_get(self._ping_cache)
        if cached is not None:
            return bool(cached)
        try:
            self._client.get_json("/models", params={"page_size": 1})
            result = True
        except RegistryError:
            result = False
        self._ping_cache = _CacheEntry(value=result, expires_at=self._clock() + _PING_CACHE_TTL_SECONDS)
        return result
