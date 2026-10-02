"""Artifact download.

The spec's SS12 endpoint list has no artifact route, because it assumes the device
can read `artifact_uri` from the registry. Verified against the live cluster, it
cannot: registry *metadata* is an HTTPS API, but the artifact *bytes* live in
object storage a Jetson at home has no identity for. So the control plane brokers
them, and this is that route.

It is **device-scoped**, and that is the authorization model rather than an
afterthought: the artifact is resolved through the requesting device's own current
desired state, so there is no parameter through which a device could name a model
it was never assigned.

`Range` and `If-Match` are both honored. CML's ingress behaviour on long transfers
is undocumented -- unknown idle timeout, unknown whether it buffers -- so a cut
transfer has to cost one round trip rather than a restart. `If-Match` against the
digest closes the matching hazard: if the artifact changed while a device was
resuming, splicing new bytes onto old ones would produce a file that matches no
digest at all, and the device would report corruption instead of a conflict.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse

from ..services import DevicePrincipal
from .auth import require_device, require_operator
from .deps import AppContext, ctx

log = logging.getLogger(__name__)

router = APIRouter(tags=["artifacts"])

_MEDIA_TYPE = "application/gzip"

# Why a device was told to wait, and what it should do about it. Returned in the
# error body so the agent branches on a stable code rather than parsing prose.
_REASON_STATUS = {
    "stale_generation": (status.HTTP_409_CONFLICT, "desired state has advanced; re-poll"),
    "revoked": (status.HTTP_403_FORBIDDEN, "deployment is revoked"),
    "no_deployment": (status.HTTP_404_NOT_FOUND, "no model is assigned to this device"),
    "not_ready": (status.HTTP_503_SERVICE_UNAVAILABLE, "artifact is still being materialized"),
}


def _parse_range(header: str, size: int) -> tuple[int, int] | None:
    """Parse a single-range `bytes=` header into inclusive (start, end).

    Returns None for a syntactically valid but unsatisfiable range, which is a 416.
    Multi-range requests are not supported -- a resuming download never needs one,
    and `multipart/byteranges` would be real complexity for no caller.
    """
    if not header.startswith("bytes="):
        return None
    spec = header[len("bytes=") :].strip()
    if "," in spec:
        return None
    start_s, _, end_s = spec.partition("-")
    try:
        if not start_s:
            # Suffix form: bytes=-500 means the last 500 bytes.
            length = int(end_s)
            if length <= 0:
                return None
            start = max(0, size - length)
            return start, size - 1
        start = int(start_s)
        end = int(end_s) if end_s else size - 1
    except ValueError:
        return None
    if start < 0 or start >= size or end < start:
        return None
    return start, min(end, size - 1)


@router.get("/devices/{device_id}/artifact")
def download_device_artifact(
    request: Request,
    device_id: str,
    generation: int | None = None,
    principal: DevicePrincipal = Depends(require_device),
    context: AppContext = Depends(ctx),
) -> Response:
    """Stream the artifact this device is currently authorized to run.

    `generation` is checked rather than decorative. A device that finishes a
    download against a superseded instruction must restart from the new one; 409
    here is what makes "superseded generation mid-download aborts cleanly" true
    instead of aspirational.
    """
    info, reason = context.deployments.resolve_artifact(device_id, generation)
    if info is None:
        code, detail = _REASON_STATUS.get(
            reason, (status.HTTP_503_SERVICE_UNAVAILABLE, "artifact unavailable")
        )
        # Retry-After gives a waiting agent a server-chosen backoff instead of
        # letting every device in the fleet pick its own.
        headers = (
            {"Retry-After": str(context.settings.heartbeat_interval_seconds)}
            if code == status.HTTP_503_SERVICE_UNAVAILABLE
            else None
        )
        raise HTTPException(code, detail=detail, headers=headers)

    etag = f'"{info.sha256}"'

    # If-Match before Range: a mismatch means the bytes changed under a resuming
    # download, and the agent must restart rather than splice.
    if_match = request.headers.get("if-match")
    if if_match and if_match.strip() not in {etag, "*"}:
        raise HTTPException(
            status.HTTP_412_PRECONDITION_FAILED,
            detail="artifact digest has changed; restart the download",
        )

    base_headers = {
        "ETag": etag,
        "Accept-Ranges": "bytes",
        # The digest is also exposed as a plain header so the agent can record it
        # without unquoting an ETag.
        "X-Lighthouse-SHA256": info.sha256,
        "X-Lighthouse-Packaging": info.packaging.value,
        # No caching: the device verifies by digest, and an intermediary holding a
        # stale copy of a revoked artifact is exactly what must not happen.
        "Cache-Control": "no-store",
    }
    if info.entrypoint:
        base_headers["X-Lighthouse-Entrypoint"] = info.entrypoint

    range_header = request.headers.get("range")
    if range_header:
        parsed = _parse_range(range_header, info.size_bytes)
        if parsed is None:
            raise HTTPException(
                status.HTTP_416_RANGE_NOT_SATISFIABLE,
                detail="unsatisfiable range",
                headers={"Content-Range": f"bytes */{info.size_bytes}"},
            )
        start, end = parsed
        headers = {
            **base_headers,
            "Content-Range": f"bytes {start}-{end}/{info.size_bytes}",
            "Content-Length": str(end - start + 1),
        }
        log.info(
            "serving %s bytes %s-%s of %s to %s",
            info.cache_key,
            start,
            end,
            info.size_bytes,
            device_id,
        )
        return StreamingResponse(
            context.artifacts.iter_range(info, start, end),
            status_code=status.HTTP_206_PARTIAL_CONTENT,
            media_type=_MEDIA_TYPE,
            headers=headers,
        )

    log.info("serving %s (%s bytes) to %s", info.cache_key, info.size_bytes, device_id)
    return StreamingResponse(
        context.artifacts.iter_range(info),
        media_type=_MEDIA_TYPE,
        headers={**base_headers, "Content-Length": str(info.size_bytes)},
    )


@router.get("/artifacts/{model_name}/{version}")
def download_artifact_as_operator(
    model_name: str,
    version: str,
    _operator=Depends(require_operator),
    context: AppContext = Depends(ctx),
) -> Response:
    """Operator-scoped download, for debugging what a device would receive.

    Separate from the device route because it resolves through the registry rather
    than through any device's desired state -- useful for ops, and deliberately not
    reachable with a device token.
    """
    mv = context.registry.get_version(model_name, version)
    info = context.artifacts.get_ready(mv.cache_key)
    if info is None:
        context.artifacts.request(mv)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="artifact is being materialized; retry shortly",
            headers={"Retry-After": "5"},
        )
    return StreamingResponse(
        context.artifacts.iter_range(info),
        media_type=_MEDIA_TYPE,
        headers={
            "ETag": f'"{info.sha256}"',
            "Content-Length": str(info.size_bytes),
            "X-Lighthouse-SHA256": info.sha256,
            "Cache-Control": "no-store",
        },
    )
