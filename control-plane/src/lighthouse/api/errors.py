"""Registry error -> HTTP status mapping.

Centralized so every route rejects an unrunnable model the same way. The
distinctions are the useful part:

  * `ModelNotFound` -> **404**: the operator mistyped a name.
  * `VersionNotReady` -> **409**: it exists but isn't built yet. Retry later.
  * `UnsupportedFlavor` -> **409**: it will never be runnable at the edge. Don't
    retry; the version needs re-exporting. Caught here rather than discovered on
    the device after a download.
  * `RegistryAuthError` -> **502**: *our* credential is bad, not the caller's.
    Returning 401 would tell the operator to re-authenticate, which would be
    wrong -- the broken credential is the control plane's workload JWT.
  * `RegistryUnavailable` -> **503** with Retry-After.
"""

from __future__ import annotations

from fastapi import HTTPException, status

from ..registry import (
    ArtifactUnavailable,
    ModelNotFound,
    RegistryAuthError,
    RegistryError,
    RegistryUnavailable,
    UnsupportedFlavor,
    VersionNotReady,
)


def registry_http_error(exc: RegistryError) -> HTTPException:
    if isinstance(exc, ModelNotFound):
        return HTTPException(status.HTTP_404_NOT_FOUND, detail=f"model not found: {exc}")
    if isinstance(exc, VersionNotReady):
        return HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"model version is not ready in the registry: {exc}",
        )
    if isinstance(exc, UnsupportedFlavor):
        return HTTPException(
            status.HTTP_409_CONFLICT,
            detail=(
                f"model version carries no edge-runnable artifact: {exc}. "
                "Re-export and register the ONNX flavor."
            ),
        )
    if isinstance(exc, RegistryAuthError):
        return HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail="the control plane's registry credential was rejected",
        )
    if isinstance(exc, (RegistryUnavailable, ArtifactUnavailable)):
        return HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"registry unavailable: {exc}",
            headers={"Retry-After": "10"},
        )
    return HTTPException(status.HTTP_502_BAD_GATEWAY, detail=f"registry error: {exc}")
