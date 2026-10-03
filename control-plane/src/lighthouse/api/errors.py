"""Registry error -> HTTP status mapping.

Centralized so every route rejects an unrunnable model the same way. The
distinctions are the useful part:

  * `ModelNotFound` -> **404**: the operator mistyped a name.
  * `VersionNotReady` -> **409**: it exists but isn't built yet. Retry later.
  * `VersionFailed` -> **409**: the registry tried to build it and gave up. Also
    a conflict, but the opposite advice -- retrying waits forever, so the detail
    says to register a new version and carries the registry's own reason.
  * `UnsupportedFlavor` -> **409**: it will never be runnable at the edge. Don't
    retry; the version needs re-exporting. Caught here rather than discovered on
    the device after a download.
  * `RegistryAuthError` -> **502**: *our* credential is bad, not the caller's.
    Returning 401 would tell the operator to re-authenticate, which would be
    wrong -- the broken credential is the control plane's workload JWT.
  * `RegistryUnavailable` -> **503** with Retry-After.

This module also owns the one shape every error leaves on, because there were
three of them. A route that caught `RegistryError` and called
`registry_http_error` emitted `{"detail": "<prose>"}`; one that let the same
exception propagate to the handler in `main.py` emitted
`{"code": ..., "message": ...}` -- the *same logical error* with a different
body depending on which call path it took. Meanwhile `ErrorResponse` in
contracts declared a third shape, with a docstring promising "the agent can
branch on a stable code rather than parsing prose", and had no producer
anywhere. The agent duly guessed at both live shapes
(`keeper/client.py:239-246`) and branched on HTTP status instead.

So `error_body` is the only thing that writes an error body now, and it
validates through contracts' `ErrorResponse` so the declared shape and the
served shape cannot drift apart again. `code` is the stable part: the registry
exception's class name where there is one, otherwise a slug naming the status.
Making the agent branch on `code` is deliberately a separate change -- this one
only makes there be something dependable to branch on.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status
from lighthouse_contracts import ErrorResponse

from ..registry import (
    ArtifactUnavailable,
    ModelNotFound,
    RegistryAuthError,
    RegistryError,
    RegistryUnavailable,
    UnsupportedFlavor,
    VersionFailed,
    VersionNotReady,
)


def _registry_mapping(exc: RegistryError) -> HTTPException:
    """The status and prose for a registry error. Private: everything outside
    this module goes through `registry_http_error` or `registry_error_body`,
    which attach the code."""
    if isinstance(exc, ModelNotFound):
        return HTTPException(status.HTTP_404_NOT_FOUND, detail=f"model not found: {exc}")
    if isinstance(exc, VersionNotReady):
        return HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"model version is not ready in the registry: {exc}",
        )
    if isinstance(exc, VersionFailed):
        return HTTPException(
            status.HTTP_409_CONFLICT,
            detail=(
                f"the registry could not build this model version: {exc}. "
                "It will not become ready; register a new version."
            ),
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


# -- the single wire shape -------------------------------------------------

# Codes for the errors that are not registry errors and so have no exception
# class to name. A slug rather than the bare number because the number is
# already in the status line, and because "conflict" survives a route deciding
# 409 was the wrong status better than "http_409" would.
#
# Written as bare integers, unlike the rest of this module: three of the
# `status.HTTP_*` constants for these numbers are deprecated in Starlette and
# merely naming them here emitted a DeprecationWarning at import. In a
# number -> slug lookup the number is also the clearer key.
_STATUS_CODES = {
    400: "bad_request",
    401: "unauthenticated",
    403: "forbidden",
    404: "not_found",
    409: "conflict",
    412: "precondition_failed",
    413: "payload_too_large",
    416: "range_not_satisfiable",
    422: "validation_failed",
    429: "rate_limited",
    500: "internal_error",
    502: "upstream_error",
    503: "unavailable",
}


def code_for_status(status_code: int) -> str:
    """A stable code for an error raised as a plain status with prose."""
    return _STATUS_CODES.get(status_code, f"http_{status_code}")


def error_body(
    code: str, message: str, detail: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The only error body this API serves.

    Built through `ErrorResponse` rather than as a literal dict so that the
    contract the agent is compiled against is the contract the wire carries --
    a field renamed in contracts fails here instead of silently serving a shape
    nothing reads.
    """
    return ErrorResponse(code=code, message=message, detail=detail).model_dump(mode="json")


def registry_error_body(exc: RegistryError) -> tuple[int, dict[str, Any], dict[str, str] | None]:
    """Status, body and headers for a registry error, by the one mapping above.

    Both the explicit route path and the `main.py` backstop handler go through
    here, which is what makes a given exception look the same on the wire
    whichever way it got out.
    """
    mapped = _registry_mapping(exc)
    body = error_body(type(exc).__name__, str(mapped.detail))
    return mapped.status_code, body, mapped.headers


def registry_http_error(exc: RegistryError) -> HTTPException:
    """The exception a route raises for a registry error.

    The detail is the envelope itself rather than prose, which is the whole
    point: `main.py`'s handler passes a dict detail through untouched, so a
    route raising this produces byte-identical output to the same exception
    reaching the backstop handler on its own. Before, the route path lost the
    code and reported a generic `conflict`.
    """
    status_code, body, headers = registry_error_body(exc)
    return HTTPException(status_code, detail=body, headers=headers)
