"""Health, fleet-wide audit, and the dashboard session exchange."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from lighthouse_contracts import AuditEventView, EventType, HealthResponse
from pydantic import BaseModel, ConfigDict

from .. import __version__
from ..util import now_utc
from .auth import SESSION_COOKIE, require_operator
from .deps import AppContext, ctx

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthResponse)
def health(context: AppContext = Depends(ctx)) -> HealthResponse:
    """Unauthenticated liveness.

    Deliberately the only open route: a CAI Application needs something to probe,
    and the fields here are operational facts, not fleet data. `registry_reachable`
    is a real call, because "the process is up but cannot see the registry" is the
    failure an operator actually needs to distinguish.
    """
    return HealthResponse(
        status="ok",
        version=__version__,
        registry=context.registry.name,
        registry_reachable=context.catalog.ping(),
        device_count=context.store.device_count(),
        server_time=now_utc(),
    )


@router.get("/events", response_model=list[AuditEventView])
def list_events(
    limit: int = 100,
    device_id: str | None = None,
    event_type: list[EventType] | None = Query(default=None),
    _operator=Depends(require_operator),
    context: AppContext = Depends(ctx),
) -> list[AuditEventView]:
    """The fleet-wide audit trail, newest first (spec SS16)."""
    return context.audit.list_events(
        device_id=device_id,
        limit=min(limit, 1000),
        event_types=list(event_type) if event_type else None,
    )


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token: str


@router.post("/session", status_code=status.HTTP_204_NO_CONTENT)
def create_session(
    body: LoginRequest,
    request: Request,
    response: Response,
    context: AppContext = Depends(ctx),
) -> Response:
    """Exchange the admin token for a session cookie.

    The dashboard is a browser page, and a token kept in `localStorage` or a query
    string is readable by any injected script and lands in access logs. One
    exchange at login keeps it out of both. `Secure` is set only off localhost so
    `make dev` over plain HTTP still works.

    What the cookie *contains* is a freshly minted session secret, not the admin
    token -- so this route hands out a credential that expires on its own and that
    `DELETE /session` can actually revoke. `services/sessions.py` has the argument.
    """
    import hmac

    expected = context.settings.admin_token
    if not expected or not hmac.compare_digest(body.token, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="invalid operator credential")

    host = (request.headers.get("host") or "").split(":")[0]
    secure = host not in {"localhost", "127.0.0.1", "::1"}
    out = Response(status_code=status.HTTP_204_NO_CONTENT)
    out.set_cookie(
        SESSION_COOKIE,
        context.sessions.create(),
        httponly=True,
        samesite="strict",
        secure=secure,
        # Matched to the store's own TTL so the browser and the server agree on
        # when this ends. The server's copy is the one that is enforced.
        max_age=context.sessions.ttl_seconds,
        path="/",
    )
    return out


@router.delete("/session", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(request: Request, context: AppContext = Depends(ctx)) -> Response:
    """End the session, server-side as well as in the browser.

    Deliberately unauthenticated: presenting the cookie is the whole of the
    argument for being allowed to destroy it, and gating logout behind
    `require_operator` would mean an expired session could not be cleaned up.
    """
    context.sessions.destroy(request.cookies.get(SESSION_COOKIE))
    out = Response(status_code=status.HTTP_204_NO_CONTENT)
    out.delete_cookie(SESSION_COOKIE, path="/")
    return out
