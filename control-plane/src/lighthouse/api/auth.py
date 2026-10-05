"""Authentication (spec SS17).

Two **disjoint** schemes, distinguished by token prefix:

| Principal | Credential | Reach |
| --- | --- | --- |
| Device | `Authorization: Bearer lhd_<token_id>.<secret>` | its own desired-state, heartbeat and artifact, nothing else |
| Operator | `X-Lighthouse-Admin-Token`, `Authorization: Bearer lha_…`, or the `lh_session` cookie | everything |

Three things here are load-bearing:

**Device binding is server-side.** The dependency resolves token -> device_id from
the stored row and asserts it matches the path parameter. The `device_id` in a
heartbeat body is validated against that, never trusted to establish identity --
otherwise any enrolled device could overwrite any other device's actual state and
the governance view would be fiction.

**401 and 403 mean different things.** A bad credential is 401; a *valid* device
token reaching another device's resource is 403. Collapsing them would hide a
misconfigured fleet (two devices sharing a token) inside generic auth noise.

**The admin credential is accepted in three places on purpose.** Whether CML's
ingress forwards custom request headers to a CAI Application is unverified, so
relying solely on `X-Lighthouse-Admin-Token` would risk an admin surface that
cannot be reached at all once deployed. `Authorization: Bearer lha_…` is the
fallback, and the cookie is what the dashboard uses so the token never sits in
`localStorage` or a URL.

**The cookie is not the admin token.** It holds a session secret with its own
expiry, revocable on logout, verified against `SessionStore` rather than against
the configured admin token. `services/sessions.py` records what went wrong when it
was the admin token verbatim.
"""

from __future__ import annotations

import hmac
import logging
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Path, Request, status

from ..services import SESSION_PREFIX, DevicePrincipal
from .deps import AppContext, ctx

log = logging.getLogger(__name__)

ADMIN_HEADER = "X-Lighthouse-Admin-Token"
SESSION_COOKIE = "lh_session"
DEVICE_TOKEN_PREFIX = "lhd_"
ADMIN_TOKEN_PREFIX = "lha_"

# WWW-Authenticate on a 401 keeps the response honest about the scheme, and stops
# httpx/requests users from guessing.
_BEARER = {"WWW-Authenticate": "Bearer"}


@dataclass(frozen=True, slots=True)
class OperatorPrincipal:
    """An authenticated operator. There is one role in the MVP; the type exists so
    adding named operators later does not change a single route signature."""

    subject: str = "operator"


def _bearer(request: Request) -> str | None:
    header = request.headers.get("authorization")
    if not header:
        return None
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer" or not value:
        return None
    return value.strip()


def require_operator(
    request: Request,
    context: AppContext = Depends(ctx),
) -> OperatorPrincipal:
    """Gate the operator surface: deploy, stop, revoke, enrollment, audit.

    Refuses when no admin token is configured. `load_settings` already makes that
    fatal at startup under `LIGHTHOUSE_ENV=cai`; this is the second gate, so a
    misconfiguration can never read as "no auth required".
    """
    expected = context.settings.admin_token
    if not expected:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="operator authentication is not configured",
        )

    # The cookie is checked first and on its own terms. It carries a *session*
    # secret, not the admin token, so it is verified against the session store and
    # never compared against `expected` -- see services/sessions.py for why the
    # cookie used to be the admin token and must not be again.
    session = request.cookies.get(SESSION_COOKIE)
    if session and context.sessions.verify(session):
        return OperatorPrincipal()

    presented = request.headers.get(ADMIN_HEADER) or _bearer(request)
    if not presented:
        # An expired or forged cookie reaches here and reports the same thing as
        # no credential at all, which is what the dashboard needs: its 401 handler
        # returns the operator to the sign-in gate either way.
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail="operator credential required", headers=_BEARER
        )
    # Neither a device token nor a session secret may satisfy an operator check,
    # even by accident. A session secret presented as a bearer token is rejected
    # here rather than quietly working: it arrives only from a client that read it
    # out of a cookie jar, which is not a flow this API supports.
    if (
        presented.startswith(DEVICE_TOKEN_PREFIX)
        or presented.startswith(SESSION_PREFIX)
        or not hmac.compare_digest(presented, expected)
    ):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail="invalid operator credential", headers=_BEARER
        )
    return OperatorPrincipal()


def is_operator(request: Request, context: AppContext = Depends(ctx)) -> bool:
    """Whether the caller holds an operator credential, without refusing if not.

    For a route that must stay open to anyone but should say *more* to an
    operator -- `/health`, which a CAI Application needs unauthenticated for
    probing, yet which used to hand `device_count` to every anonymous caller on
    a public URL.

    Deliberately implemented by calling `require_operator` and catching, rather
    than by re-checking the cookie and the header here. Two copies of this logic
    is how one of them ends up accepting a device token: the rules about
    `DEVICE_TOKEN_PREFIX`, `SESSION_PREFIX` and `compare_digest` above are
    security-relevant and must have exactly one home. The cost is an exception on
    the anonymous path, which is not a hot path.
    """
    try:
        require_operator(request, context)
    except HTTPException:
        return False
    return True


def require_device(
    request: Request,
    device_id: str = Path(...),
    context: AppContext = Depends(ctx),
) -> DevicePrincipal:
    """Gate the device surface, bound to the device in the path.

    The operator credential deliberately does **not** satisfy this: letting it
    through would mean an operator could post heartbeats and fabricate actual
    state, which is the one thing in this system that must come from the device.
    """
    presented = _bearer(request)
    if not presented:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail="device token required", headers=_BEARER
        )

    principal = context.devices.authenticate(presented)
    if principal is None:
        log.info("rejected device token for %s", device_id)
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail="invalid device token", headers=_BEARER
        )
    if principal.device_id != device_id:
        # Valid credential, wrong resource: a real signal, not noise.
        log.warning(
            "device %s presented a token bound to %s", device_id, principal.device_id
        )
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, detail="token is not valid for this device"
        )
    return principal
