"""Dashboard sessions: short-lived credentials derived from the admin token.

The point of this file is that the session cookie must **not** be the admin token.
An earlier version set `lh_session` to the admin token verbatim, which looked fine
-- HttpOnly, SameSite=strict, never in `localStorage` -- and had two consequences
that only show up when you ask what logout does:

1. **Sign-out was cosmetic.** `delete_cookie` is a request to the browser. The
   server had nothing to forget, so a cookie value captured from a proxy log, a
   cookie jar on disk or a backed-up browser profile stayed a working *root*
   credential until someone rotated `LIGHTHOUSE_ADMIN_TOKEN` by hand.
2. **`max_age` was cosmetic too**, for the same reason: the browser forgets the
   cookie, the server would still have honoured it.

That is the wrong shape to carry into the deployment this project is heading for.
Spec SS17 plus the plan's M5 enable *unauthenticated access* on the CAI
Application, which makes Lighthouse's own credentials the only gate on a publicly
reachable host. On such a host a bearer of the root credential that cannot be
revoked and does not expire is the single worst thing to hand out, and the
dashboard was handing one to every browser that signed in.

So a session is now its own secret with its own lifetime, exchanged for the admin
token once and revocable on its own:

* The cookie carries `lhs_<secret>`, distinct from `lha_` so no comparison can
  ever confuse a session for the admin token.
* Only `sha256(secret)` is stored, for the same reason device tokens are stored
  that way: a dump of this process's memory or of a future persisted table yields
  nothing usable. Plain SHA-256 is right here and not a shortcut -- these are
  256-bit CSPRNG secrets, not passwords, so a slow KDF would defend against a
  guessing attack that is already infeasible while adding CPU to every request.
* Expiry is checked server-side on every use.
* Logout deletes the record, so it means what it says.

Sessions live in memory and die with the process, which is deliberate rather than
a limitation: restarting the control plane *should* invalidate every dashboard
session, and persisting them would add a table whose only content is credentials.
It costs an operator one re-login after a deploy and removes a whole class of
stale-credential problem. A multi-replica Lighthouse would need a shared store --
noted here because that is the day this decision would have to change, and the
CAI Application model is single-process.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import threading
import time
from typing import Callable

log = logging.getLogger(__name__)

SESSION_PREFIX = "lhs_"

# Long enough to get through a working day of watching a fleet converge, short
# enough that an abandoned browser is not an indefinite liability.
DEFAULT_TTL_SECONDS = 12 * 60 * 60


def _digest(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


class SessionStore:
    """Opaque dashboard sessions, keyed by the digest of their secret."""

    def __init__(
        self,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        max_sessions: int = 256,
    ) -> None:
        self._ttl = ttl_seconds
        self._clock = clock
        self._max = max_sessions
        self._lock = threading.Lock()
        self._expiry: dict[str, float] = {}

    @property
    def ttl_seconds(self) -> int:
        return self._ttl

    def create(self) -> str:
        """Mint a session and return its secret. The secret is never stored."""
        secret = SESSION_PREFIX + secrets.token_urlsafe(32)
        now = self._clock()
        with self._lock:
            self._purge_locked(now)
            # A bound on concurrent sessions, so a script hammering the login
            # route cannot grow this dict without limit. Evicting the soonest to
            # expire keeps the freshest logins alive.
            if len(self._expiry) >= self._max:
                oldest = min(self._expiry, key=lambda key: self._expiry[key])
                self._expiry.pop(oldest, None)
            self._expiry[_digest(secret)] = now + self._ttl
        return secret

    def verify(self, secret: str | None) -> bool:
        """Is this an unexpired session?

        Comparison is a dict lookup on the *digest*, which is constant-time with
        respect to the secret in the way that matters: no byte-by-byte comparison
        against a stored value happens anywhere, so there is no prefix to time.
        """
        if not secret or not secret.startswith(SESSION_PREFIX):
            return False
        now = self._clock()
        key = _digest(secret)
        with self._lock:
            expires = self._expiry.get(key)
            if expires is None:
                return False
            if expires <= now:
                # Expired sessions are reaped on contact as well as in bulk, so a
                # single long-lived session cannot outlive its TTL just because
                # nothing else triggered a purge.
                del self._expiry[key]
                return False
            return True

    def destroy(self, secret: str | None) -> bool:
        """End one session. True if there was one to end."""
        if not secret or not secret.startswith(SESSION_PREFIX):
            return False
        with self._lock:
            return self._expiry.pop(_digest(secret), None) is not None

    def destroy_all(self) -> int:
        """Invalidate every session. Not wired to a route yet; this is what an
        admin-token rotation should call, and it is one line when that lands."""
        with self._lock:
            count = len(self._expiry)
            self._expiry.clear()
        return count

    def active(self) -> int:
        now = self._clock()
        with self._lock:
            self._purge_locked(now)
            return len(self._expiry)

    def _purge_locked(self, now: float) -> None:
        for key in [key for key, expires in self._expiry.items() if expires <= now]:
            del self._expiry[key]


__all__ = ["DEFAULT_TTL_SECONDS", "SESSION_PREFIX", "SessionStore"]
