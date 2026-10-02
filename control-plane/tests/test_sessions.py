"""The dashboard session store.

Tested directly, on a controlled clock, because every property here is about time
or about what is *not* retained -- neither of which is observable through a route
without either sleeping or reading internals. `test_auth.py` covers the HTTP
behaviour these properties produce.
"""

from __future__ import annotations

from lighthouse.services import SESSION_PREFIX, SessionStore


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_a_minted_session_verifies() -> None:
    store = SessionStore()

    assert store.verify(store.create()) is True


def test_secrets_are_prefixed_and_unique() -> None:
    store = SessionStore()

    secrets = {store.create() for _ in range(50)}

    assert len(secrets) == 50
    assert all(secret.startswith(SESSION_PREFIX) for secret in secrets)


def test_the_secret_itself_is_never_stored() -> None:
    """Only the digest is kept.

    The same reasoning as device tokens: a process dump, a core file, or a future
    decision to persist this table should yield nothing that can be replayed.
    """
    store = SessionStore()
    secret = store.create()

    assert secret not in store._expiry
    assert all(secret not in key for key in store._expiry)


def test_a_session_expires_on_its_own() -> None:
    clock = Clock()
    store = SessionStore(ttl_seconds=100, clock=clock)
    secret = store.create()

    clock.advance(99)
    assert store.verify(secret) is True

    clock.advance(2)
    assert store.verify(secret) is False


def test_an_expired_session_is_dropped_on_contact() -> None:
    """Reaped by use, not only by a periodic sweep.

    Without this, a store that nothing else touches would hold an expired record
    indefinitely -- harmless for auth, which checks the expiry anyway, but it means
    `active()` lies and the eviction bound counts dead sessions.
    """
    clock = Clock()
    store = SessionStore(ttl_seconds=10, clock=clock)
    secret = store.create()
    clock.advance(11)

    store.verify(secret)

    assert store.active() == 0


def test_destroy_ends_exactly_one_session() -> None:
    store = SessionStore()
    first, second = store.create(), store.create()

    assert store.destroy(first) is True
    assert store.verify(first) is False
    assert store.verify(second) is True


def test_destroying_an_unknown_session_is_not_an_error() -> None:
    """Logout is unauthenticated and idempotent: a double-click, a stale cookie and
    a forged one all have to be survivable without a 500."""
    store = SessionStore()

    assert store.destroy("lhs_nonexistent") is False
    assert store.destroy(None) is False
    assert store.destroy("") is False


def test_a_value_without_the_prefix_is_never_a_session() -> None:
    """Belt and braces against the bug this store exists to fix: an admin token
    presented as a cookie must not verify, whatever else is true."""
    store = SessionStore()

    assert store.verify("lha_looks_like_an_admin_token") is False
    assert store.verify("lhd_device.secret") is False


def test_destroy_all_invalidates_everything() -> None:
    store = SessionStore()
    secrets = [store.create() for _ in range(3)]

    assert store.destroy_all() == 3
    assert not any(store.verify(secret) for secret in secrets)


def test_the_store_is_bounded() -> None:
    """A login route is reachable by anyone who holds the admin token, and a script
    looping on it must not grow this dict forever."""
    store = SessionStore(max_sessions=4)

    live = [store.create() for _ in range(10)]

    assert store.active() <= 4
    assert store.verify(live[-1]) is True, "the newest login must survive eviction"
