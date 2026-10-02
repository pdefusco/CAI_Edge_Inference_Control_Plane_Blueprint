"""Authentication boundaries (spec SS17).

These matter more here than in a typical app. Reaching a CAI Application from a
home network means enabling *unauthenticated* platform access, which removes every
platform-level check -- so these dependencies are the only thing standing between
the public internet and a route that can revoke models across a fleet.

The split being asserted: an operator writes desired state and may never write
actual state; a device writes actual state and may never write desired state.
"""

from __future__ import annotations

import pytest
from conftest import ADMIN_TOKEN, DEVICE_ID
from lighthouse.services import SessionStore
from starlette.testclient import TestClient


# --------------------------------------------------------------------------
# The operator surface refuses anonymous callers
# --------------------------------------------------------------------------


def test_listing_devices_requires_an_operator(client):
    assert client.get("/api/v1/devices").status_code == 401


def test_registering_a_device_requires_an_operator(client):
    response = client.post("/api/v1/devices", json={"device_id": "sneaky"})

    assert response.status_code == 401


def test_revoking_requires_an_operator(client, device):
    """The single most destructive route in the system."""
    assert client.post(f"/api/v1/devices/{DEVICE_ID}/revoke").status_code == 401


def test_a_wrong_admin_token_is_rejected(client):
    response = client.get("/api/v1/devices", headers={"X-Lighthouse-Admin-Token": "lha_wrong"})

    assert response.status_code == 401


def test_the_admin_token_is_accepted_as_a_bearer(client):
    """CML's ingress may not forward custom headers to an Application. If it does
    not, this fallback is the only way the admin surface stays reachable at all."""
    response = client.get("/api/v1/devices", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})

    assert response.status_code == 200


# --------------------------------------------------------------------------
# The dashboard session is a credential in its own right
# --------------------------------------------------------------------------


@pytest.fixture
def dashboard(app):
    """A client that reaches the app on loopback, as a browser does under `make dev`.

    The default `client` fixture is served at `http://testserver`, and the session
    cookie is set `Secure` for any non-loopback host -- so httpx rightly declines to
    send it back over plain HTTP and every session test would fail on the transport
    rather than on the thing it is asserting. `test_the_cookie_is_secure_off_loopback`
    pins that behaviour deliberately; these tests need a client it does not break.
    """
    with TestClient(app, base_url="http://127.0.0.1") as test_client:
        yield test_client


def sign_in(client) -> str:
    """Exchange the admin token for a session, returning the cookie's value."""
    assert client.post("/api/v1/session", json={"token": ADMIN_TOKEN}).status_code == 204
    cookie = client.cookies.get("lh_session")
    assert cookie
    return cookie


def test_the_cookie_is_secure_off_loopback(client):
    """`Secure` everywhere except loopback.

    A CAI Application terminates TLS at the ingress, so the app itself may well see
    plain HTTP on a genuinely-HTTPS request -- which is why this is decided by the
    host rather than by the scheme the app observes. Defaulting to `Secure` for
    anything that is not loopback means a misconfigured deployment fails visibly,
    with a cookie the browser declines to send, rather than silently shipping an
    operator credential in clear text.
    """
    response = client.post("/api/v1/session", json={"token": ADMIN_TOKEN})

    assert "Secure" in response.headers["set-cookie"]
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "SameSite=strict" in response.headers["set-cookie"]


def test_the_cookie_is_not_secure_on_loopback(dashboard):
    """`make dev` is plain HTTP, and a Secure cookie there would never be sent."""
    response = dashboard.post("/api/v1/session", json={"token": ADMIN_TOKEN})

    assert "Secure" not in response.headers["set-cookie"]
    assert "HttpOnly" in response.headers["set-cookie"]


def test_the_session_cookie_is_accepted(dashboard):
    """What the dashboard uses, so the token never sits in localStorage or a URL."""
    sign_in(dashboard)

    assert dashboard.get("/api/v1/devices").status_code == 200


def test_the_session_cookie_is_not_the_admin_token(dashboard):
    """The cookie must be its own secret.

    If it were the admin token verbatim -- which it was -- then logout could only
    ask the browser to forget it, `max_age` would be advisory, and every value ever
    captured from a cookie jar or a proxy log would stay a working root credential
    until someone rotated `LIGHTHOUSE_ADMIN_TOKEN` by hand. On the publicly
    reachable, platform-unauthenticated host this is heading for, that is the worst
    credential to hand to a browser.
    """
    cookie = sign_in(dashboard)

    assert cookie != ADMIN_TOKEN
    assert ADMIN_TOKEN not in cookie
    assert cookie.startswith("lhs_")


def test_signing_out_invalidates_the_session_server_side(dashboard):
    """Logout has to mean something to the server, not just to the browser.

    The cookie is re-set by hand after `DELETE` precisely to defeat the browser's
    cooperation: this asserts the *server* refuses it, which is the only version of
    sign-out that protects an operator on a shared machine.
    """
    cookie = sign_in(dashboard)
    assert dashboard.get("/api/v1/devices").status_code == 200

    assert dashboard.delete("/api/v1/session").status_code == 204

    dashboard.cookies.set("lh_session", cookie)
    assert dashboard.get("/api/v1/devices").status_code == 401


def test_an_expired_session_is_refused(dashboard, app):
    """Expiry is enforced where it counts: by the server, on every request.

    The store is swapped for one on a clock this test controls, rather than the
    test reaching into its internals -- the point being asserted is the route's
    behaviour, and it should hold for any store that reports a session as expired.
    """
    now = [1_000.0]
    app.state.ctx.sessions = SessionStore(ttl_seconds=60, clock=lambda: now[0])

    cookie = sign_in(dashboard)
    assert dashboard.get("/api/v1/devices").status_code == 200

    now[0] += 61

    dashboard.cookies.set("lh_session", cookie)
    assert dashboard.get("/api/v1/devices").status_code == 401


def test_one_session_does_not_end_another(dashboard, app):
    """Two browsers, two sessions: signing out of one leaves the other working."""
    first = sign_in(dashboard)
    second = sign_in(dashboard)
    assert first != second

    dashboard.cookies.set("lh_session", first)
    assert dashboard.delete("/api/v1/session").status_code == 204

    dashboard.cookies.set("lh_session", second)
    assert dashboard.get("/api/v1/devices").status_code == 200


def test_a_forged_session_cookie_is_refused(dashboard):
    dashboard.cookies.set("lh_session", "lhs_" + "a" * 43)

    assert dashboard.get("/api/v1/devices").status_code == 401


def test_a_session_secret_is_not_accepted_as_a_bearer_token(dashboard):
    """A session is a cookie credential only.

    Accepting it as a bearer would widen it into a general-purpose API token --
    reachable from a client that scraped a cookie jar, and no longer protected by
    `SameSite=strict`.
    """
    cookie = sign_in(dashboard)
    dashboard.cookies.clear()

    response = dashboard.get("/api/v1/devices", headers={"Authorization": f"Bearer {cookie}"})

    assert response.status_code == 401


def test_a_wrong_token_mints_no_session(dashboard, app):
    assert dashboard.post("/api/v1/session", json={"token": "lha_wrong"}).status_code == 401
    assert app.state.ctx.sessions.active() == 0


def test_401_advertises_the_bearer_scheme(client):
    response = client.get("/api/v1/devices")

    assert response.headers.get("www-authenticate") == "Bearer"


# --------------------------------------------------------------------------
# The two schemes are disjoint
# --------------------------------------------------------------------------


def test_a_device_token_cannot_act_as_an_operator(client, device):
    """A device token is installed on hardware that can be physically stolen. If
    it satisfied an operator check, stealing one Jetson would mean control of the
    whole fleet."""
    response = client.get("/api/v1/devices", headers={"Authorization": f"Bearer {device}"})

    assert response.status_code == 401


def test_an_operator_token_cannot_post_a_heartbeat(admin, device):
    """Actual state is the one thing in this system that must come from the
    device. An operator who could forge a heartbeat could make the governance view
    say anything at all, which would make the whole demo a puppet show.
    """
    response = admin.post(
        f"/api/v1/devices/{DEVICE_ID}/heartbeat",
        json={
            "device_id": DEVICE_ID,
            "timestamp": "2026-10-01T00:00:00Z",
            "observed_generation": 99,
            "actual_state": "RUNNING",
        },
    )

    assert response.status_code == 401


def test_an_operator_token_cannot_read_desired_state(admin):
    assert admin.get(f"/api/v1/devices/{DEVICE_ID}/desired-state").status_code == 401


# --------------------------------------------------------------------------
# Device tokens are bound to their device, server-side
# --------------------------------------------------------------------------


def test_a_device_token_reaches_its_own_desired_state(agent):
    response = agent.get(f"/api/v1/devices/{DEVICE_ID}/desired-state")

    assert response.status_code == 200
    assert response.json()["device_id"] == DEVICE_ID


def test_a_device_token_cannot_read_another_devices_state(admin, agent):
    """403, not 401: the credential is perfectly valid, it is the resource that is
    wrong. Collapsing the two would hide a real misconfiguration -- two devices
    sharing a token -- inside generic auth noise.
    """
    admin.post("/api/v1/devices", json={"device_id": "other-device"})

    response = agent.get("/api/v1/devices/other-device/desired-state")

    assert response.status_code == 403


def test_a_heartbeat_body_cannot_claim_another_identity(agent):
    """The body's device_id is validated against the authenticated identity and
    never establishes it. Otherwise one compromised device could rewrite the
    entire fleet's actual state.
    """
    response = agent.post(
        f"/api/v1/devices/{DEVICE_ID}/heartbeat",
        json={
            "device_id": "some-other-device",
            "timestamp": "2026-10-01T00:00:00Z",
            "observed_generation": 1,
            "actual_state": "RUNNING",
        },
    )

    assert response.status_code == 400


def test_a_malformed_token_is_rejected(client):
    for bad in ("", "lhd_", "lhd_nodot", "lhd_id.", "garbage", "lhd_" + "a" * 200):
        response = client.get(
            f"/api/v1/devices/{DEVICE_ID}/desired-state",
            headers={"Authorization": f"Bearer {bad}"},
        )
        assert response.status_code == 401, bad


def test_a_non_bearer_authorization_header_is_rejected(client):
    response = client.get(
        f"/api/v1/devices/{DEVICE_ID}/desired-state",
        headers={"Authorization": "Basic dXNlcjpwYXNz"},
    )

    assert response.status_code == 401


# --------------------------------------------------------------------------
# Token lifecycle
# --------------------------------------------------------------------------


def test_the_token_secret_is_returned_once_and_never_again(admin, device):
    """Only sha256(secret) is stored. A device view that leaked the secret would
    turn the dashboard into a credential dump."""
    body = admin.get(f"/api/v1/devices/{DEVICE_ID}").text

    secret = device.split(".", 1)[1]
    assert secret not in body


def test_rotation_works_with_no_downtime(app, admin, device):
    """Issue the new token, install it, then revoke the old one. Both must be
    valid in between, or rotation requires taking the device offline."""
    from starlette.testclient import TestClient

    issued = admin.post(f"/api/v1/devices/{DEVICE_ID}/tokens", json={"reason": "rotation"})
    assert issued.status_code == 201
    new_token = issued.json()["token"]
    assert new_token != device

    with TestClient(app) as http:
        url = f"/api/v1/devices/{DEVICE_ID}/desired-state"
        old_ok = http.get(url, headers={"Authorization": f"Bearer {device}"})
        new_ok = http.get(url, headers={"Authorization": f"Bearer {new_token}"})
        assert (old_ok.status_code, new_ok.status_code) == (200, 200)

        old_id = device.split(".", 1)[0].removeprefix("lhd_")
        revoked = admin.delete(f"/api/v1/devices/{DEVICE_ID}/tokens/{old_id}")
        assert revoked.status_code == 204

        assert http.get(url, headers={"Authorization": f"Bearer {device}"}).status_code == 401
        assert http.get(url, headers={"Authorization": f"Bearer {new_token}"}).status_code == 200


def test_revoking_an_unknown_token_is_404(admin, device):
    assert admin.delete(f"/api/v1/devices/{DEVICE_ID}/tokens/nope").status_code == 404


def test_deleting_a_device_invalidates_its_token(app, admin, device):
    from starlette.testclient import TestClient

    assert admin.delete(f"/api/v1/devices/{DEVICE_ID}").status_code == 204

    with TestClient(app) as http:
        response = http.get(
            f"/api/v1/devices/{DEVICE_ID}/desired-state",
            headers={"Authorization": f"Bearer {device}"},
        )

    # Either rejection is defensible; what must not happen is a 200.
    assert response.status_code in (401, 403, 404)


# --------------------------------------------------------------------------
# Fail closed
# --------------------------------------------------------------------------


def test_an_unconfigured_admin_token_closes_the_operator_surface(tmp_path, store):
    """A silently unauthenticated control plane on the public internet is the worst
    available outcome, so a missing admin token refuses rather than waves through.

    `load_settings` already makes this fatal at startup under LIGHTHOUSE_ENV=cai;
    this is the second gate, so a misconfiguration can never read as "no auth
    required".
    """
    from starlette.testclient import TestClient

    from lighthouse.config import Settings
    from lighthouse.main import create_app

    open_settings = Settings(
        env="local", data_dir=tmp_path / "x", registry_impl="fake", admin_token=None
    )
    with TestClient(create_app(open_settings, store=store)) as http:
        response = http.get("/api/v1/devices")

    assert response.status_code == 503
