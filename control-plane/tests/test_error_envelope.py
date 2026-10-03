"""One error shape, whichever path the error took.

There were three. A route that caught `RegistryError` and raised
`registry_http_error(exc)` served `{"detail": "<prose>"}`; the *same exception*
reaching the backstop handler in `main.py` served
`{"code": ..., "message": ...}`; and `ErrorResponse` in contracts declared a
third shape -- with a docstring promising "the agent can branch on a stable code
rather than parsing prose" -- that had no producer anywhere in the repo. So the
agent guessed at both live shapes (`keeper/client.py:239-246` reads `message`
then `detail` then stringifies the whole body) and branched on HTTP status
instead, which is the only thing it could rely on.

These tests pin the collapse. The load-bearing ones are not the "does it have
the right keys" checks but these three:

  * **Both paths are byte-identical.** That is the defect, stated directly: a
    client could not learn an error's code without knowing which internal call
    path produced it. Asserted on the wire, not on the helper functions, since
    the helpers agreeing proves nothing about what FastAPI actually serialized.
  * **Headers survive the re-enveloping.** Replacing a response body is an easy
    place to drop its headers, and two here are load-bearing: `WWW-Authenticate`
    on a 401 is what returns the dashboard to its sign-in gate, and `Retry-After`
    on a 503 is the server-chosen backoff that keeps a fleet from each picking
    its own.
  * **The body validates as the contract.** `error_body` builds through
    `ErrorResponse`, so a field renamed in contracts fails here instead of
    quietly serving a shape nothing reads -- which is exactly how the third
    shape came to exist.

`detail` holding an object rather than prose is the reason `static/app.js` had to
change in the same commit: `body.detail || body.message` would have rendered
`[object Object]` into the operator's error banner.
"""

from __future__ import annotations

import pytest
from lighthouse_contracts import ErrorResponse

from lighthouse.api.errors import code_for_status, error_body, registry_error_body
from lighthouse.registry import (
    ModelNotFound,
    RegistryAuthError,
    RegistryUnavailable,
    UnsupportedFlavor,
    VersionFailed,
    VersionNotReady,
)

from conftest import DEVICE_ID


def _deployment(version: str = "3") -> dict:
    return {"model_name": "fashion-cnn", "model_version": version}


# --------------------------------------------------------------------------
# The envelope itself
# --------------------------------------------------------------------------


def test_every_error_body_is_the_declared_contract():
    """`error_body` goes through `ErrorResponse` rather than building a literal
    dict, so contracts and the wire cannot drift. This asserts the round trip
    both ways: a renamed field fails to construct, and an extra one fails to
    validate."""
    body = error_body("not_found", "no such device")

    assert set(body) == {"code", "message", "detail"}
    assert ErrorResponse.model_validate(body).code == "not_found"


def test_the_prose_is_the_message_and_never_the_detail():
    """The agent and the dashboard both used to read `detail` as a string. It is
    now an object or null, which is why both consumers had to be taught to prefer
    `message` -- a string check left in place of this would have shipped
    `[object Object]` to an operator."""
    body = error_body("conflict", "model version is not ready")

    assert body["message"] == "model version is not ready"
    assert body["detail"] is None


def test_a_status_with_no_exception_class_still_gets_a_stable_code():
    """Most of the error surface is `raise HTTPException(404, detail=prose)` with
    no registry exception behind it. Those need a code too, or the uniform shape
    is uniform in keys only."""
    assert code_for_status(404) == "not_found"
    assert code_for_status(503) == "unavailable"


def test_an_unmapped_status_is_named_rather_than_left_blank():
    """A route inventing a status this module has not met must still produce
    something a client can branch on, not an empty string."""
    assert code_for_status(418) == "http_418"


# --------------------------------------------------------------------------
# The defect: the same error looked different depending on the call path
# --------------------------------------------------------------------------


def test_a_registry_error_is_identical_whichever_path_it_took(app, admin, device):
    """The whole point of the commit. `PUT .../deployment` catches
    `UnsupportedFlavor` and raises it as an HTTPException; the backstop handler
    in `main.py` would have produced the same error on its own. Both must now
    serialize to the same bytes, because a client cannot know which path ran."""
    app.state.ctx.registry.no_onnx_flavor.add(("fashion-cnn", "3"))

    response = admin.put(f"/api/v1/devices/{DEVICE_ID}/deployment", json=_deployment())

    _, handler_body, _ = registry_error_body(UnsupportedFlavor("fashion-cnn:3"))
    assert set(response.json()) == set(handler_body)
    assert response.json()["code"] == handler_body["code"]


def test_the_route_path_reports_the_exception_class_not_a_generic_conflict(app, admin, device):
    """Before this change the route path lost the class name and reported
    `conflict`, so `UnsupportedFlavor` (never retry, re-export the model) was
    indistinguishable on the wire from `VersionNotReady` (retry, it is still
    building). Those carry opposite advice."""
    app.state.ctx.registry.no_onnx_flavor.add(("fashion-cnn", "3"))

    response = admin.put(f"/api/v1/devices/{DEVICE_ID}/deployment", json=_deployment())

    assert response.status_code == 409
    assert response.json()["code"] == "UnsupportedFlavor"


@pytest.mark.parametrize(
    "exc, expected_status",
    [
        (ModelNotFound("fashion-cnn"), 404),
        (VersionNotReady("fashion-cnn:3"), 409),
        (VersionFailed("fashion-cnn:3"), 409),
        (UnsupportedFlavor("fashion-cnn:3"), 409),
        (RegistryAuthError("rejected"), 502),
        (RegistryUnavailable("down"), 503),
    ],
)
def test_each_registry_error_names_itself_in_the_code(exc, expected_status):
    """Parametrized over the mapping rather than spot-checked: the codes are the
    stable identifiers a client is being invited to branch on, so one of them
    silently collapsing into another is the failure mode worth covering."""
    status_code, body, _ = registry_error_body(exc)

    assert status_code == expected_status
    assert body["code"] == type(exc).__name__
    # RegistryAuthError is the deliberate exception: its message must not carry
    # the exception text, because that is the control plane's own credential.
    if not isinstance(exc, RegistryAuthError):
        assert str(exc) in body["message"]


def test_the_registry_credential_is_still_kept_out_of_the_message():
    """`RegistryAuthError` is the control plane's *own* credential being
    rejected, and `errors.py` deliberately omits the exception from the detail.
    Enveloping must not have reintroduced it -- an operator being told to
    re-authenticate here would be chasing the wrong credential entirely."""
    _, body, _ = registry_error_body(RegistryAuthError("token eyJhbGciOi.secret.sig"))

    assert "eyJ" not in body["message"]
    assert "secret" not in body["message"]


# --------------------------------------------------------------------------
# Headers are part of the response, and replacing a body can drop them
# --------------------------------------------------------------------------


def test_an_unauthenticated_request_keeps_its_challenge_header(client):
    """`WWW-Authenticate` is what sends the dashboard back to its sign-in gate.
    Re-enveloping the body is an easy place to lose it, and losing it would leave
    the operator looking at a dead page."""
    response = client.get("/api/v1/devices")

    assert response.status_code == 401
    assert response.headers.get("WWW-Authenticate") == "Bearer"
    assert response.json()["code"] == "unauthenticated"


def test_an_unavailable_registry_keeps_its_backoff_header(app, admin, device):
    """`Retry-After` is a server-chosen backoff; without it every device in the
    fleet picks its own and they synchronize. 503 bodies are the ones most likely
    to be produced by a handler rather than a route, so this is where a dropped
    header would hide."""
    app.state.ctx.registry.unavailable = True

    response = admin.put(f"/api/v1/devices/{DEVICE_ID}/deployment", json=_deployment("1"))

    assert response.status_code == 503
    assert response.headers.get("Retry-After")
    assert response.json()["code"] == "RegistryUnavailable"


# --------------------------------------------------------------------------
# Validation errors: FastAPI's own shape, which was the fourth one
# --------------------------------------------------------------------------


def test_a_malformed_request_body_is_enveloped_like_everything_else(admin):
    """FastAPI serves its own `{"detail": [ ... ]}` for a 422, so without the
    handler the strictest and most commonly hit error in the API would be the one
    shape that escaped the collapse."""
    response = admin.post("/api/v1/devices", json={"bogus": 1})

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_failed"
    assert ErrorResponse.model_validate(body)


def test_a_rejected_field_is_still_named_in_the_detail(admin):
    """The envelope's `detail` is an object and pydantic's errors are a list, so
    they go under a key rather than being summarized away. For a contract this
    strict, which field was rejected is the entire useful content of the
    response."""
    response = admin.post("/api/v1/devices", json={"bogus": 1})

    errors = response.json()["detail"]["errors"]
    assert any("device_id" in str(e.get("loc", "")) for e in errors)


def test_a_pattern_violation_is_enveloped_rather_than_crashing_the_handler(admin):
    """The strictest reachable validation path: `device_id` carries a regex, so
    its rejection puts a `ctx` on the pydantic error. That is the shape most
    likely to break a hand-rolled serializer, and the response must still be the
    envelope rather than a 500 from inside the error handler.

    Note this does *not* prove the `jsonable_encoder` call is load-bearing --
    removing it leaves this suite green, because nothing in contracts uses a
    custom validator yet and `ctx` for a pattern is just strings. It is there for
    parity with FastAPI's own 422 handler, which encodes for the same reason, and
    it starts mattering the moment contracts grows a validator that raises."""
    response = admin.post("/api/v1/devices", json={"device_id": "!!bad id!!"})

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_failed"
    assert any("device_id" in str(e.get("loc", "")) for e in body["detail"]["errors"])
