"""The operator deployment surface and the device-facing read path.

Three things are being pinned down here, and only the first is ordinary CRUD.

**Generation is the only ordering mechanism** (spec SS4: ordering never uses
timestamps), so every desired-state change must advance it -- including the ones
that look like no-ops, because the dashboard and the agent both decide what to do
by comparing numbers.

**Stop and revoke preserve the artifact lineage.** Writing "stop" by clearing
`model_id`/`version_uuid` is the obvious implementation and it breaks SS5 twice
over: the cached bytes stop being referenced and become evictable while a STOPPED
device still needs them to restart, and `REVOKED -> RUNNING` turns into a cold
download of something already on the disk.

**`artifact_ready: false` is a wait state, not a failure.** It is reachable in
production -- `PUT /deployment` returns before the bytes exist, by design -- so it
is reached deterministically here by holding the registry open, rather than hoped
for by racing a background thread.
"""

from __future__ import annotations

import hashlib

from lighthouse_contracts import DesiredState, EventType

from conftest import DEVICE_ID, materialization_held, wait_for_artifact


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def deployment(version: str = "1", *, state: str = "RUNNING", **extra) -> dict:
    body = {"model_name": "fashion-cnn", "model_version": version, "desired_state": state}
    body.update(extra)
    return body


def put(admin, version: str = "1", **kwargs):
    response = admin.put(f"/api/v1/devices/{DEVICE_ID}/deployment", json=deployment(version, **kwargs))
    assert response.status_code == 200, response.text
    return response.json()


def events(admin, device_id: str = DEVICE_ID) -> list[dict]:
    response = admin.get(f"/api/v1/devices/{device_id}/events")
    assert response.status_code == 200, response.text
    return response.json()


def event_types(admin, device_id: str = DEVICE_ID) -> list[str]:
    return [e["event_type"] for e in events(admin, device_id)]


def desired_row(app, device_id: str = DEVICE_ID):
    return app.state.ctx.store.get_desired(device_id)


# --------------------------------------------------------------------------
# PUT /deployment
# --------------------------------------------------------------------------


def test_deploying_assigns_the_model_and_advances_the_generation(admin, device):
    """A freshly enrolled device sits at generation 0 (spec SS19 seeds it), so the
    first deployment is generation 1."""
    body = put(admin)

    assert body["device_id"] == DEVICE_ID
    assert body["generation"] == 1
    assert body["desired_state"] == "RUNNING"
    assert (body["model_name"], body["model_version"]) == ("fashion-cnn", "1")


def test_every_change_advances_the_generation(admin, device):
    """Including a redeployment of the version already desired. The device no-ops
    on it, but the number must still move -- an operator pressing Deploy twice and
    seeing the generation stand still would be indistinguishable from a dropped
    request.
    """
    generations = [put(admin, v)["generation"] for v in ("1", "2", "2", "1")]

    assert generations == [1, 2, 3, 4]


def test_the_deployment_is_what_the_device_will_read(admin, agent, device):
    put(admin, "2")

    state = agent.get(f"/api/v1/devices/{DEVICE_ID}/desired-state").json()

    assert state["generation"] == 1
    assert state["desired_state"] == "RUNNING"


def test_deploying_to_an_unknown_device_is_404(admin):
    response = admin.put("/api/v1/devices/ghost/deployment", json=deployment())

    assert response.status_code == 404


def test_an_unknown_model_is_404_not_a_mystery_on_the_device(app, admin, device):
    """Resolved against the registry before anything is written, so a typo is an
    operator-facing rejection rather than a device that fails to converge."""
    response = admin.put(
        f"/api/v1/devices/{DEVICE_ID}/deployment",
        json={"model_name": "no-such-model", "model_version": "1"},
    )

    assert response.status_code == 404
    assert desired_row(app).generation == 0, "desired state moved on a failure"


def test_an_unknown_version_is_404(admin, device):
    response = admin.put(
        f"/api/v1/devices/{DEVICE_ID}/deployment",
        json=deployment("99"),
    )

    assert response.status_code == 404


def test_a_version_still_building_is_409(app, admin, device):
    """409, not 404: it exists, it is simply not ready. Retrying later is correct,
    so the status has to say "conflict" rather than "gone"."""
    app.state.ctx.registry.not_ready.add(("fashion-cnn", "2"))

    response = admin.put(f"/api/v1/devices/{DEVICE_ID}/deployment", json=deployment("2"))

    assert response.status_code == 409


def test_a_version_with_no_onnx_flavor_is_409(app, admin, device):
    """Rejected at deploy time rather than discovered on the Jetson after a
    download. The device cannot run it and never will, so the operator needs to
    hear it here."""
    app.state.ctx.registry.no_onnx_flavor.add(("fashion-cnn", "3"))

    response = admin.put(f"/api/v1/devices/{DEVICE_ID}/deployment", json=deployment("3"))

    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "UnsupportedFlavor"
    assert "onnx" in body["message"].lower()


def test_an_unreachable_registry_is_503_with_a_backoff(app, admin, device):
    """503 and Retry-After, because this one *is* worth retrying and the server
    should pick the interval rather than every operator script inventing one."""
    app.state.ctx.registry.set_unavailable(True)

    response = admin.put(f"/api/v1/devices/{DEVICE_ID}/deployment", json=deployment())

    assert response.status_code == 503
    assert response.headers.get("retry-after")


def test_the_request_cannot_carry_an_event_type(admin, device):
    """Extra fields are forbidden on the wire (`Strict`), which is what stops an
    operator from declaring "this is a rollback". Classification is the server's
    job and a request that tries to influence it is a loud 422.
    """
    response = admin.put(
        f"/api/v1/devices/{DEVICE_ID}/deployment",
        json=deployment(rollback=True),
    )

    assert response.status_code == 422


# --------------------------------------------------------------------------
# Audit classification -- derived from history, never declared
# --------------------------------------------------------------------------


def test_a_first_deployment_is_a_request(admin, device):
    put(admin)

    assert event_types(admin)[0] == EventType.DEPLOYMENT_REQUESTED


def test_a_different_version_is_a_version_change(admin, device):
    put(admin, "1")

    put(admin, "2")

    assert event_types(admin)[0] == EventType.MODEL_VERSION_CHANGED


def test_returning_to_an_earlier_version_is_a_rollback(admin, device):
    """Rollback is derived, not declared: "a (model, version) pair this device was
    previously told to run, which is not what it is running now". There is no
    rollback endpoint and no rollback state, only this classification.
    """
    put(admin, "1")
    put(admin, "2")

    put(admin, "1")

    assert event_types(admin)[0] == EventType.DEPLOYMENT_ROLLED_BACK


def test_redeploying_the_current_version_is_not_a_rollback(admin, device):
    """The boundary case. v1 is in history *and* currently desired, so a naive
    "have we seen this before" check would label a plain re-assert as a rollback
    and make the audit trail lie."""
    put(admin, "1")

    put(admin, "1")

    assert event_types(admin)[0] == EventType.DEPLOYMENT_REQUESTED


def test_the_audit_event_carries_the_generation_it_belongs_to(admin, device):
    """Without the generation an event cannot be tied to the instruction it
    describes, which is most of what SS16 is for."""
    body = put(admin, "2")

    latest = events(admin)[0]
    assert latest["generation"] == body["generation"]


def test_a_reason_is_preserved(admin, device):
    put(admin, "1", reason="quarterly model refresh")

    assert events(admin)[0]["details"]["reason"] == "quarterly model refresh"


def test_events_for_an_unknown_device_are_404_not_empty(admin):
    """An empty list would read as "this device has done nothing", which is a very
    different claim from "there is no such device"."""
    assert admin.get("/api/v1/devices/ghost/events").status_code == 404


# --------------------------------------------------------------------------
# Stop and revoke
# --------------------------------------------------------------------------


def test_stop_changes_the_state_and_keeps_the_model(admin, device):
    """SS8: `model` is retained for STOPPED. To stop inference the agent has to
    know *which* model to stop."""
    put(admin)

    body = admin.post(f"/api/v1/devices/{DEVICE_ID}/stop").json()

    assert body["desired_state"] == "STOPPED"
    assert body["generation"] == 2
    assert (body["model_name"], body["model_version"]) == ("fashion-cnn", "1")


def test_stop_preserves_the_entire_artifact_lineage(app, admin, device):
    """The sharp edge this whole design note exists for. If stop dropped
    `model_id`/`version_uuid`, `referenced_cache_keys` would stop protecting the
    artifact, eviction would reclaim it, and a STOPPED device could never restart
    from its own local copy.
    """
    put(admin)
    before = desired_row(app)

    admin.post(f"/api/v1/devices/{DEVICE_ID}/stop")

    after = desired_row(app)
    assert after.desired_state is DesiredState.STOPPED
    for field in ("model_name", "model_version", "model_id", "version_uuid", "registry_artifact_uri"):
        assert getattr(after, field) == getattr(before, field), field
    assert after.cache_key == before.cache_key
    assert before.cache_key in app.state.ctx.store.referenced_cache_keys()


def test_revoke_preserves_the_lineage_too(app, admin, device):
    """Revocation is about the *device's* copy. The control plane's cache is shared
    across the fleet, and SS5 explicitly allows a later generation to re-authorize
    deployment -- which should be a cache hit."""
    put(admin)
    before = desired_row(app)

    body = admin.post(f"/api/v1/devices/{DEVICE_ID}/revoke").json()

    assert body["desired_state"] == "REVOKED"
    after = desired_row(app)
    assert after.cache_key == before.cache_key
    assert after.model_version == "1"


def test_revoked_can_be_redeployed_from_cache(app, admin, device):
    """SS5: revocation blocks restart "unless a newer desired-state generation
    explicitly authorizes deployment". A PUT *is* that authorization, and because
    the lineage survived it is a cache hit rather than a cold download.
    """
    put(admin)
    wait_for_artifact(app)
    admin.post(f"/api/v1/devices/{DEVICE_ID}/revoke")

    body = put(admin, "1")

    assert body["desired_state"] == "RUNNING"
    assert body["generation"] == 3
    assert body["artifact_ready"] is True, "re-authorizing after revoke went back to the registry"


def test_stop_and_revoke_each_advance_the_generation(admin, device):
    put(admin)

    stopped = admin.post(f"/api/v1/devices/{DEVICE_ID}/stop").json()["generation"]
    revoked = admin.post(f"/api/v1/devices/{DEVICE_ID}/revoke").json()["generation"]

    assert (stopped, revoked) == (2, 3)


def test_stopping_a_device_with_nothing_deployed_is_409(app, admin, device):
    """Not a silent success. Writing a STOPPED generation with no model attached
    would produce an instruction the agent cannot act on, and would make the
    dashboard claim a model was stopped when none was ever assigned.
    """
    response = admin.post(f"/api/v1/devices/{DEVICE_ID}/stop")

    assert response.status_code == 409
    assert desired_row(app).generation == 0


def test_revoking_a_device_with_nothing_deployed_is_409(admin, device):
    assert admin.post(f"/api/v1/devices/{DEVICE_ID}/revoke").status_code == 409


def test_stopping_an_unknown_device_is_404(admin):
    assert admin.post("/api/v1/devices/ghost/stop").status_code == 404


def test_stop_and_revoke_are_audited_with_their_reasons(admin, device):
    put(admin)

    admin.post(f"/api/v1/devices/{DEVICE_ID}/stop", json={"reason": "bench maintenance"})
    admin.post(f"/api/v1/devices/{DEVICE_ID}/revoke", json={"reason": "model withdrawn"})

    recorded = events(admin)
    assert recorded[0]["event_type"] == EventType.REVOKE_REQUESTED
    assert recorded[0]["details"]["reason"] == "model withdrawn"
    assert recorded[1]["event_type"] == EventType.STOP_REQUESTED
    assert recorded[1]["details"]["reason"] == "bench maintenance"


def test_a_revoked_deployment_still_appears_in_the_operator_view(admin, device):
    """The fleet view must keep showing a revoked device. Hiding it is how a
    revocation that the device never acknowledged goes unnoticed."""
    put(admin)
    admin.post(f"/api/v1/devices/{DEVICE_ID}/revoke")

    view = admin.get(f"/api/v1/devices/{DEVICE_ID}").json()

    assert view["desired_state"] == "REVOKED"
    assert view["desired_model_version"] == "1"


# --------------------------------------------------------------------------
# GET /desired-state -- the SS8 contract
# --------------------------------------------------------------------------


def test_a_device_with_nothing_deployed_is_not_a_wait_state(agent, device):
    """Generation 0, no model, and `artifact_ready: true`. There is genuinely
    nothing to fetch, and reporting false here would park a brand-new device in a
    retry loop forever.
    """
    state = agent.get(f"/api/v1/devices/{DEVICE_ID}/desired-state").json()

    assert state["generation"] == 0
    assert state["model"] is None
    assert state["artifact_ready"] is True
    assert state["desired_state"] == "STOPPED"


def test_the_payload_carries_everything_the_agent_needs(app, admin, agent, device):
    put(admin)
    wait_for_artifact(app)

    state = agent.get(f"/api/v1/devices/{DEVICE_ID}/desired-state").json()

    assert state["artifact_ready"] is True
    model = state["model"]
    assert model["name"] == "fashion-cnn"
    assert model["version"] == "1"
    assert model["format"] == "onnx"
    assert model["packaging"] == "mlflow_tar_gz"
    assert model["entrypoint"] == "model.onnx"
    assert model["size_bytes"] > 0
    assert len(model["sha256"]) == 64
    assert state["poll_interval_seconds"] == app.state.ctx.settings.heartbeat_interval_seconds
    assert state["server_time"]


def test_the_artifact_uri_is_relative_and_pinned_to_the_generation(app, admin, agent, device):
    """Relative so the same payload works through the CAI Application domain, an
    SSH tunnel, or localhost in `make dev`. Carrying the generation is what lets
    the download refuse to serve bytes for a superseded instruction.
    """
    body = put(admin)
    wait_for_artifact(app)

    model = agent.get(f"/api/v1/devices/{DEVICE_ID}/desired-state").json()["model"]

    assert model["artifact_uri"] == (
        f"/api/v1/devices/{DEVICE_ID}/artifact?generation={body['generation']}"
    )
    assert not model["artifact_uri"].startswith("http"), "an absolute URI would pin one hostname"
    assert "s3a://" not in model["artifact_uri"], "the device has no identity in object storage"


def test_the_advertised_digest_is_the_digest_of_the_bytes_served(app, admin, agent, device):
    """The whole point of hashing on ingest. A digest copied from registry metadata
    would make the device's verify-before-activate check a formality."""
    put(admin)
    wait_for_artifact(app)

    model = agent.get(f"/api/v1/devices/{DEVICE_ID}/desired-state").json()["model"]
    served = agent.get(model["artifact_uri"]).content

    assert hashlib.sha256(served).hexdigest() == model["sha256"]
    assert len(served) == model["size_bytes"]


def test_the_model_is_retained_while_stopped(app, admin, agent, device):
    put(admin)
    wait_for_artifact(app)

    admin.post(f"/api/v1/devices/{DEVICE_ID}/stop")

    state = agent.get(f"/api/v1/devices/{DEVICE_ID}/desired-state").json()
    assert state["desired_state"] == "STOPPED"
    assert state["model"]["version"] == "1", "the agent cannot stop a model it was not told about"


def test_the_model_is_retained_while_revoked(app, admin, agent, device):
    """The agent needs to know *whose* artifacts to delete. Nulling the model here
    would leave files on the device with nothing to tie them to."""
    put(admin)
    wait_for_artifact(app)

    admin.post(f"/api/v1/devices/{DEVICE_ID}/revoke")

    state = agent.get(f"/api/v1/devices/{DEVICE_ID}/desired-state").json()
    assert state["desired_state"] == "REVOKED"
    assert state["model"]["version"] == "1"


def test_reading_desired_state_does_not_trigger_a_download(app, admin, agent, device):
    """A poll is a cheap read. With a fleet polling every ten seconds, a read that
    kicked off materialization would turn idle devices into load."""
    put(admin)
    wait_for_artifact(app)
    original = app.state.ctx.registry.open_artifact
    calls = []

    def counting(mv):
        calls.append(mv.cache_key)
        return original(mv)

    app.state.ctx.registry.open_artifact = counting
    try:
        for _ in range(5):
            agent.get(f"/api/v1/devices/{DEVICE_ID}/desired-state")
    finally:
        app.state.ctx.registry.open_artifact = original

    assert calls == []


# --------------------------------------------------------------------------
# artifact_ready: false -- a wait state, reached deterministically
# --------------------------------------------------------------------------


def test_the_deployment_returns_before_the_bytes_exist(app, admin, device):
    """SS13: accept and return, never wait for the device. The operator gets an
    answer immediately and `artifact_ready` tells the truth about why."""
    with materialization_held(app):
        body = put(admin)

        assert body["generation"] == 1
        assert body["artifact_ready"] is False


def test_an_unmaterialized_artifact_is_a_wait_state_not_an_error(app, admin, agent, device):
    """The payload the agent must treat as "re-poll shortly". Setting FAILED here
    would be a self-inflicted outage: nothing is wrong, the control plane is
    simply still fetching.
    """
    with materialization_held(app):
        put(admin)

        state = agent.get(f"/api/v1/devices/{DEVICE_ID}/desired-state").json()

        assert state["generation"] == 1
        assert state["desired_state"] == "RUNNING"
        assert state["artifact_ready"] is False
        # No model block, because there is no digest to promise yet -- and
        # advertising one the device cannot verify against is worse than silence.
        assert state["model"] is None


def test_the_digest_is_backfilled_without_advancing_the_generation(app, admin, agent, device):
    """The deployment was stored with no digest because the bytes did not exist
    yet. Filling it in later must not bump the generation: the instruction has not
    changed, only the control plane's knowledge of it, and a bump would make every
    device re-reconcile for nothing.
    """
    with materialization_held(app):
        put(admin)
        assert desired_row(app).artifact_sha256 is None

    state = agent.get(f"/api/v1/devices/{DEVICE_ID}/desired-state").json()

    assert state["artifact_ready"] is True
    assert desired_row(app).artifact_sha256 == state["model"]["sha256"]
    assert state["generation"] == 1, "backfilling the digest advanced the generation"


def test_the_operator_view_shows_the_artifact_as_not_ready(app, admin, device):
    """So a dashboard can say "fetching" instead of showing a healthy device that
    is in fact waiting on the control plane."""
    with materialization_held(app):
        put(admin)

        view = admin.get(f"/api/v1/devices/{DEVICE_ID}").json()

        assert view["artifact_ready"] is False


# --------------------------------------------------------------------------
# POST /heartbeat -- actual state
# --------------------------------------------------------------------------


def heartbeat(
    *,
    generation: int = 0,
    state: str = "RUNNING",
    device_id: str = DEVICE_ID,
    model: dict | None = None,
    **extra,
) -> dict:
    body = {
        "device_id": device_id,
        "timestamp": "2026-10-01T12:00:00Z",
        "observed_generation": generation,
        "actual_state": state,
    }
    if model is not None:
        body["model"] = model
    body.update(extra)
    return body


def beat(agent, **kwargs):
    response = agent.post(f"/api/v1/devices/{DEVICE_ID}/heartbeat", json=heartbeat(**kwargs))
    assert response.status_code == 200, response.text
    return response.json()


def report_converged(agent):
    """Heartbeat the way a real agent does: echo back the model block it was told
    to run, at the generation it was told to run it.

    `derive_governance` compares desired and actual `(name, version)` *and* digest
    before calling anything HEALTHY -- right version label, different bytes is the
    nastiest case it has to catch -- so a heartbeat that omits the model can never
    be healthy, and a test asserting HEALTHY has to send one.
    """
    state = agent.get(f"/api/v1/devices/{DEVICE_ID}/desired-state").json()
    assert state["model"] is not None, "nothing is deployed yet"
    return beat(
        agent,
        generation=state["generation"],
        state="RUNNING",
        model=state["model"],
        runtime={"inference_running": True},
    )


def test_a_heartbeat_records_actual_state(admin, agent, device):
    beat(agent, generation=1, state="RUNNING")

    view = admin.get(f"/api/v1/devices/{DEVICE_ID}").json()
    assert view["actual_state"] == "RUNNING"
    assert view["observed_generation"] == 1


def test_the_acknowledgement_carries_the_current_desired_generation(admin, agent, device):
    """So an agent that reported generation 1 and hears 2 back can poll
    immediately instead of waiting out its interval. That is what makes an
    operator's click feel instant without shortening the loop.
    """
    put(admin, "1")
    put(admin, "2")

    ack = beat(agent, generation=1)

    assert ack["accepted"] is True
    assert ack["generation"] == 2
    assert ack["desired_state"] == "RUNNING"


def test_a_heartbeat_makes_the_device_online(admin, agent, device):
    """Connectivity is derived from heartbeat age at read time, never stored as a
    boolean -- a stored flag needs a sweeper to ever become false and is wrong for
    as long as the sweeper is behind."""
    before = admin.get(f"/api/v1/devices/{DEVICE_ID}").json()
    assert before["connectivity"] == "NEVER_SEEN"

    beat(agent)

    assert admin.get(f"/api/v1/devices/{DEVICE_ID}").json()["connectivity"] == "ONLINE"


def test_repeated_identical_heartbeats_do_not_grow_the_audit_trail(admin, agent, device):
    """At ten seconds per device this is the difference between an audit table and
    a landfill. Events record *changes*; the heartbeat itself is not news."""
    beat(agent, generation=0, state="IDLE")
    baseline = len(events(admin))

    for _ in range(5):
        beat(agent, generation=0, state="IDLE")

    assert len(events(admin)) == baseline


def test_a_state_change_appends_exactly_one_event(admin, agent, device):
    beat(agent, generation=1, state="DOWNLOADING")
    baseline = len(events(admin))

    beat(agent, generation=1, state="RUNNING")

    recorded = events(admin)
    assert len(recorded) == baseline + 1
    assert recorded[0]["event_type"] == EventType.DEVICE_STATE_CHANGED
    assert recorded[0]["details"]["previous_state"] == "DOWNLOADING"
    assert recorded[0]["details"]["actual_state"] == "RUNNING"


def test_observing_a_new_generation_is_a_change_even_at_the_same_state(admin, agent, device):
    """A device that was RUNNING v1 and is now RUNNING v2 has converged on a new
    instruction. Comparing only `actual_state` would record nothing at all."""
    beat(agent, generation=1, state="RUNNING")
    baseline = len(events(admin))

    beat(agent, generation=2, state="RUNNING")

    assert len(events(admin)) == baseline + 1


def test_a_failure_is_audited_as_a_reconcile_failure(admin, agent, device):
    """Distinct from a plain state change, because this is the one an operator has
    to go and look at -- and the device's own message is the only explanation
    anyone will get."""
    beat(agent, generation=1, state="FAILED", message="sha256 mismatch after download")

    latest = events(admin)[0]
    assert latest["event_type"] == EventType.RECONCILE_FAILED
    assert latest["details"]["message"] == "sha256 mismatch after download"


def test_a_failure_message_reaches_the_operator_view(admin, agent, device):
    beat(agent, generation=1, state="FAILED", message="onnxruntime refused the graph")

    view = admin.get(f"/api/v1/devices/{DEVICE_ID}").json()
    assert view["actual_state"] == "FAILED"
    assert view["message"] == "onnxruntime refused the graph"


def test_hardware_metadata_round_trips(admin, agent, device):
    """Free-form by design (SS7 defers richer telemetry), so an agent adding a
    field must not need a server change.

    This used to assert only the 200, because it could not assert anything else:
    the server stored the dict in `actual_deployment.hardware_json` and no read
    path ever handed it back, so the Phase 6 acceptance fields were reachable
    only from the device's journal or by opening SQLite. The second half of this
    test is the thing that was missing -- every key the device sent, including
    ones no schema names, arriving intact in the operator view.
    """
    sent = {
        "platform": "jetson-orin-nano",
        "gpu_available": True,
        "jetpack": "6.0",
        "tegra_temp_c": 44.5,
        "active_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "smoke_check": "passed",
    }
    response = agent.post(
        f"/api/v1/devices/{DEVICE_ID}/heartbeat", json=heartbeat(hardware=sent)
    )

    assert response.status_code == 200

    view = admin.get(f"/api/v1/devices/{DEVICE_ID}").json()
    assert view["hardware"] == sent
    assert view["acceleration"] == "ACCELERATED"


def test_a_cuda_build_serving_on_the_cpu_is_reported_as_cpu_only(admin, agent, device):
    """The trap §7 of `docs/jetson-setup.md` is written around.

    `providers` says the build *could* use CUDA; `active_providers` says the
    loaded session is not. A read that trusted the first would show a fleet-wide
    silent fallback to the CPU as a fleet on the GPU, which is the most
    expensive possible way for this field to be wrong because it is also the
    most reassuring.
    """
    beat(
        agent,
        generation=1,
        state="RUNNING",
        hardware={
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "active_providers": ["CPUExecutionProvider"],
            "gpu_available": True,
            "smoke_check": "passed",
        },
    )

    view = admin.get(f"/api/v1/devices/{DEVICE_ID}").json()
    assert view["acceleration"] == "CPU_ONLY"


def test_a_device_that_reported_no_hardware_reads_unknown(admin, agent, device):
    beat(agent, generation=1, state="RUNNING")

    view = admin.get(f"/api/v1/devices/{DEVICE_ID}").json()
    assert view["hardware"] == {}
    assert view["acceleration"] == "UNKNOWN"


def test_the_runtime_block_reaches_the_operator_view(admin, agent, device):
    beat(agent, generation=1, state="RUNNING", runtime={"inference_running": True, "pid": 4242})

    assert admin.get(f"/api/v1/devices/{DEVICE_ID}").json()["inference_running"] is True


def test_an_unknown_field_in_a_heartbeat_is_rejected(agent, device):
    """A typo'd field name should be a loud 422, not a value silently discarded
    while the server keeps its default."""
    response = agent.post(
        f"/api/v1/devices/{DEVICE_ID}/heartbeat",
        json=heartbeat(actul_state="RUNNING"),
    )

    assert response.status_code == 422


def test_a_negative_generation_is_rejected(agent, device):
    response = agent.post(
        f"/api/v1/devices/{DEVICE_ID}/heartbeat",
        json=heartbeat(generation=-1),
    )

    assert response.status_code == 422


def test_an_unknown_actual_state_is_rejected(agent, device):
    """The state vocabulary is on the wire (SS4). Accepting an unrecognised value
    would let a buggy agent invent states the governance rules know nothing
    about."""
    response = agent.post(
        f"/api/v1/devices/{DEVICE_ID}/heartbeat",
        json=heartbeat(state="VIBING"),
    )

    assert response.status_code == 422


# --------------------------------------------------------------------------
# Desired and actual, side by side
# --------------------------------------------------------------------------


def test_a_converged_device_is_healthy(app, admin, agent, device):
    put(admin)
    wait_for_artifact(app)

    report_converged(agent)

    view = admin.get(f"/api/v1/devices/{DEVICE_ID}").json()
    assert view["governance_status"] == "HEALTHY"
    assert view["actual_model_version"] == "1"


def test_running_the_right_version_at_the_wrong_generation_is_not_healthy(app, admin, agent, device):
    """Generation is checked before any state comparison. A device running the
    right model because it happened to already have it -- rather than because it
    saw the instruction -- is not converged, and treating it as healthy would hide
    a device that stopped polling.
    """
    put(admin, "1")
    wait_for_artifact(app)
    state = agent.get(f"/api/v1/devices/{DEVICE_ID}/desired-state").json()

    beat(agent, generation=state["generation"] - 1, state="RUNNING", model=state["model"])

    assert admin.get(f"/api/v1/devices/{DEVICE_ID}").json()["governance_status"] == "OUT_OF_SYNC"


def test_stop_shows_stop_pending_before_stopped(app, admin, agent, device):
    """The transition the dashboard must not hide. Between the operator's click and
    the device's acknowledgement the honest reading is "Desired: STOPPED / Actual:
    RUNNING", and snapping straight to STOPPED would claim an outcome nobody has
    confirmed.
    """
    put(admin)
    wait_for_artifact(app)
    report_converged(agent)

    stopped = admin.post(f"/api/v1/devices/{DEVICE_ID}/stop").json()

    mid = admin.get(f"/api/v1/devices/{DEVICE_ID}").json()
    assert mid["desired_state"] == "STOPPED"
    assert mid["actual_state"] == "RUNNING"
    assert mid["governance_status"] == "STOP_PENDING"

    beat(agent, generation=stopped["generation"], state="STOPPED")

    settled = admin.get(f"/api/v1/devices/{DEVICE_ID}").json()
    assert settled["actual_state"] == "STOPPED"
    assert settled["governance_status"] == "HEALTHY"


def test_a_device_lagging_a_generation_is_out_of_sync(app, admin, agent, device):
    put(admin, "1")
    wait_for_artifact(app)
    report_converged(agent)

    put(admin, "2")

    assert admin.get(f"/api/v1/devices/{DEVICE_ID}").json()["governance_status"] == "OUT_OF_SYNC"


def test_revocation_is_not_complete_until_the_device_says_so(app, admin, agent, device):
    """The claim the demo rests on: CAI knows what is *actually* running, not only
    what it asked for. A revoke that reads as REVOKED before the device confirms
    would be exactly the governance fiction this system exists to avoid.
    """
    put(admin)
    wait_for_artifact(app)
    report_converged(agent)

    revoked = admin.post(f"/api/v1/devices/{DEVICE_ID}/revoke").json()

    pending = admin.get(f"/api/v1/devices/{DEVICE_ID}").json()
    assert pending["governance_status"] == "REVOKE_PENDING"

    beat(agent, generation=revoked["generation"], state="REVOKED")

    assert admin.get(f"/api/v1/devices/{DEVICE_ID}").json()["governance_status"] == "REVOKED"


def test_a_failed_device_is_failed_regardless_of_desired_state(app, admin, agent, device):
    body = put(admin)
    wait_for_artifact(app)

    beat(agent, generation=body["generation"], state="FAILED", message="checksum mismatch")

    assert admin.get(f"/api/v1/devices/{DEVICE_ID}").json()["governance_status"] == "FAILED"


def test_the_fleet_listing_shows_every_device(admin, device):
    admin.post("/api/v1/devices", json={"device_id": "jetson-orin-02"})

    listed = admin.get("/api/v1/devices").json()

    assert {d["device_id"] for d in listed} == {DEVICE_ID, "jetson-orin-02"}
    assert all(d["governance_status"] for d in listed)


def test_deployments_are_per_device(admin, device):
    """Obvious, and worth a test anyway: a shared desired-state row would mean one
    deploy reconfiguring the whole fleet."""
    admin.post("/api/v1/devices", json={"device_id": "jetson-orin-02"})
    put(admin, "2")

    other = admin.get("/api/v1/devices/jetson-orin-02").json()

    assert other["generation"] == 0
    assert other["desired_model_name"] is None
