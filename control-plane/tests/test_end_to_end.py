"""The real agent against the real control plane, in one process.

Every other test file sits on one side of the wire. This one is the only place
both packages are imported together, and it exists because the interesting bugs
in this system are *contract* bugs: the server's `Range` arithmetic against the
agent's resume offset, the server's `artifact_ready: false` against the agent's
decision not to treat it as a failure, the server's derived governance status
against the state the agent actually reports. None of those are visible from
either side alone, and a mock on either side would assert my own assumptions
back at me.

What makes this possible without a socket is that `starlette.testclient.
TestClient` is an `httpx.Client` subclass with a *synchronous* ASGI transport,
and `ControlPlaneClient` accepts an injected client. So the agent below is not a
simulation of the agent: it is `keeper`'s own `Reconciler`, `ArtifactManager`,
`StateStore` and HTTP client, issuing real requests -- including `Range` and
`If-Match` -- against the real FastAPI app, real SQLite, real artifact cache and
real deterministic tarballs. The only stand-ins are the registry (the fake, which
builds genuine MLflow tarballs) and the runtime (`MockRuntime`, which really
reads the file off disk). Both get replaced in M3/M4 with nothing above them
changing.

Two mechanical notes, because they look like shortcuts and are not:

* **A reboot is modelled by `boot()` on the same `data_dir`.** New reconciler,
  new runtime, new in-memory state -- the only continuity is `state.json`, which
  is exactly what survives a power cut on the Jetson.
* **Time is moved by shrinking thresholds, not by advancing a clock.**
  Connectivity is derived from `now_utc() - last_seen` against the server's real
  clock, so "the device went quiet for a minute" is expressed as "a minute is
  now the threshold". The arithmetic under test is identical and the test does
  not sleep.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from keeper.artifact_manager import ArtifactManager
from keeper.client import ControlPlaneClient
from keeper.config import AgentSettings
from keeper.reconciler import Reconciler
from keeper.runtime.mock import MockRuntime
from keeper.state import StateStore
from lighthouse.config import Settings
from lighthouse.main import create_app
from lighthouse.registry import FakeModelRegistry
from lighthouse.registry.fake import build_fake_artifact
from lighthouse.repositories import SqliteStore
from lighthouse_contracts import (
    ActualState,
    Connectivity,
    DesiredState,
    EventType,
    GovernanceStatus,
)

from conftest import ADMIN_TOKEN, DEVICE_ID, materialization_held, wait_for_artifact

# The agent joins relative artifact URIs onto this; TestClient answers any host.
BASE_URL = "http://testserver"


class Clock:
    """A hand-driven monotonic clock.

    The reconciler's retry backoff is measured in minutes. Sleeping through it
    would make this file the slowest in the suite and would still assert less:
    with the clock in hand, "the device refuses to retry before the backoff
    expires" is a real assertion rather than a race I happened to win.
    """

    def __init__(self, start: float = 10_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass(slots=True)
class Jetson:
    """One booted agent: the same objects `keeper.main` wires up."""

    settings: AgentSettings
    client: ControlPlaneClient
    artifacts: ArtifactManager
    runtime: MockRuntime
    reconciler: Reconciler

    def tick(self):
        return self.reconciler.tick()

    @property
    def state(self):
        return self.reconciler.state

    @property
    def archive(self) -> Path:
        return self.settings.artifact_dir / "fashion-cnn-1.tar.gz"

    @property
    def partial(self) -> Path:
        return self.settings.artifact_dir / "fashion-cnn-1.tar.gz.part"


def boot(http: TestClient, token: str, data_dir: Path, *, clock=None) -> Jetson:
    """Start an agent against `http`, persisting under `data_dir`.

    Called twice on the same `data_dir` to model a restart: everything here is
    rebuilt from scratch, so anything that survives came off disk.
    """
    settings = AgentSettings(
        device_id=DEVICE_ID,
        control_plane_url=BASE_URL,
        token=token,
        data_dir=data_dir,
        runtime_impl="mock",
    )
    # The injected client is not closed by `ControlPlaneClient.close()`
    # (`_owns_client` stays False), so the fixture keeps ownership of the ASGI
    # transport and the app's lifespan is driven exactly once.
    client = ControlPlaneClient(settings, client=http)
    artifacts = ArtifactManager(settings, client)
    runtime = MockRuntime()
    reconciler = Reconciler(
        settings,
        client,
        artifacts,
        runtime,
        StateStore(settings.state_path),
        clock=clock or time.monotonic,
    )
    return Jetson(settings, client, artifacts, runtime, reconciler)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def jetson(client, device, tmp_path, clock) -> Jetson:
    """An enrolled device running the real agent against the real app.

    `client` is the *unauthenticated* TestClient on purpose: the agent supplies
    its own `Authorization: Bearer lhd_...` on every request, so if device auth
    were broken nothing here would pass.
    """
    return boot(client, device, tmp_path / "keeper", clock=clock)


# -- driving the operator side ----------------------------------------------


def deploy(admin, version: str = "1", *, state: str = "RUNNING") -> int:
    response = admin.put(
        f"/api/v1/devices/{DEVICE_ID}/deployment",
        json={
            "model_name": "fashion-cnn",
            "model_version": version,
            "desired_state": state,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["generation"]


def view(admin) -> dict:
    """The device as the dashboard sees it."""
    response = admin.get(f"/api/v1/devices/{DEVICE_ID}")
    assert response.status_code == 200, response.text
    return response.json()


def events(admin, event_type: EventType | None = None) -> list[dict]:
    response = admin.get(f"/api/v1/devices/{DEVICE_ID}/events")
    assert response.status_code == 200, response.text
    rows = response.json()
    if event_type is None:
        return rows
    return [row for row in rows if row["event_type"] == event_type.value]


def published_digest(version: str = "1", payload_size: int = 64 * 1024) -> str:
    """What the registry's bytes hash to, computed independently of the server."""
    return hashlib.sha256(build_fake_artifact("fashion-cnn", version, payload_size)).hexdigest()


@contextmanager
def downloads_counted(jetson: Jetson):
    """Record the offset of every artifact request the agent issues.

    "Did it re-download?" is the question behind idempotence, stop/restart and
    revoke/re-authorize, and it cannot be answered by looking at the resulting
    files -- those look identical either way. Counting requests answers it
    directly, and the offsets double as the resume assertion.
    """
    offsets: list[int] = []
    original = jetson.client.stream_artifact

    def counting(artifact_uri, **kwargs):
        offsets.append(kwargs.get("offset", 0))
        return original(artifact_uri, **kwargs)

    jetson.client.stream_artifact = counting
    try:
        yield offsets
    finally:
        jetson.client.stream_artifact = original


# -- an idle device ----------------------------------------------------------


def test_an_enrolled_device_with_no_instruction_is_compliant(jetson, admin):
    """Generation 0 is a real instruction: "run nothing", and it converges.

    The alternative -- treating a device with no deployment as UNKNOWN -- would
    make every freshly enrolled device look broken on the dashboard until
    someone deployed to it.
    """
    jetson.tick()

    assert jetson.state.observed_generation == 0
    assert jetson.runtime.is_running is False

    device = view(admin)
    assert device["desired_state"] == DesiredState.STOPPED.value
    assert device["actual_state"] == ActualState.STOPPED.value
    assert device["governance_status"] == GovernanceStatus.HEALTHY.value
    assert device["connectivity"] == Connectivity.ONLINE.value
    assert device["inference_running"] is False


# -- deploy ------------------------------------------------------------------


def test_a_deployment_reaches_the_device_and_runs(jetson, admin, app):
    generation = deploy(admin)
    wait_for_artifact(app)

    jetson.tick()

    assert jetson.state.actual_state is ActualState.RUNNING
    assert jetson.state.observed_generation == generation
    assert jetson.runtime.is_running is True

    device = view(admin)
    assert device["governance_status"] == GovernanceStatus.HEALTHY.value
    assert device["actual_model_name"] == "fashion-cnn"
    assert device["actual_model_version"] == "1"
    assert device["observed_generation"] == generation
    assert device["inference_running"] is True


def test_the_device_runs_the_exact_bytes_the_registry_built(jetson, admin, app):
    """The whole chain, asserted against a digest computed outside it.

    Registry tarball -> control-plane hash-on-ingest -> cache file -> HTTP ->
    `.part` -> verify -> unpack -> runtime. Every link is real, and the digest
    here is computed from `build_fake_artifact` directly, so a bug that
    corrupted the bytes *consistently* on both sides still fails this.
    """
    deploy(admin)
    wait_for_artifact(app)
    jetson.tick()

    model = jetson.state.model
    assert model is not None
    assert model.sha256 == published_digest()

    unpacked = Path(model.path)
    assert unpacked.is_dir()
    assert (unpacked / "MLmodel").is_file()

    entrypoint = Path(model.entrypoint)
    assert entrypoint.is_file()
    assert entrypoint.name == "model.onnx"

    # The unpacked ONNX file is byte-identical to the tar member the registry
    # built -- not merely present with a plausible size.
    with tarfile.open(fileobj=io.BytesIO(build_fake_artifact("fashion-cnn", "1"))) as tar:
        expected = tar.extractfile("model.onnx").read()
    assert entrypoint.read_bytes() == expected

    # And the runtime was handed the ONNX file, not the directory around it --
    # a distinction a real `InferenceSession` would discover on the Jetson.
    assert model.entrypoint.endswith("model.onnx")
    assert jetson.runtime.predict([0.0])["mock"] is True


def test_a_second_tick_downloads_nothing_and_changes_nothing(jetson, admin, app):
    deploy(admin)
    wait_for_artifact(app)
    jetson.tick()
    before = view(admin)

    with downloads_counted(jetson) as offsets:
        jetson.tick()

    assert offsets == [], "a converged device re-fetched the artifact"
    after = view(admin)
    assert after["actual_state"] == before["actual_state"]
    assert after["observed_generation"] == before["observed_generation"]
    assert after["governance_status"] == GovernanceStatus.HEALTHY.value


def test_a_replayed_generation_is_ignored(jetson, admin, app):
    """Ordering is by generation, never by arrival (spec SS6).

    The payload replayed here is the genuine earlier response, so this is the
    stale-cache case rather than a hand-built fake.
    """
    deploy(admin, "1")
    wait_for_artifact(app)
    stale = jetson.client.fetch_desired_state()
    jetson.tick()

    deploy(admin, "2")
    wait_for_artifact(app)
    jetson.tick()
    assert jetson.state.model.version == "2"

    with downloads_counted(jetson) as offsets:
        jetson.reconciler.reconcile(stale)

    assert offsets == []
    assert jetson.state.model.version == "2"
    assert jetson.state.observed_generation > stale.generation


def test_a_reboot_converges_with_no_manual_step(jetson, admin, app, client, device, tmp_path):
    """The agent's only memory is `state.json`, and it is enough.

    Nothing is carried across the restart in memory: new reconciler, new state
    object, new runtime that is definitively not running.
    """
    deploy(admin)
    wait_for_artifact(app)
    jetson.tick()
    original_digest = jetson.state.model.sha256

    rebooted = boot(client, device, tmp_path / "keeper")
    assert rebooted.state.observed_generation == jetson.state.observed_generation
    assert rebooted.runtime.is_running is False

    with downloads_counted(rebooted) as offsets:
        rebooted.tick()

    # It restarted inference from the bytes already on disk rather than
    # re-fetching them: a reboot on a metered home uplink should be cheap.
    assert offsets == []
    assert rebooted.runtime.is_running is True
    assert rebooted.state.model.sha256 == original_digest
    assert view(admin)["governance_status"] == GovernanceStatus.HEALTHY.value


# -- version changes and rollback -------------------------------------------


def test_a_new_version_replaces_the_running_one(jetson, admin, app):
    deploy(admin, "1")
    wait_for_artifact(app)
    jetson.tick()

    generation = deploy(admin, "2")
    wait_for_artifact(app)
    jetson.tick()

    assert jetson.state.model.version == "2"
    assert jetson.state.model.sha256 == published_digest("2")
    assert jetson.runtime.is_running is True

    device = view(admin)
    assert device["actual_model_version"] == "2"
    assert device["observed_generation"] == generation
    assert device["governance_status"] == GovernanceStatus.HEALTHY.value
    assert events(admin, EventType.MODEL_VERSION_CHANGED)


def test_a_rollback_is_classified_as_a_rollback(jetson, admin, app):
    """The classification is the server's, derived from history.

    An operator cannot mislabel it, and -- more usefully -- cannot *forget* to
    label it: going back to a version is a rollback whether or not anyone says
    so.
    """
    deploy(admin, "1")
    wait_for_artifact(app)
    jetson.tick()
    deploy(admin, "2")
    wait_for_artifact(app)
    jetson.tick()

    deploy(admin, "1")
    wait_for_artifact(app)
    jetson.tick()

    assert events(admin, EventType.DEPLOYMENT_ROLLED_BACK)
    assert jetson.state.model.version == "1"
    assert jetson.state.model.sha256 == published_digest("1")
    assert view(admin)["governance_status"] == GovernanceStatus.HEALTHY.value


def test_a_rollback_uses_the_copy_it_kept(jetson, admin, app):
    """`prune(keep=2)` exists for exactly this round trip.

    Keeping one generation back is what makes the most likely action after a bad
    upgrade -- going back -- instant instead of a cold download.
    """
    deploy(admin, "1")
    wait_for_artifact(app)
    jetson.tick()
    deploy(admin, "2")
    wait_for_artifact(app)
    jetson.tick()
    deploy(admin, "1")
    wait_for_artifact(app)

    with downloads_counted(jetson) as offsets:
        jetson.tick()

    assert offsets == []
    assert jetson.state.model.version == "1"


# -- stop --------------------------------------------------------------------


def test_stop_shows_stop_pending_before_stopped(jetson, admin, app):
    """The transition is visible, which is the point of the dashboard.

    A UI that snapped straight to STOPPED would be claiming the device had
    complied before it had been asked, which is precisely the lie this system
    exists to prevent.
    """
    deploy(admin)
    wait_for_artifact(app)
    jetson.tick()

    assert admin.post(f"/api/v1/devices/{DEVICE_ID}/stop").status_code == 200

    pending = view(admin)
    assert pending["desired_state"] == DesiredState.STOPPED.value
    assert pending["actual_state"] == ActualState.RUNNING.value
    assert pending["governance_status"] == GovernanceStatus.STOP_PENDING.value
    assert pending["inference_running"] is True

    jetson.tick()

    settled = view(admin)
    assert settled["actual_state"] == ActualState.STOPPED.value
    assert settled["governance_status"] == GovernanceStatus.HEALTHY.value
    assert settled["inference_running"] is False
    assert jetson.runtime.is_running is False


def test_stop_keeps_the_artifacts_and_restart_is_local(jetson, admin, app):
    deploy(admin)
    wait_for_artifact(app)
    jetson.tick()
    unpacked = Path(jetson.state.model.path)

    admin.post(f"/api/v1/devices/{DEVICE_ID}/stop")
    jetson.tick()

    # Stop is reversible: the bytes stay.
    assert unpacked.is_dir()
    assert jetson.archive.is_file()

    deploy(admin)
    wait_for_artifact(app)
    with downloads_counted(jetson) as offsets:
        jetson.tick()

    assert offsets == []
    assert jetson.runtime.is_running is True
    assert view(admin)["governance_status"] == GovernanceStatus.HEALTHY.value


# -- revoke ------------------------------------------------------------------


def test_revoke_shows_revoke_pending_then_revoked_and_deletes_the_bytes(jetson, admin, app):
    deploy(admin)
    wait_for_artifact(app)
    jetson.tick()
    unpacked = Path(jetson.state.model.path)

    assert admin.post(f"/api/v1/devices/{DEVICE_ID}/revoke").status_code == 200

    pending = view(admin)
    assert pending["governance_status"] == GovernanceStatus.REVOKE_PENDING.value
    assert pending["actual_state"] == ActualState.RUNNING.value

    jetson.tick()

    settled = view(admin)
    assert settled["actual_state"] == ActualState.REVOKED.value
    assert settled["governance_status"] == GovernanceStatus.REVOKED.value
    assert settled["inference_running"] is False

    # Revoke is the irreversible one: the unpacked model and the archive are
    # both gone, not merely stopped.
    assert not unpacked.exists()
    assert not jetson.archive.exists()
    assert jetson.state.model is None


def test_revoke_deletes_the_copy_kept_for_rollback_as_well(jetson, admin, app):
    """Revoke must leave *no* model bytes, including the ones kept for rollback.

    Found by running `scripts/dev.sh` rather than by this suite, which is the
    uncomfortable part: the single-version revoke test above passes either way.
    `prune(keep=2)` deliberately retains the previous version so a rollback is a
    local operation, so after an upgrade the device holds two complete copies. A
    revoke that removed only the version named in the instruction left the other
    one -- archive *and* unpacked ONNX -- on a device the dashboard was reporting
    as REVOKED.
    """
    deploy(admin, "1")
    wait_for_artifact(app)
    jetson.tick()
    first = Path(jetson.state.model.path)

    deploy(admin, "2")
    wait_for_artifact(app)
    jetson.tick()
    assert jetson.state.model.version == "2"
    # The premise of the test: v1 is still there, on purpose.
    assert first.exists(), "prune(keep=2) should retain the previous version"

    assert admin.post(f"/api/v1/devices/{DEVICE_ID}/revoke").status_code == 200
    jetson.tick()

    assert view(admin)["governance_status"] == GovernanceStatus.REVOKED.value
    assert not first.exists()
    # Nothing left anywhere -- not the unpacked trees, not the archives.
    assert list(jetson.settings.model_dir.rglob("*.onnx")) == []
    assert list(jetson.settings.artifact_dir.glob("*")) == []


def test_a_revoked_device_can_be_re_authorized_and_re_downloads(jetson, admin, app):
    """The sharp edge in SS5: REVOKED -> RUNNING must work, and must re-fetch.

    Revocation blocks restart "unless a newer desired-state generation
    explicitly authorizes deployment" -- a PUT carrying RUNNING *is* that
    authorization. But revoke deleted the files, so converging means a real
    download, and the local state says REVOKED with no model. Getting this wrong
    leaves a device permanently unusable after one revoke.
    """
    deploy(admin)
    wait_for_artifact(app)
    jetson.tick()
    admin.post(f"/api/v1/devices/{DEVICE_ID}/revoke")
    jetson.tick()
    assert jetson.state.actual_state is ActualState.REVOKED

    deploy(admin)
    wait_for_artifact(app)
    with downloads_counted(jetson) as offsets:
        jetson.tick()

    assert offsets == [0], "re-authorization must re-download from scratch"
    assert jetson.state.actual_state is ActualState.RUNNING
    assert jetson.runtime.is_running is True
    assert Path(jetson.state.model.path).is_dir()
    assert view(admin)["governance_status"] == GovernanceStatus.HEALTHY.value


# -- waiting is not failing --------------------------------------------------


def test_an_unmaterialized_artifact_is_a_wait_not_a_failure(jetson, admin, app):
    """`artifact_ready: false` is a transient state the spec never names.

    `PUT /deployment` returns without waiting (SS13), so a device can poll inside
    the materialization window and get a generation with no digest attached. An
    agent that read that as an error would mark itself FAILED and earn a backoff
    for something the server was about to finish.
    """
    jetson.tick()  # converge on generation 0, so the device is known-idle

    with materialization_held(app) as release:
        generation = deploy(admin)

        desired = jetson.client.fetch_desired_state()
        assert desired.generation == generation
        assert desired.artifact_ready is False
        assert desired.model is None

        jetson.tick()

        assert jetson.state.actual_state is not ActualState.FAILED
        assert jetson.state.failure_count == 0
        assert jetson.state.observed_generation < generation
        # The dashboard reads this honestly: asked for, not yet acknowledged.
        assert view(admin)["governance_status"] == GovernanceStatus.OUT_OF_SYNC.value

        release.set()
        wait_for_artifact(app)

    jetson.tick()

    assert jetson.state.actual_state is ActualState.RUNNING
    assert jetson.state.observed_generation == generation
    assert view(admin)["governance_status"] == GovernanceStatus.HEALTHY.value


# -- a failed checksum must block deployment --------------------------------


def test_corrupted_bytes_fail_closed_and_never_run(jetson, admin, app):
    """SS17, the one rule with no acceptable exception.

    The corruption is injected in the *serving* path, so the control plane's own
    digest is honest and only the bytes on the wire disagree with it -- which is
    the real failure mode (a truncating proxy, a flipped bit) rather than a
    registry that lies.
    """
    app.state.ctx.settings.dev_corrupt_artifacts = True
    generation = deploy(admin)
    wait_for_artifact(app)

    jetson.tick()

    assert jetson.state.actual_state is ActualState.FAILED
    assert jetson.runtime.is_running is False
    assert jetson.state.model is None
    # Nothing was left behind for a later resume to splice onto.
    assert not jetson.archive.exists()
    assert not jetson.partial.exists()

    device = view(admin)
    assert device["governance_status"] == GovernanceStatus.FAILED.value
    assert device["inference_running"] is False
    # The generation *is* acked on terminal failure, deliberately: the operator
    # needs to know which instruction failed, not that none arrived.
    assert device["observed_generation"] == generation

    failures = events(admin, EventType.RECONCILE_FAILED)
    assert failures, "a checksum failure must be audited"
    assert "sha256" in (failures[0]["details"].get("message") or "")


def test_a_failed_device_holds_its_backoff_then_recovers(jetson, admin, app, clock):
    app.state.ctx.settings.dev_corrupt_artifacts = True
    deploy(admin)
    wait_for_artifact(app)
    jetson.tick()
    assert jetson.state.actual_state is ActualState.FAILED

    # Still inside the penalty: retrying every 10 seconds against a control
    # plane that is serving bad bytes is how a fleet turns one bad artifact into
    # a thundering herd.
    app.state.ctx.settings.dev_corrupt_artifacts = False
    with downloads_counted(jetson) as offsets:
        jetson.tick()
    assert offsets == []
    assert jetson.state.actual_state is ActualState.FAILED

    clock.advance(jetson.settings.retry_backoff_initial_seconds + 1)
    jetson.tick()

    assert jetson.state.actual_state is ActualState.RUNNING
    assert jetson.runtime.is_running is True
    assert view(admin)["governance_status"] == GovernanceStatus.HEALTHY.value


# -- connectivity ------------------------------------------------------------


def test_connectivity_comes_from_heartbeat_age_alone(jetson, admin, app):
    """ONLINE -> STALE -> OFFLINE with nothing written in between.

    A stored `online` boolean would need a sweeper to ever go false and would be
    wrong for as long as the sweeper lagged. Here the same stored `last_seen`
    yields all three answers, which is the proof that the classification is
    derived: `last_seen` is asserted identical across the three reads.
    """
    jetson.tick()
    settings = app.state.ctx.settings

    fresh = view(admin)
    assert fresh["connectivity"] == Connectivity.ONLINE.value

    # The agent has stopped ticking. Equivalent to waiting past the threshold,
    # without the wait.
    settings.online_threshold_seconds = 0
    stale = view(admin)
    assert stale["connectivity"] == Connectivity.STALE.value

    settings.stale_threshold_seconds = 0
    offline = view(admin)
    assert offline["connectivity"] == Connectivity.OFFLINE.value

    assert fresh["last_seen"] == stale["last_seen"] == offline["last_seen"]
    # Going quiet says nothing about compliance: the device is still running
    # what it was told to run, we just cannot currently hear it.
    assert offline["governance_status"] == fresh["governance_status"]


def test_an_enrolled_device_that_never_checked_in_is_distinguishable(admin, device):
    """NEVER_SEEN is not OFFLINE: one is a provisioning bug, one is operational."""
    assert view(admin)["connectivity"] == Connectivity.NEVER_SEEN.value


# -- resume across a cut transfer -------------------------------------------


@pytest.fixture
def big_artifact(tmp_path):
    """An app whose artifact is several chunks long, plus a device on it.

    The agent reads 1 MiB at a time, so the default 64 KiB artifact arrives in a
    single chunk and can never exercise resume -- an abort either happens before
    any bytes land or after all of them. A 3 MiB payload of incompressible
    pseudo-random data gzips to ~3 MiB, giving three chunks and a genuine
    partial file to resume from.
    """
    settings = Settings(
        env="local",
        data_dir=tmp_path / "lighthouse-big",
        registry_impl="fake",
        admin_token=ADMIN_TOKEN,
    )
    payload_size = 3 * 1024 * 1024
    app = create_app(
        settings,
        store=SqliteStore(":memory:"),
        registry=FakeModelRegistry(payload_size=payload_size),
    )
    with TestClient(app) as http:
        response = http.post(
            "/api/v1/devices",
            json={"device_id": DEVICE_ID, "platform": "jetson-orin"},
            headers={"X-Lighthouse-Admin-Token": ADMIN_TOKEN},
        )
        assert response.status_code == 201, response.text
        token = response.json()["credentials"]["token"]
        operator = TestClient(
            app, headers={"X-Lighthouse-Admin-Token": ADMIN_TOKEN}, base_url=BASE_URL
        )
        yield app, http, operator, token, payload_size


def test_an_aborted_download_resumes_and_still_verifies(big_artifact, tmp_path):
    """The one test where both sides of `Range` have to agree exactly.

    The agent aborts mid-transfer, keeps its `.part`, and resumes with
    `Range: bytes=<offset>-` plus `If-Match`. The server's range arithmetic is
    inclusive at both ends; an off-by-one anywhere produces a file that hashes to
    nothing recognizable, and the assertion below is the assembled digest, not a
    206 status. That is the difference between testing resume and testing that
    resume was attempted.
    """
    app, http, operator, token, payload_size = big_artifact
    jetson = boot(http, token, tmp_path / "keeper-big")

    generation = deploy(operator)
    wait_for_artifact(app)

    # Abort before the *second* chunk, so exactly one chunk is on disk.
    chunks = {"seen": 0}

    def abort_after_first_chunk(_generation: int) -> bool:
        chunks["seen"] += 1
        return chunks["seen"] > 1

    jetson.reconciler.set_abort_probe(abort_after_first_chunk)
    with downloads_counted(jetson) as offsets:
        jetson.tick()

    assert offsets == [0]
    assert jetson.state.actual_state is not ActualState.FAILED
    assert jetson.state.observed_generation < generation
    assert jetson.partial.is_file(), "an aborted download must keep its partial"
    partial_size = jetson.partial.stat().st_size
    assert 0 < partial_size < payload_size

    jetson.reconciler.set_abort_probe(None)
    with downloads_counted(jetson) as offsets:
        jetson.tick()

    assert offsets == [partial_size], "the resume did not start where the abort stopped"
    assert jetson.state.actual_state is ActualState.RUNNING
    assert jetson.state.model.sha256 == published_digest("1", payload_size)
    assert not jetson.partial.exists()
    assert operator.get(f"/api/v1/devices/{DEVICE_ID}").json()[
        "governance_status"
    ] == GovernanceStatus.HEALTHY.value
