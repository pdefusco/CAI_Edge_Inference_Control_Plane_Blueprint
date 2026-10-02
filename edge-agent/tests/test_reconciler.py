"""The reconciler cases the spec requires (SS22), plus three the plan adds.

Spec SS22 lists twelve. Ten are agent-side and live here; the last two -- "device
heartbeat updates actual state" and "offline device detected from heartbeat age" --
are server-side derivations and are tested in `control-plane/tests`.

The three additions, each a bug that would otherwise reach the Jetson:

  * `artifact_ready: false` must be a wait, never a FAILED;
  * `REVOKED -> RUNNING` must work and must re-download, because revoke deleted
    the bytes;
  * a generation superseded mid-download must abort without corrupting state.
"""

from __future__ import annotations

from lighthouse_contracts import ActualState, DesiredState

# --------------------------------------------------------------------------
# SS22: no deployment -> deploy model
# --------------------------------------------------------------------------


def test_no_deployment_deploys_model(harness):
    model = harness.model("fashion-cnn", "1")
    desired = harness.desire(1, DesiredState.RUNNING, model)

    outcome = harness.reconcile(desired)

    assert outcome.actual_state is ActualState.RUNNING
    assert harness.runtime.is_running
    assert harness.state.observed_generation == 1
    assert harness.state.model is not None
    assert harness.state.model.version == "1"
    assert (harness.model_dir("fashion-cnn", "1") / "model.onnx").is_file()


# --------------------------------------------------------------------------
# SS22: correct model already running -> no-op
# --------------------------------------------------------------------------


def test_already_running_is_a_noop(harness):
    model = harness.model("fashion-cnn", "1")
    desired = harness.desire(1, DesiredState.RUNNING, model)
    harness.reconcile(desired)
    downloads_before = harness.client.download_count

    outcome = harness.reconcile(desired)

    assert outcome.changed is False
    assert harness.client.download_count == downloads_before, "re-downloaded a model it already had"
    assert harness.state.actual_state is ActualState.RUNNING


# --------------------------------------------------------------------------
# SS22: wrong model version -> upgrade
# --------------------------------------------------------------------------


def test_wrong_version_upgrades(harness):
    v1 = harness.model("fashion-cnn", "1")
    harness.reconcile(harness.desire(1, DesiredState.RUNNING, v1))

    v2 = harness.model("fashion-cnn", "2")
    outcome = harness.reconcile(harness.desire(2, DesiredState.RUNNING, v2))

    assert outcome.actual_state is ActualState.RUNNING
    assert harness.state.model.version == "2"
    assert harness.state.observed_generation == 2
    assert harness.runtime.is_running


def test_same_version_different_bytes_redeploys(harness):
    """A registry label can be repointed at different bytes.

    Comparing only (name, version) would leave the old model serving forever under
    a version label that now means something else -- a silent governance failure,
    and the reason `AgentState.matches` includes the digest.
    """
    v1 = harness.model("fashion-cnn", "1")
    harness.reconcile(harness.desire(1, DesiredState.RUNNING, v1))

    repointed = harness.model("fashion-cnn", "1", size=8192)
    assert repointed.sha256 != v1.sha256
    outcome = harness.reconcile(harness.desire(2, DesiredState.RUNNING, repointed))

    assert outcome.actual_state is ActualState.RUNNING
    assert harness.state.model.sha256 == repointed.sha256


# --------------------------------------------------------------------------
# SS22: RUNNING -> STOPPED, STOPPED -> RUNNING
# --------------------------------------------------------------------------


def test_running_to_stopped_keeps_artifacts(harness):
    model = harness.model("fashion-cnn", "1")
    harness.reconcile(harness.desire(1, DesiredState.RUNNING, model))

    outcome = harness.reconcile(harness.desire(2, DesiredState.STOPPED, model))

    assert outcome.actual_state is ActualState.STOPPED
    assert not harness.runtime.is_running
    # Stop is reversible, so the bytes stay: SS5 distinguishes stop from revoke
    # precisely here.
    assert (harness.model_dir("fashion-cnn", "1") / "model.onnx").is_file()
    assert harness.state.model is not None


def test_stopped_to_running_uses_local_copy(harness):
    model = harness.model("fashion-cnn", "1")
    harness.reconcile(harness.desire(1, DesiredState.RUNNING, model))
    harness.reconcile(harness.desire(2, DesiredState.STOPPED, model))
    downloads_before = harness.client.download_count

    outcome = harness.reconcile(harness.desire(3, DesiredState.RUNNING, model))

    assert outcome.actual_state is ActualState.RUNNING
    assert harness.runtime.is_running
    assert harness.client.download_count == downloads_before, "restart should not re-download"


# --------------------------------------------------------------------------
# SS22: RUNNING -> REVOKED
# --------------------------------------------------------------------------


def test_running_to_revoked_destroys_artifacts(harness):
    model = harness.model("fashion-cnn", "1")
    harness.reconcile(harness.desire(1, DesiredState.RUNNING, model))

    outcome = harness.reconcile(harness.desire(2, DesiredState.REVOKED, model))

    assert outcome.actual_state is ActualState.REVOKED
    assert not harness.runtime.is_running
    assert not harness.model_dir("fashion-cnn", "1").exists(), "revoke left the model on disk"
    assert harness.state.model is None
    # The downloaded archive must go too, or a revoked model is one `tar -xzf`
    # away from being recovered by anyone with shell access to the device.
    assert list(harness.settings.artifact_dir.glob("*.tar.gz")) == []


def test_revoke_without_known_model_clears_everything(harness):
    """Revoke must succeed on a device that lost its state file.

    Leaving unknown model bytes on a revoked device is exactly what revoke exists
    to prevent, so "I don't know what was deployed" resolves to "delete all of it".
    """
    model = harness.model("fashion-cnn", "1")
    harness.reconcile(harness.desire(1, DesiredState.RUNNING, model))
    harness.state.model = None

    outcome = harness.reconcile(harness.desire(2, DesiredState.REVOKED, None))

    assert outcome.actual_state is ActualState.REVOKED
    assert not harness.model_dir("fashion-cnn", "1").exists()


# --------------------------------------------------------------------------
# SS22: repeated same generation -> no-op; stale generation -> ignore
# --------------------------------------------------------------------------


def test_repeated_generation_is_idempotent(harness):
    model = harness.model("fashion-cnn", "1")
    desired = harness.desire(1, DesiredState.RUNNING, model)

    first = harness.reconcile(desired)
    states = [harness.reconcile(desired) for _ in range(4)]

    assert first.changed is True
    assert all(o.changed is False for o in states)
    assert harness.client.download_count == 1


def test_stale_generation_is_ignored(harness):
    model = harness.model("fashion-cnn", "1")
    harness.reconcile(harness.desire(5, DesiredState.RUNNING, model))

    # A replayed older instruction, e.g. from a cache or an out-of-order response.
    stale = harness.desire(3, DesiredState.REVOKED, model)
    outcome = harness.reconcile(stale)

    assert outcome.changed is False
    assert harness.state.actual_state is ActualState.RUNNING
    assert harness.runtime.is_running, "a stale REVOKED must not tear down a live model"
    assert harness.state.observed_generation == 5


def test_same_generation_self_heals_after_runtime_crash(harness):
    """The reason the generation gate discards *older* rather than requiring *newer*.

    A reconciler written as `if generation > observed` never repairs a runtime that
    died at the current generation -- the device sits there reporting RUNNING with
    nothing running, forever, and only an operator poking a new deployment fixes it.
    """
    model = harness.model("fashion-cnn", "1")
    desired = harness.desire(1, DesiredState.RUNNING, model)
    harness.reconcile(desired)

    harness.runtime.stop()  # simulate the inference process dying
    assert not harness.runtime.is_running

    outcome = harness.reconcile(desired)

    assert harness.runtime.is_running, "device did not self-heal at the same generation"
    assert outcome.actual_state is ActualState.RUNNING


# --------------------------------------------------------------------------
# SS22: artifact checksum failure -> FAILED
# --------------------------------------------------------------------------


def test_checksum_failure_fails_and_never_runs(harness):
    model = harness.model("fashion-cnn", "1")
    harness.client.corrupt_bytes = True

    outcome = harness.reconcile(harness.desire(1, DesiredState.RUNNING, model))

    assert outcome.actual_state is ActualState.FAILED
    assert not harness.runtime.is_running, "SS17: a failed checksum must prevent deployment"
    assert not harness.model_dir("fashion-cnn", "1").exists()
    assert "sha256" in (harness.state.message or "")
    # The generation is acked despite failing, so the dashboard can attribute the
    # failure to the deployment that caused it.
    assert harness.state.observed_generation == 1


def test_truncated_download_fails_on_checksum(harness):
    model = harness.model("fashion-cnn", "1")
    harness.client.truncate_after = 100

    outcome = harness.reconcile(harness.desire(1, DesiredState.RUNNING, model))

    assert outcome.actual_state is ActualState.FAILED
    assert not harness.runtime.is_running


# --------------------------------------------------------------------------
# SS22: runtime start failure -> FAILED
# --------------------------------------------------------------------------


def test_runtime_start_failure_fails(harness):
    model = harness.model("fashion-cnn", "1")
    harness.runtime._fail_on_start = True

    outcome = harness.reconcile(harness.desire(1, DesiredState.RUNNING, model))

    assert outcome.actual_state is ActualState.FAILED
    assert not harness.runtime.is_running
    assert harness.state.inference_running is False


def test_runtime_load_failure_fails(harness):
    model = harness.model("fashion-cnn", "1")
    harness.runtime._fail_on_load = True

    outcome = harness.reconcile(harness.desire(1, DesiredState.RUNNING, model))

    assert outcome.actual_state is ActualState.FAILED
    assert harness.state.model is None, "a model that failed to load must not be recorded as deployed"


# --------------------------------------------------------------------------
# Plan addition 1: artifact_ready false -> wait, not FAIL
# --------------------------------------------------------------------------


def test_artifact_not_ready_waits_without_failing(harness):
    """The single most dangerous easy mistake in the agent.

    `artifact_ready: false` means the control plane is still materializing. Reading
    it as an error would mark a perfectly healthy device FAILED and require an
    operator to clear it by hand.
    """
    desired = harness.desire(1, DesiredState.RUNNING, None, artifact_ready=False)

    outcome = harness.reconcile(desired)

    assert outcome.actual_state is not ActualState.FAILED
    assert harness.state.observed_generation == 0, "must not ack a generation it has not converged to"
    assert harness.client.download_count == 0
    assert "waiting" in outcome.note


def test_artifact_not_ready_preserves_the_running_model(harness):
    """While waiting for an upgrade's bytes, the device is still running the *old*
    version, and must report that rather than inventing an intermediate state."""
    v1 = harness.model("fashion-cnn", "1")
    harness.reconcile(harness.desire(1, DesiredState.RUNNING, v1))

    harness.reconcile(harness.desire(2, DesiredState.RUNNING, None, artifact_ready=False))

    assert harness.state.actual_state is ActualState.RUNNING
    assert harness.state.model.version == "1"
    assert harness.state.observed_generation == 1
    assert harness.runtime.is_running


def test_artifact_not_ready_then_ready_converges(harness):
    harness.reconcile(harness.desire(1, DesiredState.RUNNING, None, artifact_ready=False))
    model = harness.model("fashion-cnn", "1")

    outcome = harness.reconcile(harness.desire(1, DesiredState.RUNNING, model))

    assert outcome.actual_state is ActualState.RUNNING
    assert harness.state.observed_generation == 1


# --------------------------------------------------------------------------
# Plan addition 2: REVOKED -> RUNNING re-downloads
# --------------------------------------------------------------------------


def test_revoked_to_running_redownloads(harness):
    """SS5 blocks restart "unless a newer desired-state generation explicitly
    authorizes deployment". A PUT of RUNNING is that authorization -- and because
    revoke deleted the files, honoring it means a genuine re-download."""
    model = harness.model("fashion-cnn", "1")
    harness.reconcile(harness.desire(1, DesiredState.RUNNING, model))
    harness.reconcile(harness.desire(2, DesiredState.REVOKED, model))
    downloads_before = harness.client.download_count

    outcome = harness.reconcile(harness.desire(3, DesiredState.RUNNING, model))

    assert outcome.actual_state is ActualState.RUNNING
    assert harness.client.download_count == downloads_before + 1, "revoked bytes were not re-fetched"
    assert (harness.model_dir("fashion-cnn", "1") / "model.onnx").is_file()


def test_revoked_stays_revoked_on_replay(harness):
    """Replaying the revoke generation must not restart the model."""
    model = harness.model("fashion-cnn", "1")
    harness.reconcile(harness.desire(1, DesiredState.RUNNING, model))
    revoke = harness.desire(2, DesiredState.REVOKED, model)
    harness.reconcile(revoke)

    for _ in range(3):
        harness.reconcile(revoke)

    assert harness.state.actual_state is ActualState.REVOKED
    assert not harness.runtime.is_running
    assert not harness.model_dir("fashion-cnn", "1").exists()


# --------------------------------------------------------------------------
# Plan addition 3: superseded generation mid-download aborts cleanly
# --------------------------------------------------------------------------


def abort_after(chunks: int):
    """An abort probe that lets `chunks` chunks through, then trips.

    Aborting on the very first poll would leave a zero-byte partial, and an
    assertion about zero bytes is not an assertion about resume.
    """
    seen = {"n": 0}

    def probe(_generation: int) -> bool:
        seen["n"] += 1
        return seen["n"] > chunks

    return probe


def test_superseded_mid_download_aborts_cleanly(harness):
    """A slow link must not finish fetching a version the operator already replaced.

    "Cleanly" is the assertion that matters: no FAILED, no ack, no half-unpacked
    model directory, and the previously-running version still serving.
    """
    v1 = harness.model("fashion-cnn", "1")
    harness.reconcile(harness.desire(1, DesiredState.RUNNING, v1))

    v2 = harness.model("fashion-cnn", "2", size=256 * 1024)
    harness.client.chunk_size = 32 * 1024
    harness.reconciler.set_abort_probe(abort_after(2))

    outcome = harness.reconcile(harness.desire(2, DesiredState.RUNNING, v2))

    assert outcome.actual_state is not ActualState.FAILED
    assert harness.state.observed_generation == 1, "acked a generation it abandoned"
    assert not harness.model_dir("fashion-cnn", "2").exists()
    assert "abandoned" in outcome.note
    # The old version is what the device is still actually serving, and the state
    # it reports must say so.
    assert harness.state.model is not None and harness.state.model.version == "1"
    assert (harness.model_dir("fashion-cnn", "1") / "model.onnx").is_file()


def test_abort_leaves_a_resumable_partial_and_resumes_from_it(harness):
    """The abandoned transfer is not wasted: the `.part` survives so that if the
    same version comes back around, the device resumes instead of restarting."""
    model = harness.model("fashion-cnn", "1", size=256 * 1024)
    harness.client.chunk_size = 32 * 1024
    harness.reconciler.set_abort_probe(abort_after(2))
    harness.reconcile(harness.desire(1, DesiredState.RUNNING, model))

    parts = list(harness.settings.artifact_dir.glob("*.part"))
    metas = list(harness.settings.artifact_dir.glob("*.meta"))
    assert parts, "no partial download kept"
    assert metas, "no sidecar kept, so the partial could never be safely resumed"
    partial_size = parts[0].stat().st_size
    assert partial_size > 0, "kept an empty partial, which resume cannot build on"

    # Second attempt, no abort: it must pick up where the first left off and still
    # arrive at the correct whole-file digest.
    harness.reconciler.set_abort_probe(None)
    outcome = harness.reconcile(harness.desire(1, DesiredState.RUNNING, model))

    assert outcome.actual_state is ActualState.RUNNING
    assert harness.state.model.sha256 == model.sha256
    assert not list(harness.settings.artifact_dir.glob("*.part")), "partial not cleaned up"


def test_revoked_artifact_mid_download_is_not_a_failure(harness):
    """A 403 on the artifact route means a revoke landed while we were fetching.

    Treating it as FAILED would leave a device reporting a deployment failure for
    a model the operator deliberately pulled -- noise that looks like a defect.
    """
    model = harness.model("fashion-cnn", "1")
    harness.client.raise_forbidden = True

    outcome = harness.reconcile(harness.desire(1, DesiredState.RUNNING, model))

    assert outcome.actual_state is not ActualState.FAILED
    assert harness.state.observed_generation == 0


# --------------------------------------------------------------------------
# Transient failure handling and backoff
# --------------------------------------------------------------------------


def test_transient_error_does_not_fail_the_deployment(harness):
    model = harness.model("fashion-cnn", "1")
    harness.client.raise_transient = True

    outcome = harness.reconcile(harness.desire(1, DesiredState.RUNNING, model))

    assert outcome.actual_state is not ActualState.FAILED, "a network blip is not a deployment failure"
    assert harness.state.observed_generation == 0

    harness.client.raise_transient = False
    harness.reconcile(harness.desire(1, DesiredState.RUNNING, model))
    assert harness.state.actual_state is ActualState.RUNNING


def test_failure_backs_off_before_retrying(harness, clock):
    model = harness.model("fashion-cnn", "1")
    harness.client.corrupt_bytes = True
    desired = harness.desire(1, DesiredState.RUNNING, model)

    harness.reconcile(desired)
    assert harness.state.actual_state is ActualState.FAILED
    attempts_after_first = harness.client.download_count

    # Immediately re-polling must not re-attempt: a device retrying a broken
    # artifact every poll interval is a denial of service against the control plane.
    harness.reconcile(desired)
    assert harness.client.download_count == attempts_after_first
    assert "holding FAILED" in harness.reconciler.reconcile(desired).note

    clock.advance(11)
    harness.reconcile(desired)
    assert harness.client.download_count == attempts_after_first + 1


def test_backoff_grows_then_caps(harness, clock):
    model = harness.model("fashion-cnn", "1")
    harness.client.corrupt_bytes = True
    desired = harness.desire(1, DesiredState.RUNNING, model)

    delays = []
    for _ in range(6):
        harness.reconcile(desired)
        delays.append(harness.state.next_retry_monotonic - clock())
        clock.advance(max(delays[-1], 0) + 1)

    assert delays[0] == 10
    assert delays[1] == 20
    assert delays[-1] <= harness.settings.retry_backoff_max_seconds
    assert delays[-1] == harness.settings.retry_backoff_max_seconds


def test_new_generation_clears_the_backoff(harness, clock):
    """An operator pushing a fix must not wait out a penalty earned by the version
    they are replacing. This is why backoff is keyed to the failed generation."""
    bad = harness.model("fashion-cnn", "1")
    harness.client.corrupt_bytes = True
    harness.reconcile(harness.desire(1, DesiredState.RUNNING, bad))
    assert harness.state.actual_state is ActualState.FAILED

    harness.client.corrupt_bytes = False
    good = harness.model("fashion-cnn", "2")

    # No clock advance: the backoff from generation 1 is still pending.
    outcome = harness.reconcile(harness.desire(2, DesiredState.RUNNING, good))

    assert outcome.actual_state is ActualState.RUNNING, "fix was blocked by the old backoff"
    assert harness.state.failure_count == 0


# --------------------------------------------------------------------------
# Heartbeat reporting
# --------------------------------------------------------------------------


def test_heartbeat_reports_actual_state(harness):
    model = harness.model("fashion-cnn", "1")
    harness.reconcile(harness.desire(1, DesiredState.RUNNING, model))

    harness.reconciler.report()

    hb = harness.client.heartbeats[-1]
    assert hb.device_id == harness.settings.device_id
    assert hb.actual_state is ActualState.RUNNING
    assert hb.observed_generation == 1
    assert hb.model is not None and hb.model.sha256 == model.sha256
    assert hb.runtime.inference_running is True
    assert hb.timestamp.tzinfo is not None, "a naive timestamp breaks server-side staleness"


def test_heartbeat_reports_failure_message(harness):
    model = harness.model("fashion-cnn", "1")
    harness.client.corrupt_bytes = True
    harness.reconcile(harness.desire(1, DesiredState.RUNNING, model))

    harness.reconciler.report()

    hb = harness.client.heartbeats[-1]
    assert hb.actual_state is ActualState.FAILED
    assert hb.message and "sha256" in hb.message


def test_tick_reports_even_when_reconcile_does_nothing(harness):
    harness.desire(0, DesiredState.STOPPED, None)

    harness.reconciler.tick()

    assert harness.client.heartbeats, "silence is how the dashboard decides a device is offline"


# --------------------------------------------------------------------------
# Restart durability
# --------------------------------------------------------------------------


def test_state_survives_a_restart(harness, clock):
    from keeper.reconciler import Reconciler
    from keeper.state import StateStore

    model = harness.model("fashion-cnn", "1")
    harness.reconcile(harness.desire(1, DesiredState.RUNNING, model))

    # A fresh agent process against the same data directory, as after a reboot.
    restarted = Reconciler(
        harness.settings,
        harness.client,  # type: ignore[arg-type]
        harness.artifacts,
        harness.runtime.__class__(),
        StateStore(harness.settings.state_path),
        clock=clock,
    )
    assert restarted.state.observed_generation == 1
    assert restarted.state.model is not None
    assert restarted.state.model.version == "1"

    downloads_before = harness.client.download_count
    outcome = restarted.reconcile(harness.desire(1, DesiredState.RUNNING, model))

    # The new process has a fresh runtime with nothing loaded, so it must reload --
    # but from the verified local copy, not the network.
    assert outcome.actual_state is ActualState.RUNNING
    assert harness.client.download_count == downloads_before
