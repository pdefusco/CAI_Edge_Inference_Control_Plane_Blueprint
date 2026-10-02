"""The reconciliation loop. This is the file the demo stands or falls on.

Spec SS9, stated as the project's organising principle: *CAI owns desired state. The
edge agent owns reconciliation and reports actual state.* Everything here follows
from reading that literally.

Four decisions are load-bearing and worth stating before the code:

**Generation gates staleness, not looking.** A naive reconciler checks
`generation > observed` and skips otherwise -- which means a device whose runtime
crashed at generation 7 never repairs itself, because 7 is not greater than 7.
Instead the generation gate only *discards older* instructions
(`desired.generation < observed` -> ignore), and convergence is decided by
comparing desired against what is actually on this box. That one choice satisfies
three of the spec's required cases at once: "repeated same generation -> no-op",
"stale generation -> ignore", and self-healing after a crash.

**`observed_generation` means converged, not seen.** It advances on success and on
terminal failure -- the latter so the dashboard can tie a FAILED state to the
instruction that caused it. It does *not* advance while a download is in flight or
while waiting on `artifact_ready: false`, because the control plane derives
governance from it: acking early would render a mid-download device HEALTHY.

**`artifact_ready: false` is waiting, not failing.** The control plane is still
materializing bytes. Setting FAILED here would be a self-inflicted outage that an
operator then has to clear by hand.

**Failures back off, and a new generation clears the backoff.** A device retrying a
broken artifact every 10s is a denial of service against your own control plane.
But an operator pushing a *fix* must not wait out a penalty earned by the version
they are replacing, so the backoff is keyed to the generation that failed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

from lighthouse_contracts import (
    ActualState,
    DesiredState,
    DesiredStateResponse,
    HardwareInfo,
    HeartbeatRequest,
    HeartbeatResponse,
    ModelRef,
    RuntimeStatus,
)

from .artifact_manager import (
    ArtifactError,
    ArtifactManager,
    ChecksumMismatch,
    DownloadAborted,
    LocalModel,
)
from .client import (
    ArtifactForbiddenError,
    ArtifactNotReadyError,
    AuthError,
    ControlPlaneClient,
    GenerationStaleError,
    NotFoundError,
    ProtocolError,
    TransientError,
)
from .config import AgentSettings
from .runtime.base import InferenceRuntimeError, ModelRuntime
from .state import AgentState, DeployedModel, StateStore
from .util import now_utc

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReconcileOutcome:
    """What one pass did. Returned for tests and for the loop's logging."""

    changed: bool
    state: AgentState
    note: str

    @property
    def actual_state(self) -> ActualState:
        return self.state.actual_state


class Reconciler:
    def __init__(
        self,
        settings: AgentSettings,
        client: ControlPlaneClient,
        artifacts: ArtifactManager,
        runtime: ModelRuntime,
        state_store: StateStore,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._client = client
        self._artifacts = artifacts
        self._runtime = runtime
        self._states = state_store
        self._clock = clock
        self.state = state_store.load()
        # Set by the loop so a mid-download generation change can abort the
        # transfer. Left None in tests that drive `reconcile` directly.
        self._abort_probe: Callable[[int], bool] | None = None

    # -- public surface ----------------------------------------------------

    def reconcile(self, desired: DesiredStateResponse) -> ReconcileOutcome:
        """Converge the device onto `desired`. Idempotent by construction."""
        before = _snapshot(self.state)

        if desired.generation < self.state.observed_generation:
            # Spec SS6: ordering is by generation, never timestamps. An out-of-order
            # or replayed response from a stale cache is simply dropped.
            return self._outcome(
                before,
                f"ignoring stale generation {desired.generation} "
                f"(already at {self.state.observed_generation})",
            )

        if self._backoff_blocks(desired):
            remaining = int(self.state.next_retry_monotonic - self._clock())
            return self._outcome(before, f"holding FAILED for {remaining}s more before retry")

        if desired.generation > self.state.observed_generation and self.state.failure_count:
            # A new instruction. Clear the penalty earned by the old one.
            self.state.failure_count = 0
            self.state.failed_generation = None
            self.state.next_retry_monotonic = 0.0

        try:
            if desired.desired_state is DesiredState.REVOKED:
                note = self._apply_revoke(desired)
            elif desired.desired_state is DesiredState.STOPPED:
                note = self._apply_stop(desired)
            else:
                note = self._apply_running(desired)
        except DownloadAborted as exc:
            # Not a failure. The partial file stays; the next pass sees the newer
            # generation and acts on that instead.
            note = str(exc)
        except _WaitingOnServer as exc:
            note = str(exc)
        except (ArtifactForbiddenError, GenerationStaleError) as exc:
            # The control plane pulled this generation's artifact out from under us
            # -- either a revoke landed, or a newer deployment superseded it. Both
            # mean "stop working on this"; the next poll carries the truth. Acking
            # would claim convergence on an instruction we abandoned.
            note = f"abandoned: {exc}"
            log.info("abandoning generation %d: %s", desired.generation, exc)
        except (ChecksumMismatch, ArtifactError, InferenceRuntimeError) as exc:
            note = self._fail(desired, exc)
        except (TransientError, ArtifactNotReadyError) as exc:
            # The control plane or the network, not the artifact. Keep whatever we
            # were doing and try again; do not ack, do not fail.
            note = f"transient: {exc}"
            log.warning("reconcile deferred: %s", exc)
        except (AuthError, NotFoundError, ProtocolError) as exc:
            # A human has to fix this. Report it so it is visible in the
            # dashboard if the heartbeat still gets through, but do not ack the
            # generation -- nothing converged.
            note = f"blocked: {exc}"
            self.state.message = str(exc)
            log.error("reconcile blocked: %s", exc)

        self._persist()
        return self._outcome(before, note)

    def report(self) -> HeartbeatResponse | None:
        """Send a heartbeat. Returns None if it could not be delivered.

        Connectivity is derived server-side from heartbeat age (spec SS7), so a
        heartbeat must go out every tick regardless of whether anything changed --
        silence is how the dashboard learns a device went offline.
        """
        try:
            response = self._client.send_heartbeat(self._heartbeat())
        except (TransientError, AuthError, NotFoundError, ProtocolError) as exc:
            log.warning("heartbeat failed: %s", exc)
            return None
        return response

    def tick(self) -> ReconcileOutcome | None:
        """One full pass: poll, converge, report.

        Reporting happens even when the poll failed, so a control plane that is up
        but slow on one route still sees the device as alive.
        """
        try:
            desired = self._client.fetch_desired_state()
        except (TransientError, AuthError, NotFoundError, ProtocolError) as exc:
            log.warning("could not fetch desired state: %s", exc)
            self.report()
            return None

        outcome = self.reconcile(desired)
        response = self.report()
        if response is not None and response.generation > self.state.observed_generation:
            # The operator clicked something while we were working. Say so; the
            # loop will shorten its sleep rather than waiting out the interval.
            log.info(
                "control plane is at generation %d, we are at %d",
                response.generation,
                self.state.observed_generation,
            )
        return outcome

    # -- desired-state handlers --------------------------------------------

    def _apply_running(self, desired: DesiredStateResponse) -> str:
        # Readiness is checked before anything else, including whether a model is
        # attached. When the control plane has not finished materializing, `model`
        # is legitimately null -- `ModelRef.sha256` is required and the digest does
        # not exist yet -- so a model-shaped check first would misread "not ready"
        # as "nothing to deploy", ack the generation, and report IDLE while the
        # previous version is still happily serving. Both halves of that are lies.
        if not desired.artifact_ready:
            # Keep the current actual_state and do NOT ack. Governance correctly
            # reads OUT_OF_SYNC until the bytes exist.
            raise _WaitingOnServer(
                f"generation {desired.generation} waiting on the control plane to "
                "materialize the artifact"
            )

        model = desired.model
        if model is None:
            # artifact_ready with no model is a control-plane bug, not a device
            # problem. Report it and do not ack: claiming convergence on an
            # instruction we cannot even read would hide the bug behind a green
            # dashboard.
            self.state.message = "control plane reported artifact_ready with no model"
            log.error("generation %d: %s", desired.generation, self.state.message)
            return self.state.message

        converged = (
            self.state.matches(model.name, model.version, model.sha256)
            and self.state.actual_state is ActualState.RUNNING
            and self._runtime.is_running
        )
        if converged:
            self._ack(desired)
            return f"already running {model.name}/{model.version}"

        if self.state.model is not None and not self.state.matches(
            model.name, model.version, model.sha256
        ):
            # An upgrade (or a rollback). Stop the old model before loading the new
            # one; two sessions holding the same GPU is a real failure on a Jetson.
            log.info(
                "replacing %s/%s with %s/%s",
                self.state.model.name,
                self.state.model.version,
                model.name,
                model.version,
            )
            self._stop_runtime()

        self._set(ActualState.DOWNLOADING, f"fetching {model.name}/{model.version}")
        local = self._artifacts.ensure(model, should_abort=self._abort_for(desired.generation))

        self._set(ActualState.DEPLOYING, f"loading {model.name}/{model.version}")
        self._runtime.load(str(local.load_target), name=model.name, version=model.version)
        self._runtime.start()

        self.state.model = DeployedModel(
            name=local.name,
            version=local.version,
            sha256=local.sha256,
            path=str(local.path),
            entrypoint=str(local.entrypoint) if local.entrypoint else None,
        )
        self.state.inference_running = True
        self.state.actual_state = ActualState.RUNNING
        self.state.message = None
        self._ack(desired)

        # Reclaim disk from versions this one superseded. After the ack, and
        # swallowing errors: a device that is serving the right model must not be
        # reported as failed because a cleanup unlink lost a race.
        try:
            self._artifacts.prune(model.name, keep_versions=(model.version,))
        except OSError as exc:  # pragma: no cover - best effort
            log.warning("pruning old versions of %s failed: %s", model.name, exc)

        return f"running {model.name}/{model.version}"

    def _apply_stop(self, desired: DesiredStateResponse) -> str:
        """Stop inference but keep the artifacts (spec SS5).

        Stop is reversible and local: the bytes stay on disk so a later RUNNING at
        the same version is instant rather than a cold re-download.
        """
        already = (
            self.state.actual_state in (ActualState.STOPPED, ActualState.IDLE)
            and not self._runtime.is_running
        )
        if already:
            self._ack(desired)
            return "already stopped"

        self._set(ActualState.STOPPING, "stopping inference")
        self._stop_runtime()
        self.state.actual_state = ActualState.STOPPED
        self.state.inference_running = False
        self.state.message = None
        self._ack(desired)
        return "stopped"

    def _apply_revoke(self, desired: DesiredStateResponse) -> str:
        """Stop inference *and* destroy the artifacts (spec SS5).

        Revoke is the irreversible one. It removes the model, the downloaded
        archive and any derived engines, and the device must not restart the model
        afterwards -- which it cannot, because the files are gone and only a newer
        generation carrying RUNNING will cause another download.

        It clears *everything* local, not merely the version named in the
        instruction. `ArtifactManager.prune(keep=2)` deliberately retains the
        previous version so a rollback is instant, so revoking v2 while v1 sat in
        that cache would leave a complete, runnable copy of the model -- unpacked
        ONNX included -- on a device whose authorization was just withdrawn, while
        the dashboard showed REVOKED. The badge has to be true.
        """
        if self.state.actual_state is ActualState.REVOKED and self.state.model is None:
            self._ack(desired)
            return "already revoked"

        self._set(ActualState.REVOKING, "revoking model")
        self._stop_runtime()
        try:
            self._runtime.unload()
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("runtime unload during revoke failed: %s", exc)

        target = desired.model or self.state.model
        if target is None:
            # Told to revoke with no idea what was deployed (a lost state file).
            # Same action, worth its own log line: an operator reading the journal
            # should see that the device could not name what it destroyed.
            log.warning("revoke with no known model; clearing all local artifacts")
        else:
            log.info("revoking %s/%s and every other local artifact", target.name, target.version)
        self._artifacts.remove_all()

        self.state.model = None
        self.state.inference_running = False
        self.state.actual_state = ActualState.REVOKED
        self.state.message = None
        self._ack(desired)
        return "revoked"

    # -- state bookkeeping -------------------------------------------------

    def _ack(self, desired: DesiredStateResponse) -> None:
        """Record convergence on this generation.

        The only place `observed_generation` moves forward on success, so the
        "acked only when converged" invariant is enforceable by reading one method.
        """
        self.state.observed_generation = desired.generation
        self.state.failure_count = 0
        self.state.failed_generation = None
        self.state.next_retry_monotonic = 0.0

    def _fail(self, desired: DesiredStateResponse, exc: Exception) -> str:
        """Enter FAILED for this generation, with backoff.

        The generation *is* acked here even though nothing converged. That looks
        wrong and is deliberate: without it the dashboard cannot tell which
        instruction failed, and spec SS22 requires a checksum failure to surface as
        FAILED against the deployment that caused it.
        """
        self.state.actual_state = ActualState.FAILED
        self.state.inference_running = False
        self.state.message = str(exc)
        self.state.observed_generation = desired.generation

        if self.state.failed_generation == desired.generation:
            self.state.failure_count += 1
        else:
            self.state.failed_generation = desired.generation
            self.state.failure_count = 1

        delay = min(
            self._settings.retry_backoff_initial_seconds * (2 ** (self.state.failure_count - 1)),
            self._settings.retry_backoff_max_seconds,
        )
        self.state.next_retry_monotonic = self._clock() + delay
        log.error(
            "generation %d failed (attempt %d): %s -- retrying in %ds",
            desired.generation,
            self.state.failure_count,
            exc,
            delay,
        )
        return f"FAILED: {exc}"

    def _backoff_blocks(self, desired: DesiredStateResponse) -> bool:
        if self.state.failed_generation != desired.generation:
            return False
        if self.state.actual_state is not ActualState.FAILED:
            return False
        return self._clock() < self.state.next_retry_monotonic

    def _set(self, state: ActualState, message: str | None = None) -> None:
        """Record an in-flight state and persist immediately.

        Persisting mid-transition matters: a power cut during a download should
        leave the device reporting DOWNLOADING, not a stale RUNNING that was true
        ten minutes ago.
        """
        self.state.actual_state = state
        self.state.message = message
        self._persist()

    def _stop_runtime(self) -> None:
        try:
            self._runtime.stop()
        except Exception as exc:  # pragma: no cover - defensive
            # A runtime that cannot stop must not block a revoke. Log and carry on
            # to artifact deletion, which is the part that actually matters.
            log.warning("runtime stop failed: %s", exc)
        self.state.inference_running = False

    def _persist(self) -> None:
        self._states.save(self.state)

    def _outcome(self, before: tuple, note: str) -> ReconcileOutcome:
        changed = _snapshot(self.state) != before
        if changed:
            log.info("reconcile: %s", note)
        else:
            log.debug("reconcile: %s", note)
        return ReconcileOutcome(changed=changed, state=self.state, note=note)

    # -- mid-download abort ------------------------------------------------

    def _abort_for(self, generation: int) -> Callable[[], bool] | None:
        if self._abort_probe is None:
            return None
        probe = self._abort_probe
        return lambda: probe(generation)

    def set_abort_probe(self, probe: Callable[[int], bool] | None) -> None:
        """Install the callback that decides whether to abandon a download.

        Injected rather than built in so tests can make generation changes happen
        deterministically instead of racing a real poll.
        """
        self._abort_probe = probe

    # -- heartbeat ---------------------------------------------------------

    def _heartbeat(self) -> HeartbeatRequest:
        model = None
        if self.state.model is not None:
            model = ModelRef(
                name=self.state.model.name,
                version=self.state.model.version,
                sha256=self.state.model.sha256,
                # The local path, not the control-plane URL: this field reports
                # where the bytes actually are on *this* device, which is what an
                # operator debugging a Jetson wants to see.
                artifact_uri=f"file://{self.state.model.path}",
                entrypoint=self.state.model.entrypoint,
            )
        try:
            hardware = HardwareInfo(**self._runtime.hardware_info())
        except Exception:  # pragma: no cover - never let telemetry break a heartbeat
            hardware = HardwareInfo()
        return HeartbeatRequest(
            device_id=self._settings.device_id,
            timestamp=now_utc(),
            observed_generation=self.state.observed_generation,
            actual_state=self.state.actual_state,
            model=model,
            runtime=RuntimeStatus(
                inference_running=self._runtime.is_running,
                detail=self._runtime.name,
            ),
            hardware=hardware,
            message=self.state.message,
        )


class _WaitingOnServer(RuntimeError):
    """Internal: the control plane is not ready yet. Never a failure."""


def _snapshot(state: AgentState) -> tuple:
    """The fields whose change is worth reporting. Backoff counters excluded --
    a ticking retry timer is not a state change."""
    return (
        state.observed_generation,
        state.actual_state,
        state.inference_running,
        None if state.model is None else (state.model.name, state.model.version, state.model.sha256),
        state.message,
    )


__all__ = [
    "ArtifactForbiddenError",
    "GenerationStaleError",
    "LocalModel",
    "ReconcileOutcome",
    "Reconciler",
]
