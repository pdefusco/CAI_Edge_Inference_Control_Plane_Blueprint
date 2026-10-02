"""Desired-state mutation and the device-facing read path.

This module is the one place desired state changes, which is what lets spec SS6's
rule hold: "Do not rely solely on timestamps for reconciliation ordering."
Generation allocation lives in the store transaction, and every path into it
comes through here.

Two decisions worth reading before changing anything:

**Stop and revoke preserve the full artifact lineage.** They change
`desired_state` and nothing else. Dropping `model_id`/`version_uuid` -- which is
the obvious way to write "stop" -- breaks two things at once: the cached artifact
stops being referenced and becomes evictable while a STOPPED device still needs
it to restart, and a later `REVOKED -> RUNNING` turns into a cold re-download.
SS5 requires both restart-after-stop and re-authorized deployment after revoke to
work, so the lineage has to survive the transition.

**Materialization is triggered from the operator action, not the device poll.**
A poll stays a cheap read. The one exception is retrying a previous failure, on a
cooldown -- otherwise a transient registry blip would leave a device waiting for
`artifact_ready` forever with no automatic recovery.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from lighthouse_contracts import (
    ActualState,
    ArtifactFormat,
    DesiredState,
    DesiredStateResponse,
    DeploymentRequest,
    EventType,
    HeartbeatRequest,
    HeartbeatResponse,
    ModelRef,
    Packaging,
)

from ..config import Settings
from ..registry import ModelRegistry, RegistryModelVersion
from ..repositories import (
    ActualDeploymentRow,
    DesiredDeploymentRow,
    DeviceUnknown,
    Store,
)
from ..util import now_utc, to_utc
from .artifact_service import ArtifactService
from .audit import AuditService
from .errors import DeviceNotFound, NothingDeployed

log = logging.getLogger(__name__)

# How long to wait before re-attempting a materialization that failed. Long
# enough that a polling fleet cannot hammer a struggling registry, short enough
# that a blip heals without an operator noticing.
_MATERIALIZE_RETRY_COOLDOWN = timedelta(seconds=60)


class DeploymentService:
    def __init__(
        self,
        store: Store,
        registry: ModelRegistry,
        artifacts: ArtifactService,
        audit: AuditService,
        settings: Settings,
    ) -> None:
        self._store = store
        self._registry = registry
        self._artifacts = artifacts
        self._audit = audit
        self._settings = settings

    # -- operator actions ------------------------------------------------

    def deploy(self, device_id: str, request: DeploymentRequest) -> DesiredDeploymentRow:
        """Assign a model version to a device (spec SS13).

        Validates against the registry, starts materialization, bumps the
        generation and returns. It deliberately does **not** wait for the device:
        the operator gets an immediate answer and the dashboard shows the
        transition converge.

        Registry errors propagate as RegistryError subclasses for the API layer to
        map -- an unknown model must be a clear rejection here, not a mystery the
        device discovers later.
        """
        device = self._store.get_device(device_id)
        if device is None:
            raise DeviceNotFound(device_id)

        # Raises ModelNotFound / VersionNotReady / UnsupportedFlavor. Resolving
        # before touching desired state is what keeps an unrunnable version from
        # ever becoming an instruction.
        mv = self._registry.get_version(request.model_name, request.model_version)

        # Classify before writing: once set_desired appends to history, the
        # request itself is in the history we would be comparing against.
        event_type = self._audit.classify_deployment(device_id, mv.name, mv.version)

        info = self._artifacts.request(mv)

        row = self._store.set_desired(
            DesiredDeploymentRow(
                device_id=device_id,
                desired_state=request.desired_state,
                model_name=mv.name,
                model_version=mv.version,
                registry_artifact_uri=mv.artifact_uri,
                artifact_sha256=info.sha256 if info else None,
                artifact_format=mv.format.value,
                model_id=mv.model_id,
                version_uuid=mv.version_uuid,
            )
        )

        self._audit.record(
            event_type,
            device_id=device_id,
            generation=row.generation,
            model_name=mv.name,
            model_version=mv.version,
            desired_state=request.desired_state.value,
            artifact_ready=info is not None,
            registry_artifact_uri=mv.artifact_uri,
            reason=request.reason,
        )
        log.info(
            "device %s -> generation %s %s %s v%s (artifact_ready=%s)",
            device_id,
            row.generation,
            request.desired_state.value,
            mv.name,
            mv.version,
            info is not None,
        )
        return row

    def stop(self, device_id: str, reason: str | None = None) -> DesiredDeploymentRow:
        """Ask the device to stop inference but keep its artifacts (spec SS5)."""
        return self._transition(
            device_id,
            DesiredState.STOPPED,
            EventType.STOP_REQUESTED,
            reason,
        )

    def revoke(self, device_id: str, reason: str | None = None) -> DesiredDeploymentRow:
        """Withdraw authorization: stop inference and delete artifacts (spec SS5).

        The cache entry is intentionally *not* removed here. Revocation is about
        the device's copy; the control plane's cache is shared across devices, and
        SS5 explicitly allows a later generation to re-authorize deployment --
        which should be a cache hit, not a cold download.
        """
        return self._transition(
            device_id,
            DesiredState.REVOKED,
            EventType.REVOKE_REQUESTED,
            reason,
        )

    def _transition(
        self,
        device_id: str,
        desired_state: DesiredState,
        event_type: EventType,
        reason: str | None,
    ) -> DesiredDeploymentRow:
        device = self._store.get_device(device_id)
        if device is None:
            raise DeviceNotFound(device_id)

        current = self._store.get_desired(device_id)
        if current is None or current.model_name is None:
            raise NothingDeployed(
                f"device {device_id} has no deployment to {desired_state.value.lower()}"
            )

        # Carry every artifact field forward unchanged. See the module docstring:
        # this is the whole point of the method.
        row = self._store.set_desired(
            DesiredDeploymentRow(
                device_id=device_id,
                desired_state=desired_state,
                model_name=current.model_name,
                model_version=current.model_version,
                registry_artifact_uri=current.registry_artifact_uri,
                artifact_sha256=current.artifact_sha256,
                artifact_format=current.artifact_format,
                model_id=current.model_id,
                version_uuid=current.version_uuid,
            )
        )

        self._audit.record(
            event_type,
            device_id=device_id,
            generation=row.generation,
            model_name=current.model_name,
            model_version=current.model_version,
            reason=reason,
        )
        log.info("device %s -> generation %s %s", device_id, row.generation, desired_state.value)
        return row

    # -- device-facing read ----------------------------------------------

    def desired_state_for(self, device_id: str) -> DesiredStateResponse:
        """Build the payload the agent reconciles against (spec SS8)."""
        row = self._store.get_desired(device_id)
        if row is None:
            raise DeviceNotFound(device_id)

        model: ModelRef | None = None
        artifact_ready = True

        if row.model_name is not None and row.cache_key is not None:
            info = self._artifacts.get_ready(row.cache_key)
            if info is None:
                artifact_ready = False
                self._retry_materialization_if_stale(row)
            else:
                # The digest in the payload is the digest of the bytes this
                # process will actually serve, computed on ingest -- not a value
                # copied from registry metadata. That is what makes the device's
                # verify-before-activate check meaningful.
                if row.artifact_sha256 != info.sha256:
                    self._backfill_digest(row, info.sha256)
                model = ModelRef(
                    name=row.model_name,
                    version=row.model_version or "",
                    sha256=info.sha256,
                    format=ArtifactFormat(row.artifact_format or ArtifactFormat.ONNX),
                    artifact_uri=artifact_path(device_id, row.generation),
                    packaging=info.packaging,
                    entrypoint=info.entrypoint,
                    size_bytes=info.size_bytes,
                )
        elif row.model_name is None:
            # Nothing has ever been deployed. Not a wait state -- there is
            # genuinely nothing to fetch.
            artifact_ready = True

        return DesiredStateResponse(
            device_id=device_id,
            generation=row.generation,
            desired_state=row.desired_state,
            model=model,
            artifact_ready=artifact_ready,
            poll_interval_seconds=self._settings.heartbeat_interval_seconds,
            server_time=now_utc(),
        )

    def resolve_artifact(self, device_id: str, generation: int | None):
        """Resolve the artifact a device is currently authorized to download.

        Authorization is structural rather than checked: the artifact is looked up
        through *that device's own* desired state, so there is no way to name a
        model the device was never assigned.

        Returns (ArtifactInfo | None, reason). A `generation` that does not match
        current desired state is refused -- a device finishing a download against
        a superseded instruction must restart from the new one rather than
        activate stale bytes.
        """
        row = self._store.get_desired(device_id)
        if row is None:
            raise DeviceNotFound(device_id)
        if generation is not None and generation != row.generation:
            return None, "stale_generation"
        if row.desired_state is DesiredState.REVOKED:
            # Serving bytes for a revoked deployment would undo the revocation.
            return None, "revoked"
        if row.cache_key is None:
            return None, "no_deployment"
        info = self._artifacts.get_ready(row.cache_key)
        if info is None:
            self._retry_materialization_if_stale(row)
            return None, "not_ready"
        self._store.touch_artifact(row.cache_key)
        return info, "ok"

    # -- heartbeat -------------------------------------------------------

    def record_heartbeat(self, device_id: str, hb: HeartbeatRequest) -> HeartbeatResponse:
        """Record reported actual state (spec SS7).

        The device's own `device_id` field is ignored for identity -- the caller
        has already authenticated it -- and audit events are written only on an
        actual change, or a 10-second heartbeat would bury the audit table in
        thousands of identical rows.
        """
        device = self._store.get_device(device_id)
        if device is None:
            raise DeviceNotFound(device_id)

        previous = self._store.get_actual(device_id)
        self._store.set_actual(
            ActualDeploymentRow(
                device_id=device_id,
                observed_generation=hb.observed_generation,
                actual_state=hb.actual_state,
                model_name=hb.model.name if hb.model else None,
                model_version=hb.model.version if hb.model else None,
                artifact_sha256=hb.model.sha256 if hb.model else None,
                inference_running=hb.runtime.inference_running,
                message=hb.message,
                hardware=hb.hardware.model_dump(exclude_none=True),
                updated_at=to_utc(hb.timestamp),
            )
        )
        self._store.touch_device(device_id)

        changed = previous is None or (
            previous.actual_state is not hb.actual_state
            or previous.observed_generation != hb.observed_generation
        )
        if changed:
            event = (
                EventType.RECONCILE_FAILED
                if hb.actual_state is ActualState.FAILED
                else EventType.DEVICE_STATE_CHANGED
            )
            self._audit.record(
                event,
                device_id=device_id,
                generation=hb.observed_generation,
                previous_state=previous.actual_state.value if previous else None,
                actual_state=hb.actual_state.value,
                model_name=hb.model.name if hb.model else None,
                model_version=hb.model.version if hb.model else None,
                message=hb.message,
            )

        desired = self._store.get_desired(device_id)
        return HeartbeatResponse(
            accepted=True,
            generation=desired.generation if desired else 0,
            desired_state=desired.desired_state if desired else None,
            server_time=now_utc(),
        )

    # -- internals -------------------------------------------------------

    def _retry_materialization_if_stale(self, row: DesiredDeploymentRow) -> None:
        """Re-attempt a failed materialization, at most once per cooldown.

        Without this a registry hiccup during PUT /deployment would strand the
        device on `artifact_ready: false` until a human noticed. With it, recovery
        is automatic but bounded.
        """
        if row.cache_key is None or row.model_name is None:
            return
        cached = self._store.get_artifact(row.cache_key)
        if cached is None:
            # The row vanished (evicted, or a cache directory wipe). Re-request.
            self._request_from_registry(row)
            return
        if cached.status != "FAILED":
            return  # PENDING: someone is already on it.
        age_reference = cached.completed_at or cached.created_at
        if age_reference is not None and now_utc() - age_reference < _MATERIALIZE_RETRY_COOLDOWN:
            return
        log.info("retrying failed materialization for %s", row.cache_key)
        self._request_from_registry(row)

    def _request_from_registry(self, row: DesiredDeploymentRow) -> None:
        """Re-resolve the version and restart materialization. Best-effort."""
        if row.model_name is None or row.model_version is None:
            return
        try:
            mv = self._registry.get_version(row.model_name, row.model_version)
        except Exception as exc:
            log.warning(
                "cannot re-resolve %s v%s for retry: %s", row.model_name, row.model_version, exc
            )
            return
        if mv.cache_key != row.cache_key:
            # The label was repointed at different bytes. Honoring that silently
            # would change what a device runs without a generation bump, so it is
            # left for an operator to redeploy explicitly.
            log.warning(
                "registry lineage for %s v%s changed (%s -> %s); not auto-updating desired state",
                row.model_name,
                row.model_version,
                row.cache_key,
                mv.cache_key,
            )
            return
        self._artifacts.request(mv)

    def _backfill_digest(self, row: DesiredDeploymentRow, sha256: str) -> None:
        """Record the real digest once materialization has produced it.

        PUT /deployment stores None when the artifact was not yet cached. This
        writes the value in without allocating a new generation -- the instruction
        has not changed, only the control plane's knowledge of it, and bumping the
        generation here would make every device re-reconcile for nothing.
        """
        try:
            self._store.update_desired_digest(row.device_id, sha256)
            row.artifact_sha256 = sha256
        except DeviceUnknown:  # pragma: no cover - device deleted mid-poll
            pass


def artifact_path(device_id: str, generation: int) -> str:
    """The device-relative artifact URL.

    Returned relative, never absolute: the agent joins it onto its configured
    control-plane base URL, so the same payload works through the CAI Application
    domain, an SSH tunnel, or localhost in `make dev`.
    """
    return f"/api/v1/devices/{device_id}/artifact?generation={generation}"
