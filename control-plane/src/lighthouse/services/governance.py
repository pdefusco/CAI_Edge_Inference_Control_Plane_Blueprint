"""Derivation of governance status and connectivity.

Spec SS4: "The server should derive governance/compliance status from desired vs.
actual state rather than asking the agent to decide governance policy." That is
the whole point of the demo -- CAI knows what each device is *actually* running,
and it knows it because it compares, not because the device asserts compliance.

Both values here are computed on every read and stored nowhere. A persisted
`online` boolean would need a background sweeper to ever go false, and would be
wrong for exactly as long as that sweeper was behind; a persisted governance
status would go stale the moment either side changed.
"""

from __future__ import annotations

from datetime import datetime

from lighthouse_contracts import (
    GPU_PROVIDERS,
    Acceleration,
    ActualState,
    Connectivity,
    DesiredState,
    DeviceView,
    GovernanceStatus,
)

from ..config import Settings
from ..repositories import ActualDeploymentRow, DesiredDeploymentRow, DeviceRow
from ..util import now_utc

# Actual states that mean "inference is not running and the artifacts are gone".
_REVOKED_STATES = {ActualState.REVOKED}
# Actual states that mean "inference is not running".
_STOPPED_STATES = {ActualState.STOPPED, ActualState.IDLE, ActualState.REVOKED}


def derive_connectivity(
    last_seen: datetime | None,
    settings: Settings,
    *,
    now: datetime | None = None,
) -> Connectivity:
    """Classify a device by heartbeat age (spec SS14).

    NEVER_SEEN is distinct from OFFLINE on purpose: an enrolled device that has
    not yet checked in once is a provisioning problem, while a device that went
    quiet is an operational one, and an operator needs to tell those apart.
    """
    if last_seen is None:
        return Connectivity.NEVER_SEEN
    age = (now or now_utc()) - last_seen
    seconds = age.total_seconds()
    if seconds <= settings.online_threshold_seconds:
        return Connectivity.ONLINE
    if seconds <= settings.stale_threshold_seconds:
        return Connectivity.STALE
    return Connectivity.OFFLINE


def derive_governance(
    desired: DesiredDeploymentRow | None,
    actual: ActualDeploymentRow | None,
) -> GovernanceStatus:
    """Compare desired and actual state into a single operator-facing verdict."""
    if desired is None:
        return GovernanceStatus.UNKNOWN

    if actual is None or actual.actual_state is ActualState.UNKNOWN:
        # Nothing has been reported. If nothing has been asked for either, that
        # is a correctly idle device rather than an unknown one.
        if desired.generation == 0 and desired.model_name is None:
            return GovernanceStatus.HEALTHY
        return GovernanceStatus.UNKNOWN

    # A failed reconciliation outranks everything else: it is the one state an
    # operator must not have to go looking for.
    if actual.actual_state is ActualState.FAILED:
        return GovernanceStatus.FAILED

    # Has the device even seen the current instruction? Generation is the
    # ordering authority, so this is checked before any state comparison --
    # a device still running the right model for the wrong reason is not healthy.
    acknowledged = actual.observed_generation >= desired.generation

    if desired.desired_state is DesiredState.REVOKED:
        if acknowledged and actual.actual_state in _REVOKED_STATES:
            return GovernanceStatus.REVOKED
        return GovernanceStatus.REVOKE_PENDING

    if desired.desired_state is DesiredState.STOPPED:
        if acknowledged and actual.actual_state in _STOPPED_STATES:
            return GovernanceStatus.HEALTHY
        return GovernanceStatus.STOP_PENDING

    # desired RUNNING
    if not acknowledged:
        return GovernanceStatus.OUT_OF_SYNC
    if actual.actual_state is not ActualState.RUNNING:
        return GovernanceStatus.OUT_OF_SYNC
    if (actual.model_name, actual.model_version) != (desired.model_name, desired.model_version):
        return GovernanceStatus.OUT_OF_SYNC
    # Digest comparison catches the nastiest case: right name, right version
    # label, different bytes -- which is exactly what repointing a registry
    # version would produce.
    if (
        desired.artifact_sha256
        and actual.artifact_sha256
        and desired.artifact_sha256 != actual.artifact_sha256
    ):
        return GovernanceStatus.OUT_OF_SYNC
    return GovernanceStatus.HEALTHY


def derive_acceleration(actual: ActualDeploymentRow | None) -> Acceleration:
    """Whether the device's loaded session is on its accelerator.

    Reads `active_providers` and **only** `active_providers`. The sibling
    `providers` key says what the installed onnxruntime build *could* do, and
    the gap between them is the entire hazard: a provider that cannot handle a
    node falls back to the CPU silently and per-node, so a device can report
    CUDA as available while running the model on its CPU. Deriving this from
    `providers` would make a fleet-wide CPU fallback look like a fleet on the
    GPU -- which is the single most expensive thing this field could get wrong,
    because it is also the most reassuring.

    Stored nowhere, like everything else in this module: it is a reading of the
    last heartbeat, and a persisted copy would outlive the fact.
    """
    if actual is None:
        return Acceleration.UNKNOWN
    active = actual.hardware.get("active_providers")
    # `isinstance` rather than truthiness alone: this dict came back out of a
    # JSON column, so a malformed or hand-edited row must read as "not
    # reported" rather than raise inside a fleet listing.
    if not isinstance(active, list) or not active:
        return Acceleration.UNKNOWN
    if any(provider in GPU_PROVIDERS for provider in active):
        return Acceleration.ACCELERATED
    return Acceleration.CPU_ONLY


def build_device_view(
    device: DeviceRow,
    desired: DesiredDeploymentRow | None,
    actual: ActualDeploymentRow | None,
    settings: Settings,
    *,
    artifact_ready: bool = True,
    now: datetime | None = None,
) -> DeviceView:
    """Assemble the dashboard/API representation of one device.

    Desired and actual are reported side by side and never collapsed. The
    dashboard must be able to show "Desired: STOPPED / Actual: RUNNING /
    STOP_PENDING" -- hiding the transition would turn the one thing this system
    exists to demonstrate into a lie.
    """
    return DeviceView(
        device_id=device.device_id,
        display_name=device.display_name,
        platform=device.platform,
        registered_at=device.registered_at or now_utc(),
        last_seen=device.last_seen,
        connectivity=derive_connectivity(device.last_seen, settings, now=now),
        governance_status=derive_governance(desired, actual),
        generation=desired.generation if desired else 0,
        desired_state=desired.desired_state if desired else None,
        desired_model_name=desired.model_name if desired else None,
        desired_model_version=desired.model_version if desired else None,
        observed_generation=actual.observed_generation if actual else None,
        actual_state=actual.actual_state if actual else ActualState.UNKNOWN,
        actual_model_name=actual.model_name if actual else None,
        actual_model_version=actual.model_version if actual else None,
        inference_running=actual.inference_running if actual else False,
        artifact_ready=artifact_ready,
        message=actual.message if actual else None,
        # Copied, not aliased: the row's dict belongs to the repository layer and
        # a view handed out by reference is a mutation waiting to happen.
        hardware=dict(actual.hardware) if actual else {},
        acceleration=derive_acceleration(actual),
    )
