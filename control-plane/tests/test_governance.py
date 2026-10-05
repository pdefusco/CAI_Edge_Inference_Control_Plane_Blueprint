"""Governance and connectivity derivation.

Spec SS4 is the reason this file exists: the server derives compliance from
desired vs. actual, rather than asking the agent to assess itself. A device that
could report its own governance status could lie about it, by bug or otherwise --
so these are pure functions over two rows, and this is where the demo's central
claim is actually pinned down.

Covers the last two of the twelve SS22 cases, which are server-side:
"device heartbeat updates actual state" (via the view) and "offline device
detected from heartbeat age".
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from lighthouse_contracts import (
    GPU_PROVIDERS,
    Acceleration,
    ActualState,
    Connectivity,
    DesiredState,
    GovernanceStatus,
)

from lighthouse.config import Settings
from lighthouse.repositories import ActualDeploymentRow, DesiredDeploymentRow, DeviceRow
from lighthouse.services.governance import (
    build_device_view,
    derive_acceleration,
    derive_connectivity,
    derive_governance,
)
from lighthouse.util import now_utc

SHA_A = "a" * 64
SHA_B = "b" * 64


@pytest.fixture
def settings(tmp_path):
    return Settings(env="local", data_dir=tmp_path, registry_impl="fake", admin_token="lha_x" * 4)


def desired(
    *,
    generation: int = 1,
    state: DesiredState = DesiredState.RUNNING,
    name: str | None = "fashion-cnn",
    version: str | None = "1",
    sha: str | None = SHA_A,
) -> DesiredDeploymentRow:
    return DesiredDeploymentRow(
        device_id="d1",
        generation=generation,
        desired_state=state,
        model_name=name,
        model_version=version,
        artifact_sha256=sha,
    )


def actual(
    *,
    observed: int = 1,
    state: ActualState = ActualState.RUNNING,
    name: str | None = "fashion-cnn",
    version: str | None = "1",
    sha: str | None = SHA_A,
    running: bool = True,
    hardware: dict | None = None,
) -> ActualDeploymentRow:
    return ActualDeploymentRow(
        device_id="d1",
        observed_generation=observed,
        actual_state=state,
        model_name=name,
        model_version=version,
        artifact_sha256=sha,
        inference_running=running,
        hardware=hardware if hardware is not None else {},
    )


# --------------------------------------------------------------------------
# Connectivity: SS22 "offline device detected from heartbeat age"
# --------------------------------------------------------------------------


def test_never_seen_is_distinct_from_offline(settings):
    """An enrolled device that never checked in is a provisioning problem; one
    that went quiet is an operational problem. Collapsing them would send an
    operator looking in the wrong place."""
    assert derive_connectivity(None, settings) is Connectivity.NEVER_SEEN


@pytest.mark.parametrize(
    ("age_seconds", "expected"),
    [
        (0, Connectivity.ONLINE),
        (5, Connectivity.ONLINE),
        (45, Connectivity.STALE),
        (120, Connectivity.OFFLINE),
        (86_400, Connectivity.OFFLINE),
    ],
)
def test_connectivity_follows_heartbeat_age(settings, age_seconds, expected):
    now = now_utc()
    last_seen = now - timedelta(seconds=age_seconds)

    assert derive_connectivity(last_seen, settings, now=now) is expected


def test_connectivity_boundaries_are_inclusive(settings):
    """Exactly at the threshold counts as the healthier side, so a device
    heartbeating precisely on the interval does not flap between ONLINE and
    STALE on scheduler jitter alone."""
    now = now_utc()
    at_online = now - timedelta(seconds=settings.online_threshold_seconds)
    at_stale = now - timedelta(seconds=settings.stale_threshold_seconds)

    assert derive_connectivity(at_online, settings, now=now) is Connectivity.ONLINE
    assert derive_connectivity(at_stale, settings, now=now) is Connectivity.STALE


def test_killing_the_agent_eventually_reads_offline(settings):
    """M1 acceptance: stopping the agent must make the device go STALE then
    OFFLINE from heartbeat age alone -- no sweeper, no stored boolean."""
    now = now_utc()
    last = now - timedelta(seconds=1)

    seen = [
        derive_connectivity(last, settings, now=now + timedelta(seconds=offset))
        for offset in (0, 45, 300)
    ]

    assert seen == [Connectivity.ONLINE, Connectivity.STALE, Connectivity.OFFLINE]


# --------------------------------------------------------------------------
# Governance: the happy path and the two "nothing known yet" cases
# --------------------------------------------------------------------------


def test_matching_desired_and_actual_is_healthy():
    assert derive_governance(desired(), actual()) is GovernanceStatus.HEALTHY


def test_no_desired_row_is_unknown():
    assert derive_governance(None, actual()) is GovernanceStatus.UNKNOWN


def test_freshly_enrolled_device_with_nothing_asked_of_it_is_healthy():
    """Generation 0 with no model is a correctly idle device, not an unhealthy one.

    A dashboard that lights up UNKNOWN for every newly enrolled device trains its
    operator to ignore the column.
    """
    row = DesiredDeploymentRow(device_id="d1", generation=0, desired_state=DesiredState.STOPPED)

    assert derive_governance(row, None) is GovernanceStatus.HEALTHY


def test_deployment_requested_but_never_acknowledged_is_unknown():
    assert derive_governance(desired(), None) is GovernanceStatus.UNKNOWN


def test_device_reporting_unknown_is_unknown():
    assert derive_governance(desired(), actual(state=ActualState.UNKNOWN)) is GovernanceStatus.UNKNOWN


# --------------------------------------------------------------------------
# Governance: generation is checked before state
# --------------------------------------------------------------------------


def test_unacknowledged_generation_is_out_of_sync():
    """Mid-upgrade: still running v1, told to run v2.

    OUT_OF_SYNC is the honest answer. Reporting HEALTHY because *something* is
    running would hide every failed rollout.
    """
    verdict = derive_governance(
        desired(generation=2, version="2", sha=SHA_B),
        actual(observed=1, version="1", sha=SHA_A),
    )

    assert verdict is GovernanceStatus.OUT_OF_SYNC


def test_right_model_for_the_wrong_reason_is_out_of_sync():
    """Desired v1 at generation 5, device acked generation 4 and happens to run v1.

    The state matches but the instruction was never seen, so the match is a
    coincidence -- and a system that treats coincidence as compliance cannot be
    used as an audit trail.
    """
    verdict = derive_governance(desired(generation=5), actual(observed=4))

    assert verdict is GovernanceStatus.OUT_OF_SYNC


def test_observed_ahead_of_desired_is_still_healthy():
    """The device acked a generation this read has not caught up with yet. A
    strict equality check here would make every read during a write flap."""
    assert derive_governance(desired(generation=3), actual(observed=4)) is GovernanceStatus.HEALTHY


# --------------------------------------------------------------------------
# Governance: FAILED outranks everything
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state", [DesiredState.RUNNING, DesiredState.STOPPED, DesiredState.REVOKED]
)
def test_failed_outranks_every_desired_state(state):
    """A failed reconciliation is the one thing an operator must not have to go
    looking for, so it is reported regardless of what was asked."""
    verdict = derive_governance(
        desired(state=state), actual(state=ActualState.FAILED, running=False)
    )

    assert verdict is GovernanceStatus.FAILED


# --------------------------------------------------------------------------
# Governance: stop
# --------------------------------------------------------------------------


def test_stop_before_the_device_sees_it_is_stop_pending():
    """The transition the dashboard must not hide: Desired STOPPED, Actual
    RUNNING, status STOP_PENDING."""
    verdict = derive_governance(
        desired(generation=2, state=DesiredState.STOPPED),
        actual(observed=1, state=ActualState.RUNNING),
    )

    assert verdict is GovernanceStatus.STOP_PENDING


def test_stop_acknowledged_but_still_running_is_stop_pending():
    """Acked the generation, reports RUNNING anyway -- a device that said "got it"
    and did not comply is precisely what this status is for."""
    verdict = derive_governance(
        desired(generation=2, state=DesiredState.STOPPED),
        actual(observed=2, state=ActualState.RUNNING),
    )

    assert verdict is GovernanceStatus.STOP_PENDING


@pytest.mark.parametrize("reported", [ActualState.STOPPED, ActualState.IDLE])
def test_stop_confirmed_is_healthy(reported):
    """Healthy means "doing what it was told", not "running a model"."""
    verdict = derive_governance(
        desired(generation=2, state=DesiredState.STOPPED),
        actual(observed=2, state=reported, running=False),
    )

    assert verdict is GovernanceStatus.HEALTHY


# --------------------------------------------------------------------------
# Governance: revoke
# --------------------------------------------------------------------------


def test_revoke_before_confirmation_is_revoke_pending():
    verdict = derive_governance(
        desired(generation=3, state=DesiredState.REVOKED),
        actual(observed=2, state=ActualState.RUNNING),
    )

    assert verdict is GovernanceStatus.REVOKE_PENDING


def test_revoke_is_not_satisfied_by_merely_stopping():
    """STOPPED keeps the artifacts; REVOKED destroys them. Accepting STOPPED as
    proof of revocation would report a model as destroyed while its bytes are
    still sitting on a device that can be stolen off a shelf."""
    verdict = derive_governance(
        desired(generation=3, state=DesiredState.REVOKED),
        actual(observed=3, state=ActualState.STOPPED, running=False),
    )

    assert verdict is GovernanceStatus.REVOKE_PENDING


def test_revoke_confirmed_is_revoked():
    verdict = derive_governance(
        desired(generation=3, state=DesiredState.REVOKED),
        actual(observed=3, state=ActualState.REVOKED, name=None, version=None, running=False),
    )

    assert verdict is GovernanceStatus.REVOKED


# --------------------------------------------------------------------------
# Governance: the digest comparison
# --------------------------------------------------------------------------


def test_same_version_label_different_bytes_is_out_of_sync():
    """The nastiest case in the system, and the reason the digest is compared.

    A registry version label repointed at new bytes leaves name and version
    matching perfectly while the device serves something else entirely. Without
    this check the dashboard reports HEALTHY and the audit trail is worthless.
    """
    verdict = derive_governance(desired(sha=SHA_A), actual(sha=SHA_B))

    assert verdict is GovernanceStatus.OUT_OF_SYNC


def test_missing_digest_on_either_side_does_not_force_out_of_sync():
    """An older agent, or a device that has not reported a digest yet, must not be
    permanently OUT_OF_SYNC for a field it never sends."""
    assert derive_governance(desired(sha=SHA_A), actual(sha=None)) is GovernanceStatus.HEALTHY
    assert derive_governance(desired(sha=None), actual(sha=SHA_A)) is GovernanceStatus.HEALTHY


def test_wrong_version_is_out_of_sync():
    assert derive_governance(desired(version="2"), actual(version="1")) is GovernanceStatus.OUT_OF_SYNC


def test_wrong_model_name_is_out_of_sync():
    assert derive_governance(desired(), actual(name="other-model")) is GovernanceStatus.OUT_OF_SYNC


@pytest.mark.parametrize(
    "state",
    [
        ActualState.DOWNLOADING,
        ActualState.DEPLOYING,
        ActualState.STOPPED,
        ActualState.IDLE,
        ActualState.REVOKED,
    ],
)
def test_anything_short_of_running_is_out_of_sync_when_running_is_desired(state):
    verdict = derive_governance(desired(), actual(state=state, running=False))

    assert verdict is GovernanceStatus.OUT_OF_SYNC


# --------------------------------------------------------------------------
# build_device_view: SS22 "device heartbeat updates actual state"
# --------------------------------------------------------------------------


def test_device_view_shows_desired_and_actual_side_by_side(settings):
    """The spec's dashboard requirement, asserted on the payload rather than the
    HTML: both sides are present and neither is collapsed into the other."""
    now = now_utc()
    device = DeviceRow(
        device_id="d1",
        display_name="Jetson",
        platform="jetson-orin",
        registered_at=now,
        last_seen=now,
    )

    view = build_device_view(
        device,
        desired(generation=2, state=DesiredState.STOPPED, version="2"),
        actual(observed=1, state=ActualState.RUNNING, version="1"),
        settings,
        now=now,
    )

    assert view.desired_state is DesiredState.STOPPED
    assert view.desired_model_version == "2"
    assert view.generation == 2
    assert view.actual_state is ActualState.RUNNING
    assert view.actual_model_version == "1"
    assert view.observed_generation == 1
    assert view.inference_running is True
    assert view.governance_status is GovernanceStatus.STOP_PENDING
    assert view.connectivity is Connectivity.ONLINE


def test_device_view_for_a_device_that_never_reported(settings):
    device = DeviceRow(device_id="d1", registered_at=now_utc())

    view = build_device_view(device, None, None, settings)

    assert view.actual_state is ActualState.UNKNOWN
    assert view.observed_generation is None
    assert view.inference_running is False
    assert view.connectivity is Connectivity.NEVER_SEEN
    assert view.generation == 0


# --------------------------------------------------------------------------
# Acceleration: the Phase 6 acceptance field
# --------------------------------------------------------------------------


@pytest.mark.parametrize("provider", sorted(GPU_PROVIDERS))
def test_every_provider_in_the_shared_set_reads_accelerated(provider):
    """Parametrized over `GPU_PROVIDERS` itself rather than a hand-copied list,
    so adding a provider to the set cannot silently skip being tested here."""
    row = actual(hardware={"active_providers": [provider, "CPUExecutionProvider"]})

    assert derive_acceleration(row) is Acceleration.ACCELERATED


def test_the_verdict_comes_from_active_providers_not_from_availability():
    """The single most expensive thing this function could get wrong, because it
    is also the most reassuring: a build that advertises CUDA while the loaded
    session runs every node on the CPU. A provider that cannot handle a node
    falls back silently and per-node, so availability is not evidence."""
    row = actual(
        hardware={
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "active_providers": ["CPUExecutionProvider"],
            # The agent's own summary of the same availability list, and equally
            # not evidence about the session.
            "gpu_available": True,
        }
    )

    assert derive_acceleration(row) is Acceleration.CPU_ONLY


def test_coreml_and_azure_do_not_count_as_the_accelerator():
    """Both are non-CPU and neither is what the gate asks about: Azure is a
    *remote* inference endpoint, and CoreML does use a GPU but not a Jetson's.
    The development MacBook reports exactly this list, so a `!=
    "CPUExecutionProvider"` test would make a laptop pass the Jetson's gate."""
    row = actual(
        hardware={
            "active_providers": [
                "CoreMLExecutionProvider",
                "AzureExecutionProvider",
                "CPUExecutionProvider",
            ]
        }
    )

    assert derive_acceleration(row) is Acceleration.CPU_ONLY


@pytest.mark.parametrize(
    "hardware",
    [
        pytest.param({}, id="no hardware at all -- an agent that sent none"),
        pytest.param({"platform": "Linux-aarch64"}, id="hardware but no providers key"),
        pytest.param({"active_providers": []}, id="an empty list"),
        pytest.param({"active_providers": None}, id="an explicit null"),
        pytest.param({"active_providers": "CUDAExecutionProvider"}, id="a bare string"),
        pytest.param({"active_providers": {"0": "CUDA"}}, id="a hand-edited object"),
    ],
)
def test_nothing_reported_reads_unknown_rather_than_cpu_only(hardware):
    """`UNKNOWN` is not padding. A `mock` runtime, an agent with no session
    loaded, and an agent too old to send the key all land here, and calling any
    of them `CPU_ONLY` would be a confident claim about hardware nobody
    measured. The gate fails them either way -- not proven is not passed -- but
    it is a different sentence, and the difference is what says whether to go
    look at the wheel or at the agent.

    The last three cases came out of a JSON column and could have been
    hand-edited, so they must read as "not reported" rather than raise inside a
    listing of the whole fleet.
    """
    assert derive_acceleration(actual(hardware=hardware)) is Acceleration.UNKNOWN


def test_a_device_that_never_reported_reads_unknown():
    assert derive_acceleration(None) is Acceleration.UNKNOWN


def test_a_cpu_only_device_is_still_healthy_governance(settings):
    """The pair an operator has to be able to read, and the reason this is not a
    `GovernanceStatus` value. This device downloaded the right bytes, loaded them
    and is serving them, so it *is* in sync with what it was told -- and it is
    failing Phase 6's acceptance check. A failed acceptance check that is
    correctly HEALTHY governance; folding one into the other would break the only
    thing governance means."""
    device = DeviceRow(device_id="d1", registered_at=now_utc(), last_seen=now_utc())
    row = actual(hardware={"active_providers": ["CPUExecutionProvider"], "smoke_check": "passed"})

    view = build_device_view(device, desired(), row, settings)

    assert view.governance_status is GovernanceStatus.HEALTHY
    assert view.acceleration is Acceleration.CPU_ONLY


def test_the_view_carries_the_hardware_dict_the_device_sent(settings):
    """Keys no schema names included -- `active_providers`, `smoke_check` and
    `device_model` appear in `HardwareInfo` nowhere, and they are the three the
    acceptance check is about."""
    device = DeviceRow(device_id="d1", registered_at=now_utc(), last_seen=now_utc())
    reported = {
        "device_model": "NVIDIA Orin Nano Developer Kit",
        "active_providers": ["TensorrtExecutionProvider", "CPUExecutionProvider"],
        "smoke_check": "passed",
        "tegra_temp_c": 44.5,
    }

    view = build_device_view(device, desired(), actual(hardware=reported), settings)

    assert view.hardware == reported
    assert view.acceleration is Acceleration.ACCELERATED


def test_the_view_does_not_hand_out_the_rows_own_dict(settings):
    """The row's dict belongs to the repository layer, and the in-memory store
    hands back live rows. A view sharing that object would let anything holding
    the view mutate what the next read reports."""
    device = DeviceRow(device_id="d1", registered_at=now_utc(), last_seen=now_utc())
    row = actual(hardware={"smoke_check": "passed"})

    view = build_device_view(device, desired(), row, settings)
    view.hardware["smoke_check"] = "failed: tampered with"

    assert row.hardware["smoke_check"] == "passed"


def test_device_view_surfaces_the_failure_message(settings):
    """A checksum mismatch on a Jetson is only useful if its text reaches the
    dashboard; otherwise the operator has to SSH in to learn why."""
    device = DeviceRow(device_id="d1", registered_at=now_utc(), last_seen=now_utc())
    failed = actual(state=ActualState.FAILED, running=False)
    failed.message = "fashion-cnn/1: expected sha256 aaa..., got bbb..."

    view = build_device_view(device, desired(), failed, settings)

    assert view.governance_status is GovernanceStatus.FAILED
    assert "expected sha256" in view.message
