"""Wire contracts shared by the control plane and the edge agent.

Both sides import these so a field name cannot drift on one side only. The spec
names the artifact checksum three different ways -- `artifact_checksum` (SS4),
`model.checksum` (SS7) and `model.sha256` (SS8). `sha256` wins here because SS8 is
the device-facing desired-state contract, and one name on the wire beats three.
The flat `artifact_sha256` spelling survives only as a database column name.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from .enums import (
    Acceleration,
    ActualState,
    ArtifactFormat,
    Connectivity,
    DesiredState,
    EventType,
    GovernanceStatus,
    Packaging,
)

# A lowercase hex SHA-256 digest. Pinned as a type so every field that carries a
# digest validates identically; there is deliberately no `algorithm` field
# anywhere -- the field name is the algorithm.
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", description="Lowercase hex SHA-256")]


class Strict(BaseModel):
    """Reject unknown fields on anything crossing the wire.

    A typo'd field name in a device payload should be a loud 422, not a value
    silently ignored while the server keeps its default.
    """

    model_config = ConfigDict(extra="forbid")


class ArtifactRef(Strict):
    """The single definition of how an artifact is identified and verified.

    Every payload that references artifact bytes inherits from this, which is
    what stops `sha256` from becoming `checksum` in one direction only.
    """

    sha256: Sha256
    format: ArtifactFormat = ArtifactFormat.ONNX


class ModelRef(ArtifactRef):
    """A model version as the device needs to see it.

    `artifact_uri` is control-plane-relative (e.g.
    `/api/v1/devices/jetson-orin-01/artifact?generation=43`), not the registry's
    `s3a://` URI -- the device has no identity in object storage and could never
    fetch that. The agent joins it onto its configured CONTROL_PLANE_URL, so the
    same payload is correct whether reached via the CAI domain or localhost.
    """

    name: str
    version: str
    artifact_uri: str
    packaging: Packaging = Packaging.MLFLOW_TAR_GZ
    entrypoint: str | None = Field(
        default=None,
        description="Path of the ONNX file within the unpacked artifact, when known",
    )
    size_bytes: int | None = Field(default=None, ge=0)


class DesiredStateResponse(Strict):
    """Response to GET /api/v1/devices/{device_id}/desired-state (spec SS8)."""

    device_id: str
    generation: int = Field(ge=0)
    desired_state: DesiredState

    # Retained for STOPPED and REVOKED rather than nulled: to stop inference the
    # agent needs to know *which* model to stop, and to revoke it needs to know
    # whose artifacts to delete. None only before anything was ever deployed.
    model: ModelRef | None = None

    # False means "the generation advanced but the control plane is still
    # materializing the bytes". The agent must treat this as wait-and-re-poll,
    # NOT as a failure -- setting FAILED here would be a self-inflicted outage.
    artifact_ready: bool = True

    poll_interval_seconds: int = Field(default=10, ge=1)
    server_time: datetime


class RuntimeStatus(Strict):
    """Edge runtime liveness (spec SS7)."""

    inference_running: bool = False
    pid: int | None = None
    detail: str | None = None


class HardwareInfo(Strict):
    """Device self-description. Free-form on purpose; the spec defers richer
    telemetry (GPU/CPU/thermal) past the first vertical slice."""

    model_config = ConfigDict(extra="allow")

    platform: str | None = None
    gpu_available: bool | None = None


class HeartbeatRequest(Strict):
    """POST /api/v1/devices/{device_id}/heartbeat (spec SS7).

    `device_id` is present because the spec's payload shows it, but the server
    authenticates the device from its bearer token and rejects a mismatch. The
    body never establishes identity.
    """

    device_id: str
    timestamp: datetime
    observed_generation: int = Field(ge=0)
    actual_state: ActualState
    model: ModelRef | None = None
    runtime: RuntimeStatus = Field(default_factory=RuntimeStatus)
    hardware: HardwareInfo = Field(default_factory=HardwareInfo)
    message: str | None = Field(
        default=None, description="Operator-facing detail, e.g. why the state is FAILED"
    )


class HeartbeatResponse(Strict):
    """Acknowledgement carrying the current desired generation.

    The agent can compare this to what it just reported and poll desired-state
    immediately instead of waiting out its interval, which makes an operator's
    dashboard click feel responsive without shortening the poll loop.
    """

    accepted: bool = True
    generation: int = Field(ge=0)
    desired_state: DesiredState | None = None
    server_time: datetime


class DeploymentRequest(Strict):
    """PUT /api/v1/devices/{device_id}/deployment (spec SS13).

    Naming an earlier version is how rollback works -- there is no separate
    rollback endpoint or state. The server classifies the audit event by
    comparing against deployment history, so an operator cannot mislabel one.
    """

    model_name: str
    model_version: str
    desired_state: DesiredState = DesiredState.RUNNING
    reason: str | None = None


class DeviceRegistrationRequest(Strict):
    device_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")
    display_name: str | None = None
    platform: str | None = None


class DeviceTokenIssued(Strict):
    """The plaintext device token, returned exactly once at issue time.

    The server stores only sha256(secret) and the public token_id, so this value
    is unrecoverable afterwards.
    """

    device_id: str
    token_id: str
    token: str = Field(description="Full bearer token: lhd_<token_id>.<secret>")


class DeviceView(Strict):
    """A device as the dashboard and operator API present it.

    Governance status and connectivity are computed at read time, which is why
    they appear here but in no table.
    """

    device_id: str
    display_name: str | None = None
    platform: str | None = None
    registered_at: datetime
    last_seen: datetime | None = None

    connectivity: Connectivity
    governance_status: GovernanceStatus

    generation: int = Field(ge=0)
    desired_state: DesiredState | None = None
    desired_model_name: str | None = None
    desired_model_version: str | None = None

    observed_generation: int | None = None
    actual_state: ActualState = ActualState.UNKNOWN
    actual_model_name: str | None = None
    actual_model_version: str | None = None
    inference_running: bool = False
    artifact_ready: bool = True
    message: str | None = None

    # The device's own self-description, verbatim from its last heartbeat's
    # `hardware`. Before this existed the control plane stored the dict in
    # `actual_deployment.hardware_json` and nothing ever read it back, so the
    # Phase 6 acceptance gate -- `active_providers` and `smoke_check` -- was
    # only checkable from the device's journal or by opening SQLite by hand.
    #
    # A `dict` rather than `HardwareInfo`, deliberately: it was already
    # validated as `HardwareInfo` on the way in, the store round-trips it as
    # JSON, and re-validating it at read time can only turn one cosmetic field
    # from one device into a 500 on the whole fleet view. `HardwareInfo` allows
    # extra keys anyway, so the fields that matter most here -- `active_providers`,
    # `smoke_check`, `device_model` -- appear in no schema either way.
    hardware: dict[str, Any] = Field(default_factory=dict)

    # The one judgement derived from that dict, because it must not be derived
    # three times. See `Acceleration` for why this cannot live inside
    # `governance_status`.
    acceleration: Acceleration = Acceleration.UNKNOWN


class AuditEventView(Strict):
    event_id: str
    timestamp: datetime
    device_id: str | None = None
    event_type: EventType
    generation: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ModelVersionView(Strict):
    """A registry model version as exposed through GET /api/v1/models."""

    name: str
    version: str
    status: str | None = None
    format: ArtifactFormat | None = None
    created_at: datetime | None = None
    registry_artifact_uri: str | None = Field(
        default=None, description="Registry-side s3a:// location, for operator display only"
    )
    deployable: bool = True
    reason: str | None = Field(default=None, description="Why not deployable, when deployable is false")


class ModelView(Strict):
    name: str
    model_id: str | None = None
    versions: list[ModelVersionView] = Field(default_factory=list)


class HealthResponse(Strict):
    status: str
    version: str
    registry: str = Field(description="Which ModelRegistry implementation is wired in")
    registry_reachable: bool | None = None
    # Null for an anonymous caller: this is fleet data, and `/health` is open on a
    # public URL. Null means "not told", which 0 would misreport as an empty fleet.
    device_count: int | None = None
    server_time: datetime


class ErrorResponse(Strict):
    """Uniform error envelope so the agent can branch on a stable code rather
    than parsing prose."""

    code: str
    message: str
    detail: dict[str, Any] | None = None
