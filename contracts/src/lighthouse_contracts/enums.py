"""Lifecycle vocabulary shared by the control plane and the edge agent.

Spec SS4 fixes these names. They are on the wire, so renaming one is a breaking
protocol change -- not a refactor.
"""

from enum import StrEnum


class DesiredState(StrEnum):
    """What the control plane wants the device to be doing (spec SS4)."""

    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    REVOKED = "REVOKED"


class ActualState(StrEnum):
    """What the device reports it is actually doing (spec SS4).

    The intermediate states matter: they are what makes the dashboard honest
    about transitions instead of snapping straight to the desired value.
    """

    UNKNOWN = "UNKNOWN"
    IDLE = "IDLE"
    DOWNLOADING = "DOWNLOADING"
    DEPLOYING = "DEPLOYING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    REVOKING = "REVOKING"
    REVOKED = "REVOKED"
    FAILED = "FAILED"


class GovernanceStatus(StrEnum):
    """Derived server-side from the desired/actual pair -- never reported by the
    agent (spec SS4: "The server should derive governance/compliance status from
    desired vs. actual state rather than asking the agent to decide governance
    policy")."""

    HEALTHY = "HEALTHY"
    OUT_OF_SYNC = "OUT_OF_SYNC"
    STOP_PENDING = "STOP_PENDING"
    REVOKE_PENDING = "REVOKE_PENDING"
    REVOKED = "REVOKED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class Connectivity(StrEnum):
    """Derived from heartbeat age at read time, never stored as a boolean.

    A stored flag would need a background sweeper to ever become false, and
    would be wrong for exactly as long as that sweeper was behind.
    """

    ONLINE = "ONLINE"
    STALE = "STALE"
    OFFLINE = "OFFLINE"
    NEVER_SEEN = "NEVER_SEEN"


class ArtifactFormat(StrEnum):
    """ONNX is the canonical edge format (spec SS10). TensorRT engines are
    derived on-device in M6 and are not registry artifacts."""

    ONNX = "onnx"


class Packaging(StrEnum):
    """How the registry artifact bytes are laid out.

    The live registry's `artifact_uri` is a *prefix*, and what sits under it
    varies: MLflow writes a `model.tar.gz`, but a raw single file also occurs.
    Carrying this explicitly keeps the variation out of the agent's unpack path.
    """

    MLFLOW_TAR_GZ = "mlflow_tar_gz"
    RAW_FILE = "raw_file"


class EventType(StrEnum):
    """Audit event types (spec SS16)."""

    DEVICE_REGISTERED = "DEVICE_REGISTERED"
    DEVICE_TOKEN_ISSUED = "DEVICE_TOKEN_ISSUED"
    DEVICE_TOKEN_REVOKED = "DEVICE_TOKEN_REVOKED"
    DEPLOYMENT_REQUESTED = "DEPLOYMENT_REQUESTED"
    MODEL_VERSION_CHANGED = "MODEL_VERSION_CHANGED"
    DEPLOYMENT_ROLLED_BACK = "DEPLOYMENT_ROLLED_BACK"
    STOP_REQUESTED = "STOP_REQUESTED"
    REVOKE_REQUESTED = "REVOKE_REQUESTED"
    ARTIFACT_MATERIALIZED = "ARTIFACT_MATERIALIZED"
    ARTIFACT_MATERIALIZE_FAILED = "ARTIFACT_MATERIALIZE_FAILED"
    ARTIFACT_DOWNLOADED = "ARTIFACT_DOWNLOADED"
    DEVICE_STATE_CHANGED = "DEVICE_STATE_CHANGED"
    CHECKSUM_FAILED = "CHECKSUM_FAILED"
    RECONCILE_FAILED = "RECONCILE_FAILED"
