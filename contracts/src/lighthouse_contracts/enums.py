"""Lifecycle vocabulary shared by the control plane and the edge agent.

Spec SS4 fixes these names. They are on the wire, so renaming one is a breaking
protocol change -- not a refactor.
"""

import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    # Python 3.10 fallback, and it exists for one concrete reason: the Jetson.
    # JetPack 6 ships Python 3.10, and NVIDIA's accelerated aarch64 onnxruntime
    # wheels are built against it, so requiring 3.11 on the device would mean
    # either building onnxruntime from source or giving up the GPU. `keeper`
    # imports this module, so a 3.11-only enum here would have decided that.
    #
    # `__str__ = str.__str__` is the whole trick and is NOT optional. Without it
    # a plain `(str, Enum)` stringifies as "DesiredState.RUNNING" instead of
    # "RUNNING", which would change every log line and f-string that
    # interpolates one of these -- a wire-visible difference, since these names
    # are on the wire (see the module docstring). With it, behaviour is
    # identical to `enum.StrEnum`: verified 2026-10-04 on real 3.10 against real
    # 3.11+ across str(), f-strings, format(), .value, == "x", json.dumps and
    # "%s". `tests/test_enums_strenum_parity.py` pins that.
    from enum import Enum

    class StrEnum(str, Enum):  # noqa: D101
        __str__ = str.__str__


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
    derived on-device in M6 and are not registry artifacts.

    UNKNOWN exists because the real registry cannot always tell us. A version
    imported from Hugging Face or NGC carries no flavor metadata, so the format
    is genuinely unknown until the bytes are inspected -- and those versions are
    never edge-runnable anyway. Saying UNKNOWN keeps them listable (and visibly
    undeployable) instead of mislabelling them ONNX.
    """

    ONNX = "onnx"
    UNKNOWN = "unknown"


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
