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


# Execution providers that mean "this device is using the accelerator it was
# bought for". Deliberately narrower than "anything but the CPU provider":
# `CoreMLExecutionProvider` and `AzureExecutionProvider` are both non-CPU and
# neither is local GPU acceleration -- Azure is a *remote* inference endpoint,
# and the laptop this repo was written on reports it. A MacBook answering "yes"
# makes the field useless for the Jetson fleet it exists to watch.
#
# This is the single home for that judgement on purpose. The agent reports
# `hardware.gpu_available` from it and the control plane derives `Acceleration`
# from it, and two copies would drift in exactly the direction that makes a
# CPU-only Jetson look healthy.
GPU_PROVIDERS = frozenset(
    {
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "ROCMExecutionProvider",
        "MIGraphXExecutionProvider",
        "DmlExecutionProvider",
    }
)


class Acceleration(StrEnum):
    """Whether a device's *loaded session* is on its accelerator (spec Phase 6).

    Derived at read time from the last heartbeat's `hardware.active_providers`.
    The Phase 6 acceptance gate is `ACCELERATED` **and** `smoke_check: passed`;
    `docs/jetson-setup.md` §7 is the sequence.

    This needs its own field because `GovernanceStatus` cannot answer it and
    must not try. A device that downloaded the right bytes, loaded them and is
    serving them *is* in sync with its desired state -- whether it is using the
    GPU or not. So a Jetson bought for its GPU and serving on its CPU is a
    **failed acceptance check that is correctly HEALTHY governance**, and
    special-casing that inside `HEALTHY` would break the one thing governance
    means.

    Three values, and the third is not padding:

    * `ACCELERATED` -- at least one active provider is in `GPU_PROVIDERS`.
    * `CPU_ONLY` -- providers were reported and none of them qualify. Named for
      the acceptance criterion's own wording; note it also covers the CoreML and
      Azure cases above, which are not literally CPU but are equally not the
      accelerator the gate asks about.
    * `UNKNOWN` -- nothing was reported. A `mock` runtime, an agent with no
      session loaded, and an agent too old to send the key all land here, and
      calling any of them `CPU_ONLY` would be a confident claim about hardware
      nobody measured. The gate treats `UNKNOWN` as a failure all the same --
      not proven is not passed -- but it is a different sentence, and the
      difference is what tells an operator whether to look at the wheel or at
      the agent.
    """

    ACCELERATED = "ACCELERATED"
    CPU_ONLY = "CPU_ONLY"
    UNKNOWN = "UNKNOWN"


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
