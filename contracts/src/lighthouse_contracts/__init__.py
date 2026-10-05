"""Shared wire contracts for Lighthouse.

Imported by both the `lighthouse` control plane and the `keeper` edge agent so
that neither side can rename a field unilaterally. Depends on pydantic only --
nothing here should ever pull in FastAPI or boto3, because this package gets
installed on the Jetson.
"""

from .enums import (
    GPU_PROVIDERS,
    Acceleration,
    ActualState,
    ArtifactFormat,
    Connectivity,
    DesiredState,
    EventType,
    GovernanceStatus,
    Packaging,
)
from .schemas import (
    ArtifactRef,
    AuditEventView,
    DeploymentRequest,
    DesiredStateResponse,
    DeviceRegistrationRequest,
    DeviceTokenIssued,
    DeviceView,
    ErrorResponse,
    HardwareInfo,
    HealthResponse,
    HeartbeatRequest,
    HeartbeatResponse,
    ModelRef,
    ModelVersionView,
    ModelView,
    RuntimeStatus,
    Sha256,
    Strict,
)

__all__ = [
    "GPU_PROVIDERS",
    "Acceleration",
    "ActualState",
    "ArtifactFormat",
    "ArtifactRef",
    "AuditEventView",
    "Connectivity",
    "DeploymentRequest",
    "DesiredState",
    "DesiredStateResponse",
    "DeviceRegistrationRequest",
    "DeviceTokenIssued",
    "DeviceView",
    "ErrorResponse",
    "EventType",
    "GovernanceStatus",
    "HardwareInfo",
    "HealthResponse",
    "HeartbeatRequest",
    "HeartbeatResponse",
    "ModelRef",
    "ModelVersionView",
    "ModelView",
    "Packaging",
    "RuntimeStatus",
    "Sha256",
    "Strict",
]
