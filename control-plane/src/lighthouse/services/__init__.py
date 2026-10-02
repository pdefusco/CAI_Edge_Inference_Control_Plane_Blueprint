"""Business logic. The API layer is a thin shell over this package.

The split that matters: everything in here is callable from a test without an
HTTP client, and nothing in here imports FastAPI. The twelve reconciliation cases
in spec SS22 are exercised against these services plus the agent's reconciler,
not against routes.
"""

from .artifact_service import ArtifactFailed, ArtifactInfo, ArtifactNotReady, ArtifactService
from .audit import AuditService
from .catalog import ModelCatalog
from .deployment_service import DeploymentService, artifact_path
from .device_service import DevicePrincipal, DeviceService
from .errors import (
    DeviceAlreadyExists,
    DeviceNotFound,
    NothingDeployed,
    ServiceError,
)
from .governance import build_device_view, derive_connectivity, derive_governance
from .sessions import DEFAULT_TTL_SECONDS, SESSION_PREFIX, SessionStore

__all__ = [
    "DEFAULT_TTL_SECONDS",
    "SESSION_PREFIX",
    "ArtifactFailed",
    "ArtifactInfo",
    "ArtifactNotReady",
    "ArtifactService",
    "AuditService",
    "DeploymentService",
    "DeviceAlreadyExists",
    "DeviceNotFound",
    "DevicePrincipal",
    "DeviceService",
    "ModelCatalog",
    "NothingDeployed",
    "ServiceError",
    "SessionStore",
    "artifact_path",
    "build_device_view",
    "derive_connectivity",
    "derive_governance",
]
