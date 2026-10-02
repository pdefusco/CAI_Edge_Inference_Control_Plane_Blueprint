"""Model registry adapters.

`base` defines the seam; `fake` is the local implementation used by `make dev`
and the whole test suite. `cai.py` (M2) adds the real Cloudera AI adapter without
anything above this package changing.
"""

from .base import (
    ArtifactStream,
    ArtifactUnavailable,
    ModelNotFound,
    ModelRegistry,
    RegistryAuthError,
    RegistryError,
    RegistryModelVersion,
    RegistryUnavailable,
    UnsupportedFlavor,
    VersionFailed,
    VersionNotReady,
)
from .fake import FakeModelRegistry, build_fake_artifact

__all__ = [
    "ArtifactStream",
    "ArtifactUnavailable",
    "FakeModelRegistry",
    "ModelNotFound",
    "ModelRegistry",
    "RegistryAuthError",
    "RegistryError",
    "RegistryModelVersion",
    "RegistryUnavailable",
    "UnsupportedFlavor",
    "VersionFailed",
    "VersionNotReady",
    "build_fake_artifact",
]
