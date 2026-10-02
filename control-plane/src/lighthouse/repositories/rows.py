"""Storage-shaped records.

These mirror the SQL tables, not the API. The split matters in one specific way
the spec is inconsistent about: storage is **flat** (`artifact_sha256` as a
column) while the API is **nested** (`model.sha256`). Keeping separate types makes
that deliberate rather than an accident of whichever shape got written first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from lighthouse_contracts import ActualState, DesiredState, EventType


@dataclass(slots=True)
class DeviceRow:
    device_id: str
    display_name: str | None = None
    platform: str | None = None
    registered_at: datetime | None = None
    last_seen: datetime | None = None


@dataclass(slots=True)
class DesiredDeploymentRow:
    """Current desired state for one device.

    `generation` is the ordering authority for the whole system. It increments on
    every change and is never derived from a clock.
    """

    device_id: str
    generation: int = 0
    desired_state: DesiredState = DesiredState.STOPPED
    model_name: str | None = None
    model_version: str | None = None
    registry_artifact_uri: str | None = None
    artifact_sha256: str | None = None
    artifact_format: str | None = None
    model_id: str | None = None
    version_uuid: str | None = None
    updated_at: datetime | None = None

    @property
    def cache_key(self) -> str | None:
        """Lineage key for the artifact cache, when a model is assigned."""
        if self.model_id and self.version_uuid:
            return f"{self.model_id}/{self.version_uuid}"
        return None


@dataclass(slots=True)
class ActualDeploymentRow:
    """Last state a device reported. Written only from heartbeats."""

    device_id: str
    observed_generation: int = 0
    actual_state: ActualState = ActualState.UNKNOWN
    model_name: str | None = None
    model_version: str | None = None
    artifact_sha256: str | None = None
    inference_running: bool = False
    message: str | None = None
    hardware: dict[str, Any] = field(default_factory=dict)
    updated_at: datetime | None = None


@dataclass(slots=True)
class AuditEventRow:
    event_id: str
    timestamp: datetime
    event_type: EventType
    device_id: str | None = None
    generation: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DeviceTokenRow:
    """A device credential.

    Only `token_id` and `token_sha256` are persisted -- the secret itself is
    returned once at issue time and is unrecoverable afterwards.
    """

    token_id: str
    device_id: str
    token_sha256: str
    created_at: datetime
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None
    label: str | None = None

    @property
    def active(self) -> bool:
        return self.revoked_at is None


@dataclass(slots=True)
class DeploymentHistoryRow:
    """One past desired-state change, used to classify rollbacks."""

    device_id: str
    generation: int
    desired_state: DesiredState
    model_name: str | None
    model_version: str | None
    created_at: datetime
    id: int | None = None


@dataclass(slots=True)
class ArtifactCacheRow:
    """A materialized artifact.

    Keyed by `<model_id>/<version_uuid>` -- registry lineage, not the mutable
    (name, version) label.
    """

    cache_key: str
    model_name: str
    model_version: str
    model_id: str
    version_uuid: str
    status: str = "PENDING"
    sha256: str | None = None
    size_bytes: int | None = None
    packaging: str | None = None
    entrypoint: str | None = None
    cache_path: str | None = None
    source_uri: str | None = None
    error: str | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None
    last_access: datetime | None = None

    @property
    def is_ready(self) -> bool:
        return self.status == "READY" and self.sha256 is not None
