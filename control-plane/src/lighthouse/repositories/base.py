"""The persistence seam (spec SS19: "Keep persistence isolated enough that
PostgreSQL could replace SQLite later").

One `Store` protocol rather than six per-table repositories. The tables are small
and tightly coupled -- a deployment change touches desired_deployment,
deployment_history and audit_event in one transaction -- so splitting them across
repositories would mostly produce an awkward unit-of-work dance for no benefit.
What actually has to be swappable is the backend, and that is what this protocol
isolates.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from lighthouse_contracts import DesiredState, EventType

from .rows import (
    ActualDeploymentRow,
    ArtifactCacheRow,
    AuditEventRow,
    DeploymentHistoryRow,
    DesiredDeploymentRow,
    DeviceRow,
    DeviceTokenRow,
)


class StoreError(RuntimeError):
    """Persistence failure."""


class DeviceExists(StoreError):
    """A device with that id is already registered."""


class DeviceUnknown(StoreError):
    """No such device."""


@runtime_checkable
class Store(Protocol):
    """Everything the services layer needs from storage."""

    # -- devices ---------------------------------------------------------

    def create_device(
        self, device_id: str, display_name: str | None, platform: str | None
    ) -> DeviceRow:
        """Register a device and seed its desired/actual rows.

        Raises DeviceExists on a duplicate id.
        """
        ...

    def get_device(self, device_id: str) -> DeviceRow | None: ...

    def list_devices(self) -> list[DeviceRow]: ...

    def touch_device(self, device_id: str) -> None:
        """Record that the device was just heard from (updates last_seen)."""
        ...

    def delete_device(self, device_id: str) -> None: ...

    # -- desired state ---------------------------------------------------

    def get_desired(self, device_id: str) -> DesiredDeploymentRow | None: ...

    def set_desired(self, row: DesiredDeploymentRow) -> DesiredDeploymentRow:
        """Write desired state with the next generation, atomically.

        The caller must not compute the generation itself: two concurrent
        operator actions would both read N and both write N+1, losing one. The
        implementation allocates it inside the write transaction.

        Also appends to deployment_history, so rollback classification has
        something to compare against.
        """
        ...

    def update_desired_digest(self, device_id: str, sha256: str) -> None:
        """Record the artifact digest without allocating a new generation.

        Deliberately separate from `set_desired`. A deployment request may be
        accepted before the bytes are cached, so the digest arrives later; that is
        new *knowledge* about an unchanged instruction, and bumping the generation
        for it would make every device re-reconcile for nothing.
        """
        ...

    def list_history(self, device_id: str, limit: int = 50) -> list[DeploymentHistoryRow]:
        """Past desired-state changes, newest first."""
        ...

    # -- actual state ----------------------------------------------------

    def get_actual(self, device_id: str) -> ActualDeploymentRow | None: ...

    def set_actual(self, row: ActualDeploymentRow) -> None: ...

    # -- audit -----------------------------------------------------------

    def append_event(self, row: AuditEventRow) -> None: ...

    def list_events(
        self,
        device_id: str | None = None,
        limit: int = 100,
        event_types: list[EventType] | None = None,
    ) -> list[AuditEventRow]: ...

    # -- device tokens ---------------------------------------------------

    def create_token(self, row: DeviceTokenRow) -> None: ...

    def get_token(self, token_id: str) -> DeviceTokenRow | None:
        """Look a token up by its public id -- an indexed point lookup.

        Returns the row regardless of revocation; the caller decides, so that a
        revoked token can be distinguished from an unknown one in audit.
        """
        ...

    def list_tokens(self, device_id: str) -> list[DeviceTokenRow]: ...

    def mark_token_used(self, token_id: str) -> None:
        """Record use. Throttled by the caller to avoid a write per heartbeat."""
        ...

    def revoke_token(self, token_id: str) -> bool: ...

    # -- artifact cache --------------------------------------------------

    def get_artifact(self, cache_key: str) -> ArtifactCacheRow | None: ...

    def claim_artifact(self, row: ArtifactCacheRow) -> bool:
        """Insert a PENDING cache row, returning False if one already exists.

        This is the concurrency gate on materialization: whichever request wins
        the insert does the download, and the others find an existing row and
        wait. Without it, two deployment requests for the same new version would
        both stream the artifact.
        """
        ...

    def update_artifact(self, row: ArtifactCacheRow) -> None: ...

    def list_artifacts(self) -> list[ArtifactCacheRow]: ...

    def touch_artifact(self, cache_key: str) -> None:
        """Update last_access, for LRU eviction ordering."""
        ...

    def delete_artifact(self, cache_key: str) -> None: ...

    def referenced_cache_keys(self) -> set[str]:
        """Cache keys a live desired deployment still points at.

        Eviction must never remove one of these: a device that comes back online
        after a long absence would otherwise be told to fetch an artifact the
        control plane just deleted.
        """
        ...

    # -- lifecycle -------------------------------------------------------

    def device_count(self) -> int: ...

    def close(self) -> None: ...
