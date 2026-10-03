"""Wiring and FastAPI dependencies.

Everything the routes need hangs off one `AppContext` on `app.state`, built once
at startup. The alternative -- module-level singletons -- would make it impossible
to stand up two independently-configured apps in one test process, which is
exactly what the reconciler tests need.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Request

from ..config import ConfigError, Settings
from ..registry import FakeModelRegistry, ModelRegistry
from ..repositories import SqliteStore, Store
from ..services import (
    ArtifactService,
    AuditService,
    DeploymentService,
    DeviceService,
    ModelCatalog,
    SessionStore,
)

log = logging.getLogger(__name__)


@dataclass(slots=True)
class AppContext:
    """Everything a request handler can reach."""

    settings: Settings
    store: Store
    registry: ModelRegistry
    artifacts: ArtifactService
    audit: AuditService
    devices: DeviceService
    deployments: DeploymentService
    catalog: ModelCatalog
    sessions: SessionStore
    _closed: bool = field(default=False, repr=False)

    def close(self) -> None:
        """Shut down in dependency order, at most once.

        Order is load-bearing: `artifacts.close()` first, because a materialization
        thread is still writing to the store, and closing the store under it
        segfaults the process rather than raising.

        Idempotent because shutdown genuinely runs more than once. Two TestClients
        over one app each drive the ASGI lifespan, and a second `store.close()`
        would operate on an already-closed connection -- a mistake that reads as a
        flaky test rather than as the double-shutdown it is.
        """
        if self._closed:
            return
        self._closed = True
        self.artifacts.close()
        # Duck-typed rather than widened into the `ModelRegistry` protocol: the
        # fake has no transport to release, and making `close()` part of the
        # seam would force every future implementation to carry an empty method.
        # After `artifacts.close()`, because a materialization still in flight
        # is reading from the registry's connection pool.
        registry_close = getattr(self.registry, "close", None)
        if callable(registry_close):
            try:
                registry_close()
            except Exception:  # pragma: no cover - shutdown is best-effort
                log.warning("registry close failed during shutdown", exc_info=True)
        self.store.close()


def build_registry(settings: Settings) -> ModelRegistry:
    """Pick a registry implementation.

    The fake is not a stub: it produces a real gzipped MLflow-layout tarball with
    a deterministic digest, so the artifact, checksum and cache paths under test
    are the same code that runs against CAI.
    """
    if settings.registry_impl == "fake":
        return FakeModelRegistry()
    # Imported lazily to keep httpx off the local dev path. The guard turns a
    # missing extra into the one actionable sentence that fixes it: this runs
    # inside `create_app`, so without it a deployment that forgot the extra dies
    # on a bare ModuleNotFoundError traceback at startup instead.
    try:
        from ..registry.cai import CAIModelRegistry  # type: ignore[attr-defined]
    except ImportError as exc:
        raise ConfigError(
            "LIGHTHOUSE_REGISTRY=cai needs the 'cai' extra, which is not installed.\n"
            "Install it with: pip install -e 'control-plane[cai]'\n"
            f"(import failed with: {exc})"
        ) from exc

    return CAIModelRegistry.from_env(settings)


def build_context(
    settings: Settings,
    *,
    store: Store | None = None,
    registry: ModelRegistry | None = None,
) -> AppContext:
    """Assemble the service graph. `store`/`registry` overrides exist for tests."""
    if store is None:
        Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
        store = SqliteStore(settings.db_path)
    if registry is None:
        registry = build_registry(settings)

    audit = AuditService(store)
    artifacts = ArtifactService(store, registry, settings)
    deployments = DeploymentService(store, registry, artifacts, audit, settings)
    devices = DeviceService(store, audit, settings, artifacts=artifacts)
    catalog = ModelCatalog(registry)

    return AppContext(
        settings=settings,
        store=store,
        registry=registry,
        artifacts=artifacts,
        audit=audit,
        devices=devices,
        deployments=deployments,
        catalog=catalog,
        sessions=SessionStore(),
    )


def ctx(request: Request) -> AppContext:
    """The per-app context. Every route depends on this rather than on globals."""
    return request.app.state.ctx  # type: ignore[no-any-return]
