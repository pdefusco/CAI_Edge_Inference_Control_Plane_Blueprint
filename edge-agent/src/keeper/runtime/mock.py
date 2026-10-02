"""A runtime that does everything except inference.

This is what makes M1 demonstrable on a laptop with no Jetson and no ONNX Runtime
wheel. It is not a stub that returns `True` to everything: it reads the model file,
rejects an empty or missing one, and tracks load/start/stop state faithfully, so a
reconciler bug that calls `start()` without `load()` fails here rather than in M4
on hardware.

The fault-injection flags exist for the spec SS22 case "runtime start failure ->
FAILED", which is otherwise unreachable without breaking a real runtime on purpose.
"""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from typing import Any

from .base import ModelLoadError, ModelStartError

log = logging.getLogger(__name__)


class MockRuntime:
    """In-process fake. Implements `ModelRuntime`."""

    def __init__(self, *, fail_on_load: bool = False, fail_on_start: bool = False) -> None:
        self._fail_on_load = fail_on_load
        self._fail_on_start = fail_on_start
        self._loaded: tuple[str, str, Path] | None = None
        self._running = False

    @property
    def name(self) -> str:
        return "mock"

    def load(self, model_path: str, *, name: str, version: str) -> None:
        if self._fail_on_load:
            raise ModelLoadError(f"mock runtime configured to fail loading {name}/{version}")
        path = Path(model_path)
        if not path.exists():
            raise ModelLoadError(f"model path does not exist: {path}")
        if path.is_file():
            # Actually touch the bytes. A reconciler that activates before the
            # download finished should be caught by the mock, not in production.
            size = path.stat().st_size
            if size == 0:
                raise ModelLoadError(f"model file is empty: {path}")
            with open(path, "rb") as fh:
                head = fh.read(8)
            if not head:
                raise ModelLoadError(f"model file unreadable: {path}")
        self._loaded = (name, version, path)
        self._running = False
        log.info("mock runtime loaded %s/%s from %s", name, version, path)

    def start(self) -> None:
        if self._loaded is None:
            raise ModelStartError("start() called before load()")
        if self._fail_on_start:
            raise ModelStartError("mock runtime configured to fail starting")
        self._running = True
        name, version, _ = self._loaded
        log.info("mock runtime serving %s/%s", name, version)

    def stop(self) -> None:
        # No guard: stop() on an already-stopped runtime is a legitimate no-op,
        # because the reconciler re-asserts desired state every pass.
        if self._running:
            log.info("mock runtime stopped")
        self._running = False

    def unload(self) -> None:
        self.stop()
        self._loaded = None

    @property
    def is_running(self) -> bool:
        return self._running

    def predict(self, inputs: Any) -> Any:
        if not self._running:
            raise ModelStartError("predict() called while not running")
        return {"mock": True, "echo": inputs}

    def hardware_info(self) -> dict[str, Any]:
        return {
            "platform": f"{platform.system()}-{platform.machine()}",
            "gpu_available": False,
            "runtime": self.name,
            "cpu_count": os.cpu_count(),
        }
