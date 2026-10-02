"""Real inference via ONNX Runtime. Exercised for real in M4 on the Jetson.

Written now rather than in M4 so `build_runtime("onnx")` is not a lie, and so the
reconciler is proven against the same interface both implementations satisfy. It
imports `onnxruntime` only when instantiated.

Provider selection is the one Jetson-specific decision here. `onnxruntime-gpu` on
Jetson exposes `CUDAExecutionProvider` (and `TensorrtExecutionProvider` when built
with it); plain `onnxruntime` exposes only CPU. Requesting a provider that is not
available raises, so we intersect our preference list with what the installed build
actually reports -- otherwise the agent would fail to start on a CPU-only box for
no good reason.
"""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from typing import Any

from .base import ModelLoadError, ModelStartError

log = logging.getLogger(__name__)

# Most to least preferred.
_PREFERRED_PROVIDERS = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)


class OnnxRuntime:
    """Implements `ModelRuntime` over `onnxruntime.InferenceSession`."""

    def __init__(self) -> None:
        try:
            import onnxruntime  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ModelLoadError(
                "onnxruntime is not installed; install keeper[onnx] or set KEEPER_RUNTIME=mock"
            ) from exc
        self._ort = onnxruntime
        self._session: Any = None
        self._running = False
        self._model: tuple[str, str] | None = None
        self._input_name: str | None = None

    @property
    def name(self) -> str:
        return "onnxruntime"

    def load(self, model_path: str, *, name: str, version: str) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise ModelLoadError(f"not an ONNX file: {path}")

        available = set(self._ort.get_available_providers())
        providers = [p for p in _PREFERRED_PROVIDERS if p in available] or ["CPUExecutionProvider"]

        try:
            session = self._ort.InferenceSession(str(path), providers=providers)
        except Exception as exc:
            # onnxruntime raises a variety of its own exception types for a bad
            # opset or an unsupported op. Normalised here so the reconciler only
            # has to know about ModelLoadError.
            raise ModelLoadError(f"onnxruntime could not load {path}: {exc}") from exc

        # Swap in only after a successful load, so a failed upgrade leaves the
        # previously-working session serving rather than tearing it down first.
        if self._session is not None:
            self._running = False
        self._session = session
        self._model = (name, version)
        inputs = session.get_inputs()
        self._input_name = inputs[0].name if inputs else None
        self._running = False
        log.info(
            "loaded %s/%s with providers %s (input=%s)",
            name,
            version,
            session.get_providers(),
            self._input_name,
        )

    def start(self) -> None:
        if self._session is None:
            raise ModelStartError("start() called before load()")
        self._running = True
        log.info("serving %s", "/".join(self._model) if self._model else "model")

    def stop(self) -> None:
        self._running = False

    def unload(self) -> None:
        self._running = False
        self._session = None
        self._model = None
        self._input_name = None

    @property
    def is_running(self) -> bool:
        return self._running

    def predict(self, inputs: Any) -> Any:
        if self._session is None or not self._running:
            raise ModelStartError("predict() called while not running")
        feed = inputs if isinstance(inputs, dict) else {self._input_name: inputs}
        return self._session.run(None, feed)

    def hardware_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "platform": f"{platform.system()}-{platform.machine()}",
            "runtime": self.name,
            "cpu_count": os.cpu_count(),
        }
        try:
            providers = self._ort.get_available_providers()
            info["providers"] = providers
            info["gpu_available"] = any(p != "CPUExecutionProvider" for p in providers)
            info["onnxruntime_version"] = self._ort.__version__
        except Exception:  # pragma: no cover - never break a heartbeat
            info["gpu_available"] = None
        # Jetson identifies itself here; absent on anything else.
        model_file = Path("/proc/device-tree/model")
        try:
            if model_file.is_file():
                info["device_model"] = model_file.read_text().strip("\x00").strip()
        except OSError:  # pragma: no cover
            pass
        return info
