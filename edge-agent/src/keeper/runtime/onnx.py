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

from .base import InferenceRuntimeError, ModelLoadError, ModelStartError

log = logging.getLogger(__name__)

# Most to least preferred.
_PREFERRED_PROVIDERS = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)

# What `gpu_available` is allowed to mean: a provider that offloads to a discrete
# or integrated GPU. An allow-list and not `!= "CPUExecutionProvider"`, which is
# what this used to be and which is wrong in both directions.
#
# Measured on the development MacBook 2026-10-04, where onnxruntime reports
# `['CoreMLExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']`:
# the old test called that a GPU twice over. `AzureExecutionProvider` is a remote
# inference endpoint and is not local acceleration at all, and CoreML -- which
# genuinely does use the GPU -- is still not the thing this field is asked about.
# The question the dashboard is really asking is "is this device using the
# accelerator it was bought for", and a MacBook answering yes makes the column
# useless for the Jetson fleet it exists to watch.
_GPU_PROVIDERS = frozenset(
    {
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "ROCMExecutionProvider",
        "MIGraphXExecutionProvider",
        "DmlExecutionProvider",
    }
)


class OnnxRuntime:
    """Implements `ModelRuntime` over `onnxruntime.InferenceSession`.

    `ort` exists so this class can be tested without the wheel installed, which
    until now it could not be: every line below the import was unreachable on a
    laptop, so the first place this code ever ran was a Jetson over SSH.

    A constructor parameter rather than patching `sys.modules["onnxruntime"]`,
    for two reasons. It is what this repo does everywhere else -- dependencies
    arrive through constructors, and `httpx.MockTransport` is the single
    sanctioned exception. And a `sys.modules` entry is global: a fake left behind
    by one test would be picked up by the tests that mean to exercise the *real*
    wheel, which is a failure that only appears on machines that have it and only
    in some test orders. A parameter cannot leak.
    """

    def __init__(self, ort: Any = None) -> None:
        if ort is None:
            try:
                import onnxruntime  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - environment-dependent
                raise ModelLoadError(
                    "onnxruntime is not installed; install keeper[onnx] or set "
                    "KEEPER_RUNTIME=mock"
                ) from exc
            ort = onnxruntime
        self._ort = ort
        self._session: Any = None
        self._running = False
        self._model: tuple[str, str] | None = None
        self._input_names: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return "onnxruntime"

    @property
    def _input_name(self) -> str | None:
        """The input to key a bare tensor by, or None if that is not well defined.

        None for a graph with no declared inputs *and* for one with several, both
        of which used to end in a feed onnxruntime cannot use -- see `predict`.
        """
        return self._input_names[0] if len(self._input_names) == 1 else None

    def load(self, model_path: str, *, name: str, version: str) -> None:
        path = Path(model_path)
        # These four checks exist because onnxruntime's own messages for them are
        # bad in a specific way: they describe the parse, not the cause. Measured
        # 2026-10-04 against onnxruntime 1.30.0, the errors an operator would have
        # had to work from were `INVALID_PROTOBUF : ... Protobuf parsing failed`
        # for a directory, `FAIL : ... system error number 13` for a file the agent
        # cannot read, and for an empty file a truncated path from onnxruntime's own
        # *build* machine. The one case left to onnxruntime is a file with the wrong
        # contents, where `INVALID_PROTOBUF` is exactly right.
        #
        # The permission case is the one that will really happen. The unit runs as
        # `User=keeper` and an artifact unpacked by a root-run install step is a
        # plausible mistake; "system error number 13" is not a clue anyone should
        # have to decode over SSH.
        if not path.exists():
            raise ModelLoadError(f"model file does not exist: {path}")
        if path.is_dir():
            raise ModelLoadError(
                f"expected an ONNX file but found a directory: {path} -- the "
                "MLmodel entrypoint may name the model directory rather than the "
                "file inside it"
            )
        if not path.is_file():
            raise ModelLoadError(f"not a regular file: {path}")
        if not os.access(path, os.R_OK):
            raise ModelLoadError(f"ONNX file is not readable by this user: {path}")
        if path.stat().st_size == 0:
            raise ModelLoadError(f"ONNX file is empty: {path}")

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
        self._input_names = tuple(arg.name for arg in session.get_inputs())
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
        self._input_names = ()

    @property
    def is_running(self) -> bool:
        return self._running

    def predict(self, inputs: Any) -> Any:
        if self._session is None or not self._running:
            raise ModelStartError("predict() called while not running")
        if isinstance(inputs, dict):
            return self._session.run(None, inputs)
        # Keying a bare tensor requires exactly one declared input. Neither other
        # case used to be refused: a graph with no inputs produced the literal feed
        # `{None: tensor}`, and a multi-input graph silently fed the first name and
        # left the rest missing. onnxruntime then raised something about an invalid
        # feed, several frames away from the caller who could have passed a dict.
        if self._input_name is None:
            raise InferenceRuntimeError(
                "this model declares "
                f"{len(self._input_names)} inputs {list(self._input_names)}, so a "
                "bare tensor cannot be matched to one -- pass a {name: tensor} dict"
            )
        return self._session.run(None, {self._input_name: inputs})

    def hardware_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "platform": f"{platform.system()}-{platform.machine()}",
            "runtime": self.name,
            "cpu_count": os.cpu_count(),
        }
        try:
            providers = self._ort.get_available_providers()
            info["providers"] = providers
            info["gpu_available"] = any(p in _GPU_PROVIDERS for p in providers)
            info["onnxruntime_version"] = self._ort.__version__
        except Exception:  # pragma: no cover - never break a heartbeat
            info["gpu_available"] = None
        # What the loaded session is *actually* using, which is the field the M4
        # acceptance gate is about. `providers` above only says what this build
        # could do: a device can report CUDA as available and still be running the
        # model on its CPU, because a provider that cannot handle the graph falls
        # back silently and per-node. Reporting only availability would let a
        # fleet-wide fallback to CPU look like a fleet on the GPU.
        if self._session is not None:
            try:
                info["active_providers"] = self._session.get_providers()
            except Exception:  # pragma: no cover - never break a heartbeat
                pass
        # Jetson identifies itself here; absent on anything else.
        model_file = Path("/proc/device-tree/model")
        try:
            if model_file.is_file():
                info["device_model"] = model_file.read_text().strip("\x00").strip()
        except OSError:  # pragma: no cover
            pass
        return info
