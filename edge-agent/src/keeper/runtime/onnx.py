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

`start()` runs one inference on a synthesized zero input before it reports the
model as serving, so `RUNNING` means "the graph executed once" rather than "a
session object was constructed" -- see its docstring for why that distinction is
the point of the milestone and why the check lives here and not in the reconciler.
Graphs it cannot invent an input for start anyway and say so through
`hardware_info()["smoke_check"]`, which the fleet view reads to tell a proven
device from an unproven one.
"""

from __future__ import annotations

import logging
import math
import os
import platform
from pathlib import Path
from typing import Any

from lighthouse_contracts import GPU_PROVIDERS

from .base import InferenceRuntimeError, ModelLoadError, ModelStartError

log = logging.getLogger(__name__)

# Most to least preferred.
_PREFERRED_PROVIDERS = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)

# What `gpu_available` is allowed to mean lives in `lighthouse_contracts` as
# `GPU_PROVIDERS`, because the control plane now derives `DeviceView.acceleration`
# from the same set and two copies of this judgement would drift in exactly the
# direction that makes a CPU-only Jetson look healthy. The reasoning for the
# membership is in that module; the measurement behind it belongs here, where it
# was taken:
#
# Measured on the development MacBook 2026-10-04, where onnxruntime reports
# `['CoreMLExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']`.
# An earlier version of this was `!= "CPUExecutionProvider"`, which called that a
# GPU twice over -- `AzureExecutionProvider` is a remote inference endpoint and
# not local acceleration at all, and CoreML, which genuinely does use the GPU, is
# still not the accelerator a Jetson fleet is being watched for.

# Declared input types the smoke check will invent a zero for.
#
# Measured 2026-10-04 against onnxruntime 1.30.0: a nested *Python list* is
# accepted for every element type there is, converted using the graph's own
# declared type. So the smoke check needs no numpy -- which matters more than it
# sounds, because it means this code path is identical on a Jetson and on a
# laptop with no ML stack, and the tier-1 tests exercise the real thing rather
# than a stand-in.
#
# Restricted to float tensors on purpose, and the restriction is about honesty
# rather than capability. Zeros through a float graph is a safe probe. Zeros
# through a graph whose input is an *index* -- a token id, a category, a sequence
# length -- is not: the model may legitimately reject it, and a healthy model
# reported FAILED is the same kind of lie as a broken one reported RUNNING, only
# in the other direction. Those graphs get an honest "unproven" instead.
_SMOKE_INPUT_TYPES = frozenset({"tensor(float)", "tensor(float16)", "tensor(double)"})

# A ceiling on what the smoke check will allocate. A Python list of floats costs
# ~32 bytes an element, so a 1x3x1024x1024 input would be ~100 MB of transient
# list on an 8 GB device that is also holding CUDA -- a smoke check that OOMs the
# agent is worse than no smoke check. Models above the cap are left unproven and
# say so. The fixture graph is 784 elements.
_SMOKE_MAX_ELEMENTS = 1 << 20


def _smoke_dims(shape: Any) -> tuple[int, ...] | None:
    """A concrete shape to allocate, or None if the graph did not declare one.

    onnxruntime reports a `NodeArg.shape` mixing ints with *symbolic* dims --
    `['N', 1, 28, 28]` for the fixture graph, and names like `unk__6` or
    `batch_size` for models exported with dynamic axes. Symbolic dims become 1,
    which is the whole point: a batch of one is the cheapest thing that proves
    the graph runs.

    A non-positive int is also treated as 1. Some exporters write `-1` for a
    dynamic axis instead of a name, and `0` would allocate an empty tensor that
    proves nothing.
    """
    if not shape:
        # `None` (unknown rank) or `[]` (a rank-0 input). Neither is worth
        # guessing at, and a rank-0 probe would raise questions about scalars
        # that no fleet model actually asks.
        return None
    dims: list[int] = []
    for dim in shape:
        if isinstance(dim, bool):  # bool is an int subclass; never a dimension
            return None
        if isinstance(dim, int):
            dims.append(dim if dim > 0 else 1)
        elif dim is None or isinstance(dim, str):
            dims.append(1)
        else:
            return None
    return tuple(dims)


def _zeros(dims: tuple[int, ...]) -> Any:
    """Nested lists of 0.0 in the given shape.

    Fresh lists rather than `[inner] * n`, which would alias one row across the
    whole tensor. onnxruntime only reads the feed, so aliasing would work today
    and become a genuinely baffling bug the first time anything wrote to it.
    """
    if not dims:
        return 0.0
    return [_zeros(dims[1:]) for _ in range(dims[0])]


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

    def __init__(self, ort: Any = None, *, smoke_check: bool = True) -> None:
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
        self._smoke_check = smoke_check
        # Reported in every heartbeat, so a device that is RUNNING-but-unproven
        # is visible as such rather than indistinguishable from a proven one.
        self._smoke_status = "not run"

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
        # A new version inherits nothing from the old one's proof. Carrying a
        # "passed" across a hot-swap would report the *previous* model's
        # successful inference as evidence for this one.
        self._smoke_status = "not run"
        log.info(
            "loaded %s/%s with providers %s (input=%s)",
            name,
            version,
            session.get_providers(),
            self._input_name,
        )

    def start(self) -> None:
        """Begin serving -- after proving the graph can actually execute.

        Until this ran one inference, `RUNNING` meant "a session object was
        constructed". A model can load cleanly and still be unable to execute:
        a provider that accepts the graph and then fails on a kernel, a CUDA
        library that resolves at load and dies at the first launch, an opset the
        build parses but has no implementation for. All of those used to report
        RUNNING to the dashboard, which is a governance lie in a governance tool.

        This belongs here and not in the reconciler. `base.py` already documents
        `predict` as "only used by the smoke check", `start` is already allowed
        to raise `ModelStartError`, and `reconciler.py:157` already maps
        `InferenceRuntimeError` to FAILED with a backoff. So RUNNING comes to
        mean "the graph executed once" with **zero** reconciler changes and no
        effect on `MockRuntime`. Putting it in the reconciler would have taught
        the reconciler about tensor shapes, which is the boundary `base.py`
        calls load-bearing.

        Not every graph can be probed -- see `_smoke_plan`. A graph that cannot
        be starts anyway and reports why, because refusing to serve a model over
        an unsynthesizable input shape would be a worse failure than the one
        this is guarding against.
        """
        if self._session is None:
            raise ModelStartError("start() called before load()")
        label = "/".join(self._model) if self._model else "model"

        # The flag goes up before the probe so the probe can go through the real
        # `predict`, feed-keying and all, rather than a private path that could
        # drift from it. Safe because the agent is single-threaded: the reconcile
        # loop and the heartbeat share a thread, so nothing can observe RUNNING
        # between here and the failure branch below, which puts it back down.
        self._running = True

        if not self._smoke_check:
            self._smoke_status = "disabled"
            log.info("serving %s (smoke check disabled)", label)
            return

        dims, reason = self._smoke_plan()
        if reason is not None:
            self._smoke_status = f"skipped: {reason}"
            log.warning("serving %s but could not prove it executes: %s", label, reason)
            return

        try:
            outputs = self.predict(_zeros(dims))
        except Exception as exc:
            self._running = False
            self._smoke_status = f"failed: {exc}"
            raise ModelStartError(
                f"{label} loaded but failed to execute a zero input of shape "
                f"{list(dims)}: {exc}"
            ) from exc

        if not outputs:
            self._running = False
            self._smoke_status = "failed: the graph returned no outputs"
            raise ModelStartError(
                f"{label} executed but returned no outputs, so nothing about it "
                "can be trusted"
            )

        self._smoke_status = "passed"
        log.info(
            "serving %s -- smoke inference on a zero %s input returned %d output(s)",
            label,
            list(dims),
            len(outputs),
        )

    def _smoke_plan(self) -> tuple[tuple[int, ...], None] | tuple[None, str]:
        """The shape to probe with, or why this graph cannot be probed.

        Every reason here is a *skip*, never a refusal to serve, and each one is
        reported through `hardware_info` so the fleet view can separate "proven"
        from "unproven" instead of showing both as RUNNING.
        """
        if self._input_name is None:
            return None, (
                f"the graph declares {len(self._input_names)} inputs "
                f"{list(self._input_names)}, and a zero input can only be matched "
                "to a single declared one"
            )
        arg = self._session.get_inputs()[0]
        declared = getattr(arg, "type", None)
        if declared not in _SMOKE_INPUT_TYPES:
            return None, f"no zero input can be synthesized for declared type {declared!r}"
        shape = getattr(arg, "shape", None)
        dims = _smoke_dims(shape)
        if dims is None:
            return None, f"the graph does not declare a usable input shape (got {shape!r})"
        count = math.prod(dims)
        if count > _SMOKE_MAX_ELEMENTS:
            return None, (
                f"a zero input of shape {list(dims)} would be {count} elements, "
                f"over the {_SMOKE_MAX_ELEMENTS}-element cap this check allocates"
            )
        return dims, None

    def stop(self) -> None:
        self._running = False

    def unload(self) -> None:
        self._running = False
        self._session = None
        self._model = None
        self._input_names = ()
        self._smoke_status = "not run"

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
            # "passed" | "not run" | "disabled" | "skipped: <why>" | "failed: <why>".
            # A RUNNING device whose smoke check was skipped has not been proven
            # to execute anything, and the fleet view needs to be able to say so
            # -- otherwise "unproven" and "proven" look identical from here.
            # `HardwareInfo` allows extra fields, so this needs no contract change.
            "smoke_check": self._smoke_status,
        }
        try:
            providers = self._ort.get_available_providers()
            info["providers"] = providers
            info["gpu_available"] = any(p in GPU_PROVIDERS for p in providers)
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
