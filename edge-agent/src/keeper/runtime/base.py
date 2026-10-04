"""The inference runtime boundary (spec SS18).

The reconciler must not know whether inference is ONNX Runtime in-process, a
TensorRT engine, or a mock that only pretends. That is not abstraction for its own
sake: the entire M1 test suite runs against `MockRuntime` on a laptop, and the
*same* reconciler code then drives `OnnxRuntime` on the Jetson in M4. If
reconciliation logic leaked runtime details, M1 would prove nothing about M4.

`load` and `start` are deliberately separate. Loading can fail on a corrupt or
incompatible model -- which the agent must report as FAILED without ever claiming
RUNNING -- while starting is what makes the loaded model serve.

`start` is not merely a flag flip, and the separation is what buys room for that.
`OnnxRuntime.start` runs one inference on a synthesized input before it reports
the model as serving, because a model can load cleanly and still be unable to
execute: a provider that accepts the graph and then has no kernel for a node, a
CUDA library that resolves at load and dies at the first launch. `MockRuntime`
keeps the cheap flip, and the reconciler cannot tell the two apart -- which is the
point of this boundary.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class RuntimeError_(RuntimeError):
    """Base for runtime failures. Named with a trailing underscore to avoid
    shadowing the builtin; exported as `InferenceRuntimeError`."""


InferenceRuntimeError = RuntimeError_


class ModelLoadError(InferenceRuntimeError):
    """The model could not be loaded: bad opset, unsupported op, corrupt file.

    Distinct from a start failure because the remedy differs -- a load error means
    the *artifact* is wrong for this device and retrying will not help, so the
    operator needs to see it rather than watch a retry loop.
    """


class ModelStartError(InferenceRuntimeError):
    """The model loaded but inference could not be started."""


@runtime_checkable
class ModelRuntime(Protocol):
    """What the reconciler is allowed to assume about inference.

    Every method must be safe to call repeatedly: the reconciler is idempotent and
    will happily call `start()` on an already-started runtime after a restart.
    """

    @property
    def name(self) -> str:
        """Implementation name, surfaced in heartbeats for debugging."""
        ...

    def load(self, model_path: str, *, name: str, version: str) -> None:
        """Prepare a model for inference. Raises `ModelLoadError` on bad input."""
        ...

    def start(self) -> None:
        """Begin serving the loaded model. Raises `ModelStartError`.

        May do real work first -- `OnnxRuntime` proves the graph executes. A
        `ModelStartError` is an `InferenceRuntimeError`, which the reconciler
        already maps to FAILED with a backoff.
        """
        ...

    def stop(self) -> None:
        """Stop serving. Must succeed even if nothing is running."""
        ...

    def unload(self) -> None:
        """Release the model and any resources. Must be idempotent."""
        ...

    @property
    def is_running(self) -> bool:
        """Whether inference is currently being served."""
        ...

    def predict(self, inputs: Any) -> Any:
        """Run one inference. Only used by the smoke check, not by reconciliation."""
        ...

    def hardware_info(self) -> dict[str, Any]:
        """Device self-description for the heartbeat. Best-effort; never raises."""
        ...
