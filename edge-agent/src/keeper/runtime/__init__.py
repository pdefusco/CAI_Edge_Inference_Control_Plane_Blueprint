"""Inference runtime implementations.

`build_runtime` is the only place an implementation is chosen, and the ONNX import
is deliberately lazy: `onnxruntime` is a large wheel that on Jetson comes from
NVIDIA's index rather than PyPI, so importing it at package load would make the
agent uninstallable anywhere it is not needed.
"""

from __future__ import annotations

from .base import (
    InferenceRuntimeError,
    ModelLoadError,
    ModelRuntime,
    ModelStartError,
)
from .mock import MockRuntime

__all__ = [
    "InferenceRuntimeError",
    "MockRuntime",
    "ModelLoadError",
    "ModelRuntime",
    "ModelStartError",
    "build_runtime",
]


def build_runtime(impl: str) -> ModelRuntime:
    if impl == "mock":
        return MockRuntime()
    if impl == "onnx":
        from .onnx import OnnxRuntime  # noqa: PLC0415 - lazy on purpose

        return OnnxRuntime()
    raise ValueError(f"unknown runtime implementation: {impl!r}")
