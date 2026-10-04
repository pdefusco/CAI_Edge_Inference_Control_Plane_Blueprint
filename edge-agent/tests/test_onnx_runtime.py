"""Tier 1: the ONNX runtime's own logic, with no onnxruntime wheel installed.

Until the `ort` constructor parameter landed, this file could not exist. Every line
of `runtime/onnx.py` below its import was unreachable on a laptop, so the first
machine that ever executed it was a Jetson over SSH -- which is a terrible place to
discover a typo. These tests run in the default `make test` on a bare laptop.

What a fake can and cannot prove is worth being explicit about, because the
temptation is to believe this file covers the runtime:

* It **does** prove the decisions this module makes on its own -- which providers it
  asks for and in what order, what it does with a failed load, how it builds the
  feed dict, when `is_running` flips.
* It **cannot** prove that onnxruntime accepts the arguments we pass it, or that a
  real graph loads. `FakeOrt.InferenceSession` accepts anything. That is the job of
  the tier-2 tests behind `pytest.importorskip("onnxruntime")`, and of the Jetson.

So the fake is deliberately *unforgiving* about the shape of the call -- it asserts
`providers` arrives as a keyword, for instance -- because a fake that shrugs at a
call the real library would reject is worse than no test at all: it turns a green
suite into evidence of nothing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from keeper.runtime.base import (
    InferenceRuntimeError,
    ModelLoadError,
    ModelRuntime,
    ModelStartError,
)
from keeper.runtime.onnx import _PREFERRED_PROVIDERS, _SMOKE_MAX_ELEMENTS, OnnxRuntime


# `outputs` has to be settable to `None` itself -- a graph that returns nothing is
# one of the smoke check's failure cases -- so "unset" cannot be spelled `None`.
_UNSET: Any = object()


@dataclass
class FakeNodeArg:
    """Stands in for onnxruntime's `NodeArg`.

    `shape` and `type` are not decoration: the smoke check in `start()` reads
    them to decide what tensor to synthesize. The defaults are what the real
    fixture graph reports, measured 2026-10-04 -- `['N', 1, 28, 28]` with a
    symbolic batch dim as a `str`, and `'tensor(float)'`.

    Getting these wrong in the *other* direction is the subtle failure: when this
    class had only `.name`, `getattr(arg, "type", None)` returned None, every
    tier-1 `start()` quietly took the "cannot synthesize an input" skip, and the
    whole suite passed while proving nothing about the smoke check at all.
    """

    name: str
    shape: list[Any] | None = field(default_factory=lambda: ["N", 1, 28, 28])
    type: str | None = "tensor(float)"


class FakeSession:
    """A loaded model that records what it was asked to do.

    `providers` is stored as handed over rather than normalised, so a test can
    assert on the exact list -- the ordering of that list is the one Jetson-relevant
    decision `load` makes, and onnxruntime honours it as a preference order.
    """

    def __init__(
        self,
        path: str,
        providers: list[str],
        inputs: list[FakeNodeArg],
        *,
        raise_on_run: BaseException | None = None,
        outputs: Any = _UNSET,
    ) -> None:
        self.path = path
        self.providers = providers
        self.inputs = inputs
        self.runs: list[tuple[Any, Any]] = []
        self.raise_on_run = raise_on_run
        self.outputs = ["fake-output"] if outputs is _UNSET else outputs

    def get_inputs(self) -> list[FakeNodeArg]:
        return self.inputs

    def get_providers(self) -> list[str]:
        return self.providers

    def run(self, output_names: Any, feed: Any) -> Any:
        self.runs.append((output_names, feed))
        if self.raise_on_run is not None:
            raise self.raise_on_run
        return self.outputs


@dataclass
class FakeOrt:
    """The `onnxruntime` module, as much of it as `OnnxRuntime` touches.

    Faults are attributes rather than subclasses so one test can flip a failure on
    mid-scenario -- which is how the failed-upgrade case is arranged without two
    fixtures. Same pattern as `StubClient` in `conftest.py`.
    """

    available: list[str] = field(
        default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    declared_inputs: list[str] = field(default_factory=lambda: ["input"])
    declared_shape: Any = field(default_factory=lambda: ["N", 1, 28, 28])
    declared_type: str | None = "tensor(float)"
    raise_on_load: BaseException | None = None
    # Faults in the *session*, which is where the smoke check meets the graph: a
    # model that loads and then cannot execute is the exact failure commit 10
    # exists to catch, and it has no load-time symptom at all.
    raise_on_run: BaseException | None = None
    outputs: Any = _UNSET
    __version__: str = "1.30.0"

    sessions: list[FakeSession] = field(default_factory=list)

    def get_available_providers(self) -> list[str]:
        return list(self.available)

    def InferenceSession(  # noqa: N802 - mirrors onnxruntime's own class name
        self, path: str, *args: Any, providers: Any = None, **kwargs: Any
    ) -> FakeSession:
        # The real constructor is positional-or-keyword, but passing `providers`
        # positionally would land on `sess_options`. Asserting the call shape here
        # means a refactor that "simplifies" the call site fails on a laptop instead
        # of on the device.
        assert not args, "providers must be passed as a keyword, not positionally"
        assert isinstance(providers, list), f"providers must be a list, got {providers!r}"
        if self.raise_on_load is not None:
            raise self.raise_on_load
        # Copied so a test mutating the fake between two loads does not reach back
        # into the first session's node args -- but `None` has to survive as
        # `None`, because unknown rank is one of the shapes the smoke check skips.
        shape = list(self.declared_shape) if isinstance(self.declared_shape, list) else None
        session = FakeSession(
            path,
            providers,
            [FakeNodeArg(n, shape, self.declared_type) for n in self.declared_inputs],
            raise_on_run=self.raise_on_run,
            outputs=self.outputs,
        )
        self.sessions.append(session)
        return session


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    """A real file on disk. `load` checks `is_file()` before touching `ort`, so the
    path has to exist even though `FakeSession` never opens it."""
    path = tmp_path / "model.onnx"
    path.write_bytes(b"not really a protobuf")
    return path


def _shape_of(tensor: Any) -> tuple[int, ...]:
    """The shape of a nested list, by walking the first element down.

    Deliberately not numpy: these tests run in the default `make test`, where
    there is no ML stack, and the thing being inspected is a plain Python list
    precisely because the smoke check needs no numpy either.
    """
    dims: list[int] = []
    while isinstance(tensor, list):
        dims.append(len(tensor))
        tensor = tensor[0] if tensor else None
    return tuple(dims)


def _flatten(tensor: Any) -> list[Any]:
    if not isinstance(tensor, list):
        return [tensor]
    return [leaf for row in tensor for leaf in _flatten(row)]


def test_it_satisfies_the_runtime_protocol() -> None:
    """The reconciler is typed against `ModelRuntime`, so this is the contract."""
    assert isinstance(OnnxRuntime(ort=FakeOrt()), ModelRuntime)


def test_the_fake_never_reaches_sys_modules(model_file: Path) -> None:
    """The whole argument for a constructor parameter over patching `sys.modules`.

    If this ever fails, the tier-2 tests on a machine that *has* the wheel are being
    handed a fake, and their green is meaningless.
    """
    import sys

    runtime = OnnxRuntime(ort=FakeOrt())
    runtime.load(str(model_file), name="fashion-cnn", version="1")
    assert "onnxruntime" not in sys.modules or sys.modules["onnxruntime"] is not runtime._ort


class TestProviderSelection:
    """The one Jetson-specific decision in this module.

    Requesting a provider onnxruntime does not have raises, so the preference list
    is intersected with what the installed build reports. Getting this wrong means
    either refusing to start on a CPU-only box, or silently serving on the CPU of a
    device bought for its GPU.
    """

    def test_it_asks_for_the_available_providers_in_preference_order(
        self, model_file: Path
    ) -> None:
        """Preference order is ours, not onnxruntime's.

        The available list here is deliberately *reversed* relative to
        `_PREFERRED_PROVIDERS`: an implementation that filtered the available list
        instead of the preference list would pass a plain membership assertion and
        hand ORT `[CPU, CUDA, Tensorrt]`, which prefers the CPU on a Jetson.
        """
        ort = FakeOrt(
            available=[
                "CPUExecutionProvider",
                "CUDAExecutionProvider",
                "TensorrtExecutionProvider",
            ]
        )
        OnnxRuntime(ort=ort).load(str(model_file), name="m", version="1")
        assert ort.sessions[0].providers == list(_PREFERRED_PROVIDERS)

    def test_it_drops_providers_this_build_does_not_have(self, model_file: Path) -> None:
        """Plain `onnxruntime` exposes only CPU; `onnxruntime-gpu` on Jetson adds
        CUDA. Neither should be asked for what it cannot do."""
        ort = FakeOrt(available=["CUDAExecutionProvider", "CPUExecutionProvider"])
        OnnxRuntime(ort=ort).load(str(model_file), name="m", version="1")
        assert ort.sessions[0].providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_it_ignores_providers_we_have_no_opinion_about(self, model_file: Path) -> None:
        """A build can report providers outside our list -- CoreML on this laptop,
        ROCm, DirectML. Those are not asked for; the intersection is the point."""
        ort = FakeOrt(available=["CoreMLExecutionProvider", "CPUExecutionProvider"])
        OnnxRuntime(ort=ort).load(str(model_file), name="m", version="1")
        assert ort.sessions[0].providers == ["CPUExecutionProvider"]

    def test_an_empty_intersection_still_falls_back_to_cpu(self, model_file: Path) -> None:
        """`or ["CPUExecutionProvider"]` is not dead code: a build reporting
        nothing we recognise must still load rather than raise on an empty list."""
        ort = FakeOrt(available=["SomethingExoticExecutionProvider"])
        OnnxRuntime(ort=ort).load(str(model_file), name="m", version="1")
        assert ort.sessions[0].providers == ["CPUExecutionProvider"]


class TestLoadFailures:
    """The pre-flight checks exist because onnxruntime's own messages for these
    describe the parse rather than the cause. Measured against onnxruntime 1.30.0
    on 2026-10-04, an operator debugging over SSH would have got
    `INVALID_PROTOBUF ... Protobuf parsing failed` for a directory, `FAIL ... system
    error number 13` for a file the agent cannot read, and for an empty file a
    truncated path from onnxruntime's own build machine.

    Each test asserts onnxruntime was never reached, which is the other half: these
    are not better messages layered on top of a successful call, they are refusals.
    """

    def test_a_missing_path_is_a_load_error_and_never_reaches_onnxruntime(
        self, tmp_path: Path
    ) -> None:
        ort = FakeOrt()
        runtime = OnnxRuntime(ort=ort)
        with pytest.raises(ModelLoadError, match="does not exist"):
            runtime.load(str(tmp_path / "absent.onnx"), name="m", version="1")
        assert ort.sessions == []

    def test_a_directory_says_so_and_names_the_likely_cause(self, tmp_path: Path) -> None:
        """The realistic way this happens: an MLmodel `data` field naming the model
        *directory*, which is a legal MLflow layout for some flavours. The old
        message was `not an ONNX file:` for this and for every other case."""
        directory = tmp_path / "model"
        directory.mkdir()
        ort = FakeOrt()
        with pytest.raises(ModelLoadError, match="directory"):
            OnnxRuntime(ort=ort).load(str(directory), name="m", version="1")
        assert ort.sessions == []

    def test_an_unreadable_file_names_the_permission_problem(self, tmp_path: Path) -> None:
        """The failure this milestone is most likely to actually hit: the unit runs
        as `User=keeper` and an artifact unpacked by a root-run install step is an
        easy mistake. onnxruntime calls that `system error number 13`."""
        path = tmp_path / "model.onnx"
        path.write_bytes(b"\x08\x09")
        path.chmod(0o000)
        if os.access(path, os.R_OK):
            pytest.skip("running as a user that ignores file modes (root?)")
        ort = FakeOrt()
        try:
            with pytest.raises(ModelLoadError, match="not readable"):
                OnnxRuntime(ort=ort).load(str(path), name="m", version="1")
        finally:
            path.chmod(0o644)
        assert ort.sessions == []

    def test_an_empty_file_says_it_is_empty(self, tmp_path: Path) -> None:
        """A zero-byte `.onnx` is what a truncated write or a failed packaging step
        leaves behind. It would have to pass the artifact SHA-256 to get here, so
        this is a packaging bug rather than a transport one -- which is exactly why
        the message should not be about protobuf."""
        path = tmp_path / "model.onnx"
        path.write_bytes(b"")
        ort = FakeOrt()
        with pytest.raises(ModelLoadError, match="empty"):
            OnnxRuntime(ort=ort).load(str(path), name="m", version="1")
        assert ort.sessions == []

    def test_a_file_with_the_wrong_contents_is_left_to_onnxruntime(
        self, model_file: Path
    ) -> None:
        """Deliberately not pre-checked. `INVALID_PROTOBUF` is already the right
        answer for a file that exists, is readable and is not an ONNX graph, and a
        magic-byte check here would be a second, worse parser."""
        ort = FakeOrt(raise_on_load=RuntimeError("INVALID_PROTOBUF"))
        with pytest.raises(ModelLoadError, match="INVALID_PROTOBUF"):
            OnnxRuntime(ort=ort).load(str(model_file), name="m", version="1")

    @pytest.mark.parametrize(
        "exc",
        [
            RuntimeError("[ONNXRuntimeError] : 7 : INVALID_PROTOBUF"),
            ValueError("opset 21 is not supported"),
            OSError("libcublas.so.12: cannot open shared object file"),
        ],
        ids=["invalid-protobuf", "bad-opset", "missing-cuda-lib"],
    )
    def test_whatever_onnxruntime_raises_becomes_a_model_load_error(
        self, model_file: Path, exc: BaseException
    ) -> None:
        """onnxruntime raises several of its own types plus plain builtins. The
        reconciler only knows `ModelLoadError`, so everything is normalised -- and
        the original text is preserved, because that text is the only thing an
        operator can act on."""
        runtime = OnnxRuntime(ort=FakeOrt(raise_on_load=exc))
        with pytest.raises(ModelLoadError) as caught:
            runtime.load(str(model_file), name="m", version="1")
        assert str(exc) in str(caught.value)
        assert caught.value.__cause__ is exc

    def test_a_failed_upgrade_leaves_the_previous_model_serving(self, model_file: Path) -> None:
        """The reason `load` builds the new session *before* dropping the old one.

        A device that tore down a working model and then failed to load its
        replacement would stop serving on a bad artifact -- the worst possible
        outcome of a governance push, and invisible from the control plane until the
        next heartbeat.
        """
        ort = FakeOrt()
        # Smoke check off: this test counts `run` calls to prove the old session is
        # still live, and a smoke inference at `start()` would be one of them.
        runtime = OnnxRuntime(ort=ort, smoke_check=False)
        runtime.load(str(model_file), name="fashion-cnn", version="1")
        runtime.start()
        good = ort.sessions[0]

        ort.raise_on_load = RuntimeError("v2 is corrupt")
        with pytest.raises(ModelLoadError):
            runtime.load(str(model_file), name="fashion-cnn", version="2")

        assert runtime.is_running is True
        assert runtime._session is good
        runtime.predict([1.0])
        assert len(good.runs) == 1


class TestLifecycle:
    def test_a_successful_load_requires_a_fresh_start(self, model_file: Path) -> None:
        """Loading a new version must not inherit the old one's RUNNING flag.

        Otherwise `is_running` would claim the *new* model is serving before
        anything started it, which is precisely the governance lie this milestone
        exists to remove.
        """
        ort = FakeOrt()
        runtime = OnnxRuntime(ort=ort)
        runtime.load(str(model_file), name="m", version="1")
        runtime.start()
        assert runtime.is_running is True

        runtime.load(str(model_file), name="m", version="2")
        assert runtime.is_running is False
        assert runtime._session is ort.sessions[1]

    def test_start_before_load_is_refused(self) -> None:
        with pytest.raises(ModelStartError):
            OnnxRuntime(ort=FakeOrt()).start()

    def test_predict_before_start_is_refused(self, model_file: Path) -> None:
        runtime = OnnxRuntime(ort=FakeOrt())
        runtime.load(str(model_file), name="m", version="1")
        with pytest.raises(ModelStartError):
            runtime.predict([1.0])

    def test_predict_after_stop_is_refused(self, model_file: Path) -> None:
        """`stop()` is what a remote Stop becomes, so this is the assertion that a
        stopped device really is not serving."""
        runtime = OnnxRuntime(ort=FakeOrt())
        runtime.load(str(model_file), name="m", version="1")
        runtime.start()
        runtime.stop()
        with pytest.raises(ModelStartError):
            runtime.predict([1.0])

    def test_stop_and_unload_are_idempotent(self, model_file: Path) -> None:
        """`base.py:45` requires this: the reconciler is idempotent and will call
        these again after a restart."""
        runtime = OnnxRuntime(ort=FakeOrt())
        runtime.load(str(model_file), name="m", version="1")
        runtime.start()
        for _ in range(2):
            runtime.stop()
            runtime.unload()
        assert runtime.is_running is False
        assert runtime._session is None
        assert runtime._model is None
        assert runtime._input_name is None

    def test_unload_then_start_is_refused(self, model_file: Path) -> None:
        runtime = OnnxRuntime(ort=FakeOrt())
        runtime.load(str(model_file), name="m", version="1")
        runtime.unload()
        with pytest.raises(ModelStartError):
            runtime.start()


class TestFeedConstruction:
    """How `predict` turns an argument into onnxruntime's `{name: tensor}` feed.

    The input name is read once at load time rather than per call. That is the right
    trade -- `get_inputs()` is not free and the graph cannot change under a session
    -- but it means a stale `_input_name` would send every inference to the wrong
    input, so the caching is worth pinning.

    The tests that assert on the recorded `runs` build their runtime with
    `smoke_check=False`, because `start()` now runs an inference of its own and it
    would be the first entry in that list. The smoke check has its own class
    below; here it is noise.
    """

    def test_a_bare_tensor_is_keyed_by_the_graphs_first_input(self, model_file: Path) -> None:
        ort = FakeOrt(declared_inputs=["pixel_values"])
        runtime = OnnxRuntime(ort=ort, smoke_check=False)
        runtime.load(str(model_file), name="m", version="1")
        runtime.start()
        runtime.predict([[0.0]])
        assert ort.sessions[0].runs == [(None, {"pixel_values": [[0.0]]})]

    def test_a_dict_is_passed_through_untouched(self, model_file: Path) -> None:
        """Multi-input graphs have to be callable, and the caller already knows the
        names. Re-keying a dict would make them unreachable."""
        ort = FakeOrt(declared_inputs=["a", "b"])
        runtime = OnnxRuntime(ort=ort)
        runtime.load(str(model_file), name="m", version="1")
        runtime.start()
        feed = {"a": [1.0], "b": [2.0]}
        runtime.predict(feed)
        assert ort.sessions[0].runs == [(None, feed)]

    def test_the_input_name_follows_a_hot_swap(self, model_file: Path) -> None:
        """A new version can rename its input. The cached name must move with the
        session, or the first inference after an upgrade feeds a name the new graph
        has never heard of."""
        ort = FakeOrt(declared_inputs=["old_name"])
        runtime = OnnxRuntime(ort=ort, smoke_check=False)
        runtime.load(str(model_file), name="m", version="1")

        ort.declared_inputs = ["new_name"]
        runtime.load(str(model_file), name="m", version="2")
        runtime.start()
        runtime.predict([[0.0]])
        assert ort.sessions[1].runs == [(None, {"new_name": [[0.0]]})]

    def test_a_bare_tensor_is_refused_when_the_graph_declares_no_inputs(
        self, model_file: Path
    ) -> None:
        """This used to build the literal feed `{None: tensor}` and hand it to
        onnxruntime, which then complained about an invalid feed several frames
        away from the caller who could have fixed it."""
        ort = FakeOrt(declared_inputs=[])
        runtime = OnnxRuntime(ort=ort)
        runtime.load(str(model_file), name="m", version="1")
        runtime.start()
        with pytest.raises(InferenceRuntimeError, match="0 inputs"):
            runtime.predict([[0.0]])
        assert ort.sessions[0].runs == []

    def test_a_bare_tensor_is_refused_when_the_graph_declares_several(
        self, model_file: Path
    ) -> None:
        """Worse than the zero case, because it used to half-succeed: the first
        input was fed and the rest left missing, so the error came from onnxruntime
        and said nothing about the other inputs. The message now names them."""
        ort = FakeOrt(declared_inputs=["image", "mask"])
        runtime = OnnxRuntime(ort=ort)
        runtime.load(str(model_file), name="m", version="1")
        runtime.start()
        with pytest.raises(InferenceRuntimeError) as caught:
            runtime.predict([[0.0]])
        assert "image" in str(caught.value)
        assert "mask" in str(caught.value)
        assert "dict" in str(caught.value)
        assert ort.sessions[0].runs == []

    def test_a_dict_still_works_for_those_graphs(self, model_file: Path) -> None:
        """The refusal is about *keying a bare tensor*, not about the graph. A
        caller who knows the names is always allowed through."""
        ort = FakeOrt(declared_inputs=["image", "mask"])
        runtime = OnnxRuntime(ort=ort)
        runtime.load(str(model_file), name="m", version="1")
        runtime.start()
        runtime.predict({"image": [1.0], "mask": [0.0]})
        assert ort.sessions[0].runs == [(None, {"image": [1.0], "mask": [0.0]})]

    def test_all_outputs_are_requested(self, model_file: Path) -> None:
        """`None` for output names means "every output", which is what a smoke
        check wants: it proves the whole graph executed, not a prefix of it."""
        ort = FakeOrt()
        runtime = OnnxRuntime(ort=ort, smoke_check=False)
        runtime.load(str(model_file), name="m", version="1")
        runtime.start()
        assert runtime.predict([[0.0]]) == ["fake-output"]
        assert ort.sessions[0].runs[0][0] is None


class TestTheSmokeCheck:
    """`start()` proves the graph executes before anything reports RUNNING.

    The failure being guarded against has no load-time symptom: a provider that
    accepts a graph and then has no kernel for one of its nodes, a CUDA library
    that resolves at load and dies at the first launch, an opset the build parses
    but cannot run. All of those used to construct a session cleanly, flip a
    boolean, and tell the dashboard RUNNING.

    Note what the fake can and cannot do here. It cannot tell us onnxruntime
    accepts a nested Python list as a feed -- that was measured against 1.30.0 and
    is asserted for real in tier 2. What it can prove is the whole decision tree:
    which graphs get probed, what tensor gets synthesized, and what happens to
    `is_running` and the reported status in each outcome.
    """

    def test_it_runs_one_inference_through_the_real_predict_path(
        self, model_file: Path
    ) -> None:
        """Not a private shortcut: the probe goes through `predict`, so the feed
        keying it exercises is the same code the first real inference will use.

        `None` for the output names means every output, so a graph whose last node
        is the broken one still fails here.
        """
        ort = FakeOrt()
        runtime = OnnxRuntime(ort=ort)
        runtime.load(str(model_file), name="m", version="1")
        runtime.start()

        assert len(ort.sessions[0].runs) == 1
        output_names, feed = ort.sessions[0].runs[0]
        assert output_names is None
        assert list(feed) == ["input"]
        assert runtime.is_running is True

    def test_the_probe_substitutes_one_for_every_symbolic_dimension(
        self, model_file: Path
    ) -> None:
        """`['N', 1, 28, 28]` -> a batch of one, which is the cheapest thing that
        proves the graph runs. A real exported model reports names like `unk__6`
        or `batch_size` here, and some exporters write `-1` instead of a name."""
        ort = FakeOrt(declared_shape=["batch_size", 1, -1, 3])
        runtime = OnnxRuntime(ort=ort)
        runtime.load(str(model_file), name="m", version="1")
        runtime.start()

        tensor = ort.sessions[0].runs[0][1]["input"]
        assert _shape_of(tensor) == (1, 1, 1, 3)
        assert _flatten(tensor) == [0.0] * 3

    def test_the_probe_tensor_is_not_aliased_across_rows(self, model_file: Path) -> None:
        """`[inner] * n` would hand onnxruntime one row repeated n times. It reads
        the feed and never writes it, so that would work today and become a
        genuinely baffling bug the first time anything did."""
        ort = FakeOrt(declared_shape=[3, 2])
        runtime = OnnxRuntime(ort=ort)
        runtime.load(str(model_file), name="m", version="1")
        runtime.start()

        tensor = ort.sessions[0].runs[0][1]["input"]
        tensor[0][0] = 1.0
        assert tensor == [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

    def test_a_pass_is_reported_in_every_heartbeat(self, model_file: Path) -> None:
        ort = FakeOrt()
        runtime = OnnxRuntime(ort=ort)
        assert runtime.hardware_info()["smoke_check"] == "not run"

        runtime.load(str(model_file), name="m", version="1")
        assert runtime.hardware_info()["smoke_check"] == "not run", (
            "a loaded-but-not-started model has not been proven to execute"
        )

        runtime.start()
        assert runtime.hardware_info()["smoke_check"] == "passed"

    def test_disabling_it_is_reported_rather_than_looking_like_a_pass(
        self, model_file: Path
    ) -> None:
        """`smoke_check=False` exists for the tests above that assert on recorded
        runs. If it ever reaches a device, the heartbeat has to say so -- a silent
        `not run` would read as a model nothing had gotten around to starting."""
        ort = FakeOrt()
        runtime = OnnxRuntime(ort=ort, smoke_check=False)
        runtime.load(str(model_file), name="m", version="1")
        runtime.start()

        assert runtime.is_running is True
        assert ort.sessions[0].runs == []
        assert runtime.hardware_info()["smoke_check"] == "disabled"


class TestTheSmokeCheckSkips:
    """Graphs that cannot be probed start anyway, and say why.

    Every case here is a *skip*, never a refusal to serve. Refusing to serve a
    model because this check could not invent an input for it would be a worse
    failure than the one the check guards against -- so the device goes RUNNING
    and `hardware_info` carries the word "unproven" in effect, which the fleet
    view can separate from a genuine pass.
    """

    def _status(self, ort: FakeOrt, model_file: Path) -> str:
        runtime = OnnxRuntime(ort=ort)
        runtime.load(str(model_file), name="m", version="1")
        runtime.start()
        assert runtime.is_running is True, "a skip must never stop the model serving"
        assert ort.sessions[0].runs == [], "a skip must not reach the session"
        status = runtime.hardware_info()["smoke_check"]
        assert status.startswith("skipped: "), status
        return status

    def test_a_graph_with_several_inputs_is_unproven(self, model_file: Path) -> None:
        """A zero input cannot be matched to one of several names. `predict` already
        refuses a bare tensor for these graphs, and inventing a dict of zeros for
        every input would be guessing at semantics this code does not know."""
        status = self._status(FakeOrt(declared_inputs=["image", "mask"]), model_file)
        assert "2 inputs" in status
        assert "image" in status and "mask" in status

    def test_a_graph_with_no_declared_inputs_is_unproven(self, model_file: Path) -> None:
        status = self._status(FakeOrt(declared_inputs=[]), model_file)
        assert "0 inputs" in status

    @pytest.mark.parametrize(
        "declared",
        [
            "tensor(int64)",  # token ids, class indices
            "tensor(string)",
            "tensor(bool)",
            None,  # a NodeArg that does not report a type at all
        ],
    )
    def test_a_non_float_input_is_unproven_on_purpose(
        self, declared: str | None, model_file: Path
    ) -> None:
        """The restriction is about honesty, not capability -- a nested list is
        accepted for every element type there is.

        Zeros through a float graph is a safe probe. Zeros through a graph whose
        input is an *index* -- a token id, a category, a sequence length -- is not:
        the model may legitimately reject it, and a healthy model reported FAILED
        is the same kind of lie as a broken one reported RUNNING.
        """
        status = self._status(FakeOrt(declared_type=declared), model_file)
        assert repr(declared) in status

    @pytest.mark.parametrize("shape", [None, [], ["N", None, True]])
    def test_an_unusable_shape_is_unproven(self, shape: Any, model_file: Path) -> None:
        """Unknown rank (`None`), a rank-0 input (`[]`), and a dim that is neither
        an int nor a name. `True` is in there deliberately: `bool` is an `int`
        subclass, so a naive check would allocate a one-element axis for it."""
        status = self._status(FakeOrt(declared_shape=shape), model_file)
        assert "usable input shape" in status

    @pytest.mark.parametrize("declared", ["tensor(float16)", "tensor(double)"])
    def test_the_other_float_widths_are_probed(
        self, declared: str, model_file: Path
    ) -> None:
        """The counterpart to the skip list: a float32-only check would leave every
        half-precision model on the fleet unproven, which on a Jetson is most of
        them."""
        ort = FakeOrt(declared_type=declared)
        runtime = OnnxRuntime(ort=ort)
        runtime.load(str(model_file), name="m", version="1")
        runtime.start()
        assert runtime.hardware_info()["smoke_check"] == "passed"

    def test_an_input_over_the_element_cap_is_unproven(self, model_file: Path) -> None:
        """A smoke check that OOMs the agent is worse than no smoke check.

        A Python list of floats costs ~32 bytes an element, so an uncapped probe of
        a 1x3x2048x2048 input would be ~400 MB of transient list on a device with
        8 GB that is also holding CUDA.
        """
        ort = FakeOrt(declared_shape=[1, 3, 2048, 2048])
        status = self._status(ort, model_file)
        assert str(_SMOKE_MAX_ELEMENTS) in status
        assert str(3 * 2048 * 2048) in status

    def test_the_fixture_graph_is_comfortably_under_the_cap(self, model_file: Path) -> None:
        """The cap is a guard against pathological models, not a limit the fleet's
        own model brushes against. 1x1x28x28 is 784 elements."""
        ort = FakeOrt()
        runtime = OnnxRuntime(ort=ort)
        runtime.load(str(model_file), name="m", version="1")
        runtime.start()
        tensor = ort.sessions[0].runs[0][1]["input"]
        assert len(_flatten(tensor)) == 784
        assert 784 < _SMOKE_MAX_ELEMENTS


class TestTheSmokeCheckFails:
    """The case the whole commit is for: loads fine, cannot execute."""

    def test_a_graph_that_cannot_execute_never_reports_running(
        self, model_file: Path
    ) -> None:
        """`ModelStartError` is an `InferenceRuntimeError`, which
        `reconciler.py:157` already maps to FAILED with a backoff -- so this costs
        the reconciler no changes and `MockRuntime` no behaviour.
        """
        boom = RuntimeError("NOT_IMPLEMENTED : Could not find an implementation for Conv")
        ort = FakeOrt(raise_on_run=boom)
        runtime = OnnxRuntime(ort=ort)
        runtime.load(str(model_file), name="m", version="1")

        with pytest.raises(ModelStartError) as caught:
            runtime.start()

        assert runtime.is_running is False, (
            "the flag goes up before the probe so the probe can use the real "
            "predict path; the failure branch has to put it back down"
        )
        assert caught.value.__cause__ is boom, "onnxruntime's own message must survive"
        message = str(caught.value)
        assert "m/1" in message, "the operator needs to know which model this was"
        assert "[1, 1, 28, 28]" in message, "...and what was fed to it"
        assert "NOT_IMPLEMENTED" in message
        assert isinstance(caught.value, InferenceRuntimeError)

    def test_the_failure_reason_reaches_the_heartbeat(self, model_file: Path) -> None:
        """A device in FAILED has already reported the reason through the
        reconciler, but `hardware_info` goes out on every heartbeat including the
        ones after a restart, so the status must not be left at "not run"."""
        ort = FakeOrt(raise_on_run=RuntimeError("libcublas.so.12: cannot open"))
        runtime = OnnxRuntime(ort=ort)
        runtime.load(str(model_file), name="m", version="1")
        with pytest.raises(ModelStartError):
            runtime.start()
        assert runtime.hardware_info()["smoke_check"] == "failed: libcublas.so.12: cannot open"

    @pytest.mark.parametrize("outputs", [[], None])
    def test_a_graph_that_returns_nothing_is_a_failure_not_a_pass(
        self, outputs: Any, model_file: Path
    ) -> None:
        """`session.run(None, ...)` returning nothing means the graph produced no
        output, and a truthiness check on the result is the only thing standing
        between that and a reported pass."""
        ort = FakeOrt(outputs=outputs)
        runtime = OnnxRuntime(ort=ort)
        runtime.load(str(model_file), name="m", version="1")

        with pytest.raises(ModelStartError, match="no outputs"):
            runtime.start()
        assert runtime.is_running is False
        assert runtime.hardware_info()["smoke_check"].startswith("failed:")


class TestTheSmokeStatusIsNotStale:
    """A status that outlived the session it described would be the governance lie
    in a subtler form: the *previous* model's successful inference offered as
    evidence for this one."""

    def test_a_hot_swap_resets_it(self, model_file: Path) -> None:
        ort = FakeOrt()
        runtime = OnnxRuntime(ort=ort)
        runtime.load(str(model_file), name="m", version="1")
        runtime.start()
        assert runtime.hardware_info()["smoke_check"] == "passed"

        runtime.load(str(model_file), name="m", version="2")
        assert runtime.hardware_info()["smoke_check"] == "not run", (
            "v2 inherits nothing from v1's proof"
        )
        runtime.start()
        assert runtime.hardware_info()["smoke_check"] == "passed"
        assert len(ort.sessions[1].runs) == 1

    def test_unload_resets_it(self, model_file: Path) -> None:
        runtime = OnnxRuntime(ort=FakeOrt())
        runtime.load(str(model_file), name="m", version="1")
        runtime.start()
        runtime.unload()
        assert runtime.hardware_info()["smoke_check"] == "not run"

    def test_a_failed_start_followed_by_a_good_version_reports_the_good_one(
        self, model_file: Path
    ) -> None:
        """The upgrade-out-of-a-hole path, which is how a device recovers: v1 is
        broken and FAILED, the operator deploys v2, and the device must stop
        reporting v1's failure."""
        ort = FakeOrt(raise_on_run=RuntimeError("NOT_IMPLEMENTED"))
        runtime = OnnxRuntime(ort=ort)
        runtime.load(str(model_file), name="m", version="1")
        with pytest.raises(ModelStartError):
            runtime.start()

        ort.raise_on_run = None
        runtime.load(str(model_file), name="m", version="2")
        runtime.start()
        assert runtime.is_running is True
        assert runtime.hardware_info()["smoke_check"] == "passed"


class TestHardwareInfo:
    """This dict goes into every heartbeat, so it must never be the reason one
    fails. `base.py:79` says best-effort, never raises -- and that is load-bearing:
    a device that cannot heartbeat is indistinguishable from a device that is gone.
    """

    def test_it_reports_the_providers_the_build_has(self) -> None:
        info = OnnxRuntime(ort=FakeOrt()).hardware_info()
        assert info["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert info["onnxruntime_version"] == "1.30.0"
        assert info["runtime"] == "onnxruntime"
        assert info["cpu_count"]
        assert "-" in info["platform"]

    def test_gpu_available_means_a_gpu_and_not_merely_not_the_cpu(self) -> None:
        """The exact provider list this MacBook reports, measured 2026-10-04.

        The old test was `any(p != "CPUExecutionProvider")`, which this list
        satisfies twice: `AzureExecutionProvider` is a remote inference endpoint
        and not local acceleration at all, and CoreML really does use the GPU but
        is not what the dashboard's column is asking about. A laptop answering yes
        makes the column useless for the Jetson fleet it exists to watch.
        """
        ort = FakeOrt(
            available=[
                "CoreMLExecutionProvider",
                "AzureExecutionProvider",
                "CPUExecutionProvider",
            ]
        )
        assert OnnxRuntime(ort=ort).hardware_info()["gpu_available"] is False

    @pytest.mark.parametrize(
        "provider",
        ["CUDAExecutionProvider", "TensorrtExecutionProvider", "ROCMExecutionProvider"],
    )
    def test_gpu_available_is_true_for_a_real_gpu_provider(self, provider: str) -> None:
        ort = FakeOrt(available=[provider, "CPUExecutionProvider"])
        assert OnnxRuntime(ort=ort).hardware_info()["gpu_available"] is True

    def test_a_cpu_only_build_reports_no_gpu(self) -> None:
        ort = FakeOrt(available=["CPUExecutionProvider"])
        assert OnnxRuntime(ort=ort).hardware_info()["gpu_available"] is False

    def test_the_active_providers_are_reported_once_a_model_is_loaded(
        self, model_file: Path
    ) -> None:
        """The field the acceptance gate actually reads.

        `providers` says what this *build* can do; `active_providers` says what the
        loaded session is using. They differ in the case that matters: a provider
        that cannot handle the graph falls back to the CPU silently and per-node, so
        a fleet that quietly fell back would look like a fleet on the GPU if only
        availability were reported.
        """
        ort = FakeOrt()
        runtime = OnnxRuntime(ort=ort)
        assert "active_providers" not in runtime.hardware_info()

        runtime.load(str(model_file), name="m", version="1")
        info = runtime.hardware_info()
        assert info["active_providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_the_active_providers_come_from_the_session_not_the_build(
        self, model_file: Path
    ) -> None:
        """Pinning the distinction directly: the fake session is made to report
        something the build does not, so a refactor that reads
        `get_available_providers` twice fails here."""
        ort = FakeOrt()
        runtime = OnnxRuntime(ort=ort)
        runtime.load(str(model_file), name="m", version="1")
        ort.sessions[0].providers = ["CPUExecutionProvider"]
        info = runtime.hardware_info()
        assert info["active_providers"] == ["CPUExecutionProvider"]
        assert info["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert info["gpu_available"] is True  # the build has it; the session is not using it

    def test_unloading_stops_reporting_active_providers(self, model_file: Path) -> None:
        runtime = OnnxRuntime(ort=FakeOrt())
        runtime.load(str(model_file), name="m", version="1")
        runtime.unload()
        assert "active_providers" not in runtime.hardware_info()

    def test_it_survives_an_onnxruntime_that_blows_up(self) -> None:
        """A wheel that imported but cannot enumerate providers is a real Jetson
        failure (a half-installed CUDA). The heartbeat still has to go out."""

        class Exploding:
            def get_available_providers(self) -> list[str]:
                raise OSError("libcudart.so.12: cannot open shared object file")

        info = OnnxRuntime(ort=Exploding()).hardware_info()
        assert info["gpu_available"] is None
        assert info["runtime"] == "onnxruntime"

    def test_it_works_before_anything_is_loaded(self) -> None:
        """Heartbeats start before the first model arrives."""
        assert OnnxRuntime(ort=FakeOrt()).hardware_info()["providers"]
