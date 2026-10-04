"""Tier 2: `OnnxRuntime` against the real onnxruntime wheel and real ONNX bytes.

`test_onnx_runtime.py` is tier 1 -- a `FakeOrt` through the constructor, no wheel
needed, part of the default `make test`. It proves the code's *decisions*: which
providers get requested, what the pre-flight checks refuse, how errors are
wrapped. What a fake cannot prove is that onnxruntime agrees with any of it. A
fake session returns whatever we told it to; it would keep passing if the feed
shape were wrong, if the fixture bytes were not a model at all, or if
`get_inputs()` did not exist on the real object.

That is what this file is for, and it is deliberately **off** the default test
path -- marked `onnx` and deselected by `addopts` in `edge-agent/pyproject.toml`.
Run it with:

    make test-agent-onnx

Two reasons, not one. `make test` has to behave identically for everyone, and a
suite whose membership depends on what happens to be pip-installed hands two
developers different results from the same command. And the wheel is not a
laptop dependency here: on a Jetson it comes from NVIDIA's index rather than
PyPI, so "install it to run the tests" is not a neutral instruction.

The fixture bytes come from `conftest.FASHION_ONNX_BASE64`, which is the same
constant `registry/fake.py` serves and which `scripts/build_model.py
--self-check` holds equal to its builder. So the graph these tests load is the
graph the control plane will hand a device, down to the byte.
"""

from __future__ import annotations

import base64
import io
import tarfile
from pathlib import Path

import pytest
from conftest import FASHION_ONNX_BASE64, make_archive

from keeper.runtime.base import InferenceRuntimeError, ModelLoadError, ModelStartError
from keeper.runtime.onnx import _GPU_PROVIDERS, OnnxRuntime

# Module-level so a direct `pytest -m onnx` on a machine without the wheel skips
# the file instead of erroring on the import. The marker is what keeps it out of
# `make test`; this is the belt to that braces.
ort = pytest.importorskip("onnxruntime", reason="tier 2 needs the real wheel")
numpy = pytest.importorskip("numpy", reason="onnxruntime's feeds are numpy arrays")

pytestmark = pytest.mark.onnx

FIXTURE = base64.b64decode(FASHION_ONNX_BASE64)


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    """The fixture graph on disk, which is the only form `load()` accepts."""
    path = tmp_path / "model.onnx"
    path.write_bytes(FIXTURE)
    return path


@pytest.fixture
def runtime() -> OnnxRuntime:
    """A runtime holding the *real* module -- no `ort=` argument anywhere here."""
    return OnnxRuntime()


def image(batch: int = 1):
    return numpy.zeros((batch, 1, 28, 28), dtype=numpy.float32)


class TestTheFixtureIsReal:
    """The claims tier 1 has to take on faith."""

    def test_the_embedded_constant_is_a_model_onnxruntime_will_load(self):
        """The whole point of commits 7 and 8, asserted by the real parser.

        What this replaces was chained SHA-256 pretending to be a graph, which
        `onnxruntime` rejects with INVALID_PROTOBUF. Nothing in the suite noticed,
        because nothing in the suite ever tried.
        """
        session = ort.InferenceSession(FIXTURE, providers=["CPUExecutionProvider"])
        assert [i.name for i in session.get_inputs()] == ["input"]
        assert [o.name for o in session.get_outputs()] == ["output"]

    def test_the_declared_batch_dimension_is_genuinely_dynamic(self, runtime, model_path):
        """`['N', 1, 28, 28]` has to mean it.

        A fixed `1` would still pass every other test in this file, and then fail
        the first time anything fed it two images.
        """
        runtime.load(str(model_path), name="fashion-cnn", version="1")
        runtime.start()
        for batch in (1, 2, 7):
            (logits,) = runtime.predict(image(batch))
            assert logits.shape == (batch, 10)

    def test_the_padding_in_the_artifact_fixtures_is_invisible_to_onnxruntime(
        self, runtime, tmp_path
    ):
        """The 536870911 trick, checked through the agent's own loader.

        `make_archive` pads the graph with an unknown protobuf field to reach a
        requested size, because the download tests need megabytes and the graph
        is a kilobyte. That padding is only safe if a real parser skips it, and
        this is the test that says so -- at a size that is mostly padding.
        """
        with tarfile.open(fileobj=io.BytesIO(make_archive("fashion-cnn", "1", size=65536))) as tar:
            member = tar.extractfile("model.onnx").read()
        assert len(member) == 65536, "the fixture is supposed to be mostly padding here"

        path = tmp_path / "padded.onnx"
        path.write_bytes(member)
        runtime.load(str(path), name="fashion-cnn", version="1")
        runtime.start()
        (logits,) = runtime.predict(image())
        assert logits.shape == (1, 10)


class TestLifecycle:
    def test_load_start_predict_stop(self, runtime, model_path):
        runtime.load(str(model_path), name="fashion-cnn", version="1")
        assert runtime.is_running is False, "load must not start serving"

        with pytest.raises(ModelStartError):
            runtime.predict(image())

        runtime.start()
        assert runtime.is_running is True
        (logits,) = runtime.predict(image())
        assert logits.shape == (1, 10)

        runtime.stop()
        assert runtime.is_running is False
        with pytest.raises(ModelStartError):
            runtime.predict(image())

    def test_start_before_load_is_refused(self, runtime):
        with pytest.raises(ModelStartError):
            runtime.start()

    def test_unload_releases_the_session_and_predict_refuses_again(self, runtime, model_path):
        runtime.load(str(model_path), name="fashion-cnn", version="1")
        runtime.start()
        runtime.unload()
        assert runtime.is_running is False
        with pytest.raises(ModelStartError):
            runtime.predict(image())

    def test_a_second_load_swaps_the_model_and_keeps_serving_possible(self, runtime, model_path):
        """The upgrade path: v1 serving, v2 loaded, v2 serving.

        `load` builds the new session *before* dropping the old one, so a failed
        upgrade leaves the working model in place. Tier 1 asserts the ordering
        against a fake; here the second session is a real one.
        """
        runtime.load(str(model_path), name="fashion-cnn", version="1")
        runtime.start()
        first = runtime.predict(image())[0]

        runtime.load(str(model_path), name="fashion-cnn", version="2")
        assert runtime.is_running is False, "a freshly loaded model is not yet serving"
        runtime.start()
        second = runtime.predict(image())[0]

        # Same bytes, so the same answer. The assertion is that the swap did not
        # leave a dead session behind that silently stopped computing.
        assert numpy.array_equal(first, second)


class TestFeeds:
    def test_a_bare_array_is_keyed_by_the_single_declared_input(self, runtime, model_path):
        runtime.load(str(model_path), name="fashion-cnn", version="1")
        runtime.start()
        (logits,) = runtime.predict(image())
        assert logits.shape == (1, 10)

    def test_an_explicit_dict_feed_gives_the_same_answer(self, runtime, model_path):
        runtime.load(str(model_path), name="fashion-cnn", version="1")
        runtime.start()
        (bare,) = runtime.predict(image())
        (keyed,) = runtime.predict({"input": image()})
        assert numpy.array_equal(bare, keyed)

    def test_a_wrong_shape_raises_rather_than_returning_nonsense(self, runtime, model_path):
        """onnxruntime's own validation, which a fake session has none of.

        Deliberately *not* wrapped into an `InferenceRuntimeError`, and that is
        what the first assertion pins. The pre-flight checks in `load` exist
        because onnxruntime's messages for a missing or unreadable *file* describe
        the parse instead of the cause; this message has the opposite problem,
        which is none. Measured against 1.30.0, three channels instead of one
        gives:

            INVALID_ARGUMENT : Got invalid dimensions for input: input for the
            following indices index: 1 Got: 3 Expected: 1

        Which index, what arrived, what was wanted. Wrapping that would only put
        our own words in front of it.
        """
        runtime.load(str(model_path), name="fashion-cnn", version="1")
        runtime.start()
        with pytest.raises(Exception) as caught:
            runtime.predict(numpy.zeros((1, 3, 28, 28), dtype=numpy.float32))

        assert not isinstance(caught.value, InferenceRuntimeError), (
            "a caller passing the wrong shape is a caller bug, not a runtime "
            "failure -- wrapping it would hide onnxruntime's better message"
        )
        message = str(caught.value)
        assert "invalid dimensions" in message.lower()
        assert "Got: 3" in message and "Expected: 1" in message


class TestLoadFailures:
    """The one pre-flight case left to onnxruntime, and the wrapping of it."""

    def test_bytes_that_are_not_a_model_become_a_ModelLoadError(self, runtime, tmp_path):
        """Measured: real onnxruntime says INVALID_PROTOBUF here, which is a good
        message -- so `load` adds no check of its own and only normalises the
        type. These are exactly the bytes the fixtures used to contain.
        """
        path = tmp_path / "model.onnx"
        path.write_bytes(b"not really a protobuf" * 64)
        with pytest.raises(ModelLoadError) as caught:
            runtime.load(str(path), name="fashion-cnn", version="1")
        assert caught.value.__cause__ is not None, "the original error must stay attached"

    def test_a_truncated_model_is_refused_rather_than_half_loaded(self, runtime, tmp_path):
        path = tmp_path / "model.onnx"
        path.write_bytes(FIXTURE[: len(FIXTURE) // 2])
        with pytest.raises(ModelLoadError):
            runtime.load(str(path), name="fashion-cnn", version="1")

    def test_an_empty_file_is_caught_before_onnxruntime_sees_it(self, runtime, tmp_path):
        """Real onnxruntime leaks a path from its own build machine for this one,
        which is why `load` checks the size itself. The assertion is that the
        operator gets our message and not that.
        """
        path = tmp_path / "model.onnx"
        path.write_bytes(b"")
        with pytest.raises(ModelLoadError) as caught:
            runtime.load(str(path), name="fashion-cnn", version="1")
        assert "empty" in str(caught.value)
        assert caught.value.__cause__ is None


class TestHardwareInfo:
    def test_it_reports_what_this_build_can_do(self, runtime):
        info = runtime.hardware_info()
        assert info["runtime"] == "onnxruntime"
        assert info["onnxruntime_version"] == ort.__version__
        assert isinstance(info["providers"], list) and info["providers"]
        assert "CPUExecutionProvider" in info["providers"], (
            "every onnxruntime build has a CPU fallback; a list without it means "
            "this is reporting something other than what it thinks"
        )

    def test_gpu_available_follows_the_allow_list_and_not_merely_not_cpu(self, runtime):
        """The regression that made the dashboard's GPU column untrustworthy.

        This used to be `any(p != "CPUExecutionProvider")`, which is True on a
        MacBook -- CoreML satisfies it, and so does `AzureExecutionProvider`,
        which is a *remote* endpoint and not local acceleration at all. On the
        Jetson this assertion is the one that must come out True, and if it does
        not, the device is running on the CPU of a machine bought for its GPU.
        """
        info = runtime.hardware_info()
        accelerated = [p for p in info["providers"] if p in _GPU_PROVIDERS]
        assert info["gpu_available"] is bool(accelerated)
        if not accelerated:
            # Spelled out so the failure is readable on a dev laptop: a provider
            # list of CoreML/Azure/CPU must report False.
            assert info["gpu_available"] is False, info["providers"]

    def test_active_providers_appears_only_once_a_model_is_loaded(self, runtime, model_path):
        """Availability and use are different questions, and the gate asks the
        second one. A provider that cannot handle a graph falls back silently and
        per-node, so a fleet on the CPU can look like a fleet on the GPU if only
        `providers` is reported.
        """
        assert "active_providers" not in runtime.hardware_info()

        runtime.load(str(model_path), name="fashion-cnn", version="1")
        info = runtime.hardware_info()
        assert info["active_providers"], "a loaded session must say what it is using"
        assert set(info["active_providers"]) <= set(info["providers"])

    def test_the_preferred_provider_is_the_one_actually_selected(self, runtime, model_path):
        """`load` intersects `_PREFERRED_PROVIDERS` with what is available, and
        onnxruntime then reports back what it took. Those two have to agree, and
        only the real wheel can say whether they do.
        """
        from keeper.runtime.onnx import _PREFERRED_PROVIDERS

        runtime.load(str(model_path), name="fashion-cnn", version="1")
        active = runtime.hardware_info()["active_providers"]
        available = runtime.hardware_info()["providers"]
        wanted = [p for p in _PREFERRED_PROVIDERS if p in available] or ["CPUExecutionProvider"]
        assert active[0] == wanted[0], (
            f"asked for {wanted} and got {active}; the preference list and this "
            "build of onnxruntime disagree"
        )
