#!/usr/bin/env python3
"""Build the ONNX graphs this project runs on, and prove them locally.

Two graphs, for two different jobs:

**`build_minimal_onnx`** -- one `MatMul`, one `Add`, a 1x4 float32 input and a
1x2 float32 output. M3 needed *a* model in the Cloudera AI AI Registry so the
adapter's three fail-safe guesses could be checked against reality; it did not
need a good model. `register_model.py` imports this one.

**`build_fashion_onnx`** -- `Conv` + `Relu` + `GlobalAveragePool` + `Flatten` +
`MatMul` + `Add`, a dynamic batch of 1x28x28 images in and ten logits out.
Added in M4 to replace the test fixtures' unloadable `model.onnx`, which was
chained SHA-256 that onnxruntime rejects outright. Still untrained -- neither
graph has an accuracy story, and this file will not grow one.

This half of the work is separate from `register_model.py` because it runs on a
laptop with no tenant, which makes it the only part that can be iterated on
quickly.

    pip install onnx                 # required
    pip install onnxruntime numpy    # optional, enables the load checks

`onnx` is deliberately in **no** `pyproject.toml`, the same way
`probe_registry.py` needs `cdpcli` and says so here rather than in a
dependency list. Keeping it out is what leaves `make test` runnable on a
laptop with no ML stack.

Usage:

    python scripts/build_model.py                 # write the .onnx and the .tar.gz
    python scripts/build_model.py --self-check    # ...then validate them
    python scripts/build_model.py --self-check --quiet
    python scripts/build_model.py --emit-fixture  # the base64 constant, nothing else

Outputs land in `.dev/m3/`, which is gitignored at the directory level --
`.gitignore` covers `*.onnx` but not `model.tar.gz`, so the directory is what
keeps both out of a public repo. The fixture graph is never written as a file
for that same reason; see the note above `fixture_base64`.

## What `--self-check` actually checks

Eight things have to be true for this artifact to reach a running
`InferenceSession` on the Jetson, and they are enforced in five different
files. The check calls **the real production functions** rather than
restating their rules, so it fails when they change:

  1. metadata carries the MLflow signal       `cai.py` (registry-side, not here)
  2. version status is exactly READY          `cai.py` (registry-side, not here)
  3. the bytes are a gzip tar                 `artifact_service._read_entrypoint`
  4. an `MLmodel` member with `flavors.onnx.data`      same
  5. at least one `*.onnx` in the tree        `ArtifactManager._resolve_entrypoint`
  6. no symlinks, absolute paths or `..`      `keeper._safe_extract`
  7. the `.onnx` bytes genuinely load         `onnxruntime.InferenceSession`
  8. exactly one declared input               `runtime/onnx.py:_input_name`

1 and 2 are properties of the registry, not of these bytes, so only a real
registration settles them -- that is what `register_model.py` is for.

It also checks the fixture graph, and that the base64 copies of it embedded in
`registry/fake.py` and `edge-agent/tests/conftest.py` still equal what
`build_fashion_onnx` produces. Nobody reviews 15 lines of base64; the check is
there so nothing has to.

This is a script mode and **not a pytest test**, on purpose: a test would put
`onnx` on the default `make test` path, and that suite has to stay runnable
with no cloud and no ML dependency. The check still runs before anyone touches
the tenant, which is the point of having it.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import re
import sys
import tarfile
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OUT_DIR = _REPO / ".dev" / "m3"

# The graph. Shapes are fixed rather than dynamic because the only consumer is
# a plumbing proof, and a fixed shape is one less thing for the Jetson's
# runtime to disagree about.
INPUT_NAME = "input"
OUTPUT_NAME = "output"
INPUT_SHAPE = (1, 4)
OUTPUT_SHAPE = (1, 2)

# The *fixture* graph is a second, larger model with the same input and output
# names -- see `build_fashion_onnx`. It exists for the test suite rather than
# for the registry, and its shapes are the ones `registry/fake.py` has always
# claimed in its MLmodel signature.
FIXTURE_MODEL_NAME = "fashion-cnn"
FIXTURE_INPUT_SHAPE: tuple[str | int, ...] = ("N", 1, 28, 28)
FIXTURE_OUTPUT_SHAPE: tuple[str | int, ...] = ("N", 10)
_FIXTURE_CHANNELS = 8
_FIXTURE_CLASSES = 10
# Changing this string changes the fixture bytes, which changes the digest two
# files have embedded. It is versioned for that reason: a future `/v2` is an
# intentional act with a visible diff, not an accident.
_FIXTURE_SEED = b"lighthouse-fashion-cnn/v1"

# Pinned, not defaulted, and both for the same reason: the Jetson runs whatever
# `onnxruntime` wheel NVIDIA ships for aarch64, which is older than the `onnx`
# on this laptop. A current `onnx` emits IR version 10 or 11, and an
# onnxruntime from the 1.17 era refuses to load those outright -- a failure
# that would surface on the device, after a successful registration and a
# successful download, which is the worst possible place to find it.
#
# Opset 13 covers every op used here with room to spare and has been supported
# for years. IR version 9 is the newest that onnxruntime 1.17 accepts.
#
# **The pin costs nothing on a newer runtime, measured rather than assumed.**
# On 2026-10-04 both graphs below were re-stamped as IR 3, 9, 10, 11 and 12 and
# handed to onnxruntime 1.30.0 (with onnx 1.23.1): all ten loaded and ran. So
# IR 9 is forward-compatible, and a Jetson that turns out to have a current
# onnxruntime is **not** a reason to raise these numbers -- it only means the
# `onnxruntime>=1.17` floor has room. Raising them would buy nothing and would
# re-expose the one failure mode that only appears on the device.
OPSET = 13
IR_VERSION = 9

# Fixed so the bytes are reproducible: same input, same digest, every run.
# A changing artifact would make the registry's version lineage meaningless.
_FIXED_MTIME = 0
_PRODUCER = "lighthouse-build-model"

MLFLOW_VERSION = "2.14.1"
ONNX_FLAVOR_VERSION = "1.15.0"


# --------------------------------------------------------------------------
# the graph
# --------------------------------------------------------------------------


def build_minimal_onnx() -> bytes:
    """A serialized ONNX model: `output = input @ W + b`.

    `W` and `b` are initializers with hand-written values rather than anything
    random, so two runs of this function produce identical bytes.
    """
    from onnx import TensorProto, helper

    # Written out literally. The values mean nothing; being *fixed* is the
    # entire requirement.
    weights = [
        [0.10, -0.20],
        [0.30, 0.40],
        [-0.50, 0.60],
        [0.70, -0.80],
    ]
    bias = [0.01, -0.02]

    w_init = _float_tensor("W", weights)
    b_init = _float_tensor("b", [bias])
    # b is 1x2 so the Add broadcasts against the 1x2 MatMul result without
    # relying on opset-dependent unidirectional broadcasting rules.

    graph = helper.make_graph(
        nodes=[
            helper.make_node("MatMul", [INPUT_NAME, "W"], ["hidden"], name="matmul"),
            helper.make_node("Add", ["hidden", "b"], [OUTPUT_NAME], name="add"),
        ],
        name="smoke_test",
        inputs=[
            # The single declared input (requirement 8). `runtime/onnx.py` keys
            # a bare tensor by the one declared name and refuses outright if
            # there is not exactly one, so a second input here would take the
            # edge from "works" to "cannot predict".
            helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, list(INPUT_SHAPE))
        ],
        outputs=[
            helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, list(OUTPUT_SHAPE))
        ],
        initializer=[w_init, b_init],
    )

    model = helper.make_model(
        graph,
        producer_name=_PRODUCER,
        # Pinned rather than left to default to the installed onnx version, so
        # the bytes do not change when someone upgrades onnx.
        producer_version="1",
        opset_imports=[helper.make_operatorsetid("", OPSET)],
    )
    model.ir_version = IR_VERSION

    # Fail here rather than on the device. `full_check` also runs shape
    # inference, which is what catches a MatMul whose operands do not line up.
    import onnx

    onnx.checker.check_model(model, full_check=True)
    return model.SerializeToString()


def _float_tensor(name: str, rows: list[list[float]]):
    """A 2-D float32 initializer built without numpy."""
    flat = [v for row in rows for v in row]
    return _flat_tensor(name, (len(rows), len(rows[0])), flat)


def _flat_tensor(name: str, dims: tuple[int, ...], vals: list[float]):
    """A float32 initializer of any rank, built without numpy.

    numpy is `onnxruntime`'s dependency, not `onnx`'s, and the build half of
    this script must work with `onnx` alone -- requiring numpy to *produce* the
    model would make the optional load check mandatory. That constraint is why
    the fixture's weights are hashed into existence below instead of being
    drawn from `numpy.random`.
    """
    from onnx import TensorProto, helper

    expected = 1
    for d in dims:
        expected *= d
    if len(vals) != expected:
        raise ValueError(f"{name}: {len(vals)} values for dims {dims} (want {expected})")
    return helper.make_tensor(
        name=name,
        data_type=TensorProto.FLOAT,
        dims=list(dims),
        vals=[float(v) for v in vals],
    )


# --------------------------------------------------------------------------
# the fixture graph
# --------------------------------------------------------------------------


def _deterministic_floats(count: int, *, seed: bytes, scale: float = 0.05) -> list[float]:
    """`count` floats in (-scale, scale), from a chained SHA-256 stream.

    The same construction `registry/fake.py:_deterministic_payload` uses for
    artifact bytes, carried one step further: each 16-bit slice of the digest is
    read as a signed int16 and scaled into a small symmetric range, so the
    initializers look like weights rather than like noise and `Relu` has
    something to clip. Two bytes per value, not four, because an int16 maps onto
    a float32 exactly -- no rounding to argue about across architectures.

    Chained (each digest hashes the previous one) rather than counter-based, so
    the stream cannot accidentally repeat, and seeded per tensor so the Conv
    kernel and the classifier are not the same numbers twice.

    `round(..., 6)` is the one thing here that is load-bearing for determinism
    *across Python builds*: it makes the decimal values written into the
    protobuf short and exact, instead of depending on repr of a full-precision
    quotient.
    """
    out: list[float] = []
    block = seed
    while len(out) < count:
        block = hashlib.sha256(block).digest()
        for i in range(0, len(block), 2):
            if len(out) == count:
                break
            raw = int.from_bytes(block[i : i + 2], "big", signed=True)
            out.append(round(raw * scale / 32768, 6))
    return out


def build_fashion_onnx() -> bytes:
    """A serialized ONNX model shaped like a Fashion-MNIST classifier.

    `Conv(8,1,3,3)` -> `Relu` -> `GlobalAveragePool` -> `Flatten` ->
    `MatMul(8x10)` -> `Add`: a dynamic batch of 1x28x28 images in, ten logits
    out, about a kilobyte of weights. Untrained and never claimed otherwise --
    the outputs are meaningless. What it is for is making the *plumbing* honest:

    * `registry/fake.py` already declares a Fashion-MNIST signature in the
      MLmodel it synthesizes, while its `model.onnx` member is chained SHA-256
      that onnxruntime rejects with `INVALID_PROTOBUF`. These bytes make that
      declaration true without editing it.
    * `Conv` is the reason the graph is not just a pool and a MatMul. It is the
      op `CUDAExecutionProvider` and `TensorrtExecutionProvider` actually claim
      nodes for, so a device loading this fixture exercises its accelerator --
      the fixture doubles as the GPU smoke test the M4 acceptance gate needs.
    * `GlobalAveragePool` is why it is a kilobyte and not 31 KB. A real dense
      `784x10` head would be 440 lines of base64, and the constant has to be
      *duplicated* into the edge-agent test tree (that tree may not import
      `lighthouse`), so a bigger weight matrix costs 880 lines of diff for
      nothing.

    The batch dimension is the symbol `N`, not a fixed 1, because the fixture is
    loaded by tests that have no reason to agree on a batch size.
    """
    from onnx import TensorProto, helper

    c, k = _FIXTURE_CHANNELS, _FIXTURE_CLASSES
    initializers = [
        _flat_tensor(
            "conv_w", (c, 1, 3, 3), _deterministic_floats(c * 9, seed=_FIXTURE_SEED + b"/conv_w")
        ),
        _flat_tensor("conv_b", (c,), _deterministic_floats(c, seed=_FIXTURE_SEED + b"/conv_b")),
        _flat_tensor("fc_w", (c, k), _deterministic_floats(c * k, seed=_FIXTURE_SEED + b"/fc_w")),
        # 1xk rather than k, for the same reason `build_minimal_onnx` uses 1x2:
        # it broadcasts against the MatMul result without depending on
        # opset-dependent unidirectional broadcasting.
        _flat_tensor("fc_b", (1, k), _deterministic_floats(k, seed=_FIXTURE_SEED + b"/fc_b")),
    ]

    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Conv",
                [INPUT_NAME, "conv_w", "conv_b"],
                ["conv_out"],
                name="conv",
                kernel_shape=[3, 3],
                # Spelled out rather than defaulted. An older onnxruntime that
                # disagreed about an implicit default would fail on the device,
                # and `pads` in particular decides the output shape.
                pads=[1, 1, 1, 1],
                strides=[1, 1],
                dilations=[1, 1],
                group=1,
            ),
            helper.make_node("Relu", ["conv_out"], ["relu_out"], name="relu"),
            helper.make_node("GlobalAveragePool", ["relu_out"], ["pooled"], name="pool"),
            helper.make_node("Flatten", ["pooled"], ["features"], name="flatten", axis=1),
            helper.make_node("MatMul", ["features", "fc_w"], ["logits"], name="matmul"),
            helper.make_node("Add", ["logits", "fc_b"], [OUTPUT_NAME], name="add"),
        ],
        name="fashion_cnn",
        # One declared input, like the minimal graph: `runtime/onnx.py` refuses
        # to key a bare tensor for a graph with zero or several.
        inputs=[
            helper.make_tensor_value_info(
                INPUT_NAME, TensorProto.FLOAT, list(FIXTURE_INPUT_SHAPE)
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                OUTPUT_NAME, TensorProto.FLOAT, list(FIXTURE_OUTPUT_SHAPE)
            )
        ],
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        producer_name=_PRODUCER,
        producer_version="1",
        opset_imports=[helper.make_operatorsetid("", OPSET)],
    )
    model.ir_version = IR_VERSION

    import onnx

    onnx.checker.check_model(model, full_check=True)
    return model.SerializeToString()


# --------------------------------------------------------------------------
# shipping the fixture as source
# --------------------------------------------------------------------------
#
# As a base64 constant and **not** as a `.onnx` file next to `fake.py`.
# `.gitignore` contains `*.onnx`, so a committed binary fixture would be
# silently excluded: present for whoever generated it, missing for everyone who
# clones. `git add -f` would work and would last exactly until the next person
# regenerated it. Base64 in a `.py` cannot be ignored by accident, diffs
# visibly, and -- being in two files that may not import each other -- is held
# in sync by `--self-check` below rather than by hope.

_FIXTURE_WIDTH = 76


def fixture_base64(onnx_bytes: bytes) -> str:
    """The fixture bytes as wrapped base64, one chunk per line."""
    packed = base64.b64encode(onnx_bytes).decode("ascii")
    return "\n".join(packed[i : i + _FIXTURE_WIDTH] for i in range(0, len(packed), _FIXTURE_WIDTH))


def emit_fixture(onnx_bytes: bytes) -> str:
    """A paste-ready Python assignment for the two files that embed the fixture."""
    body = "\n".join(f'    "{line}"' for line in fixture_base64(onnx_bytes).splitlines())
    digest = hashlib.sha256(onnx_bytes).hexdigest()
    return (
        f"# Generated by `python scripts/build_model.py --emit-fixture`. Do not hand-edit:\n"
        f"# `--self-check` compares this constant against the builder's output and fails\n"
        f"# if they have drifted. {len(onnx_bytes)} bytes, sha256 {digest[:16]}...\n"
        f"FASHION_ONNX_BASE64 = (\n{body}\n)\n"
    )


# The two places the constant has to live. `fake.py` is the control plane's
# fixture source; `conftest.py` is the edge-agent's, and it cannot import the
# first because `edge-agent/tests/conftest.py` exists partly to prove the agent
# never depends on `lighthouse`. Duplication is the price of that boundary, so
# the drift check is what makes it safe.
_FIXTURE_SITES = (
    _REPO / "control-plane" / "src" / "lighthouse" / "registry" / "fake.py",
    _REPO / "edge-agent" / "tests" / "conftest.py",
)


def _embedded_fixture(path: Path) -> str | None:
    """The `FASHION_ONNX_BASE64` literal in `path`, whitespace removed.

    Read as *text*, not imported, for both sites. Importing `fake.py` would
    prove only that some module-level name decodes; reading the file proves the
    bytes are actually written down in the file that ships. And the edge-agent
    copy lives in a conftest, which is not importable as a module at all.
    """
    if not path.is_file():
        return None
    match = re.search(r"^FASHION_ONNX_BASE64\s*=\s*\(([^)]*)\)", path.read_text(), re.M | re.S)
    if match is None:
        return None
    return "".join(re.findall(r'"([^"]*)"', match.group(1)))


# --------------------------------------------------------------------------
# the MLflow layout
# --------------------------------------------------------------------------


def mlmodel_yaml(model_name: str, onnx_path: str = "model.onnx") -> str:
    """The `MLmodel` descriptor, in the shape `mlflow.onnx.log_model()` writes.

    Deliberately parallel to `registry/fake.py:_mlmodel_yaml`, whose docstring
    says "If the shape here drifted from reality, M2 would be the first thing
    to notice." This is that comparison becoming checkable: the signature
    describes *this* graph (1x4 -> 1x2) rather than the fake's Fashion-MNIST
    shape, but every key the artifact service reads is in the same place.

    `flavors.onnx.data` is the load-bearing line -- `_read_entrypoint` pulls
    the entrypoint out of exactly that path, and nothing else here is read by
    the control plane.
    """
    uuid = hashlib.sha256(model_name.encode()).hexdigest()[:32]
    in_shape = list(INPUT_SHAPE)
    out_shape = list(OUTPUT_SHAPE)
    return (
        "artifact_path: model\n"
        "flavors:\n"
        "  onnx:\n"
        f"    data: {onnx_path}\n"
        f"    onnx_version: {ONNX_FLAVOR_VERSION}\n"
        "    providers:\n"
        "    - CPUExecutionProvider\n"
        "  python_function:\n"
        "    env: conda.yaml\n"
        "    loader_module: mlflow.onnx\n"
        f"model_uuid: {uuid}\n"
        f"mlflow_version: {MLFLOW_VERSION}\n"
        "utc_time_created: '2026-01-01 00:00:00.000000'\n"
        "signature:\n"
        f'  inputs: \'[{{"name": "{INPUT_NAME}", "type": "tensor", "tensor-spec":'
        f' {{"dtype": "float32", "shape": {in_shape}}}}}]\'\n'
        f'  outputs: \'[{{"name": "{OUTPUT_NAME}", "type": "tensor", "tensor-spec":'
        f' {{"dtype": "float32", "shape": {out_shape}}}}}]\'\n'
    )


CONDA_YAML = (
    "channels:\n- conda-forge\ndependencies:\n- python=3.11\n"
    "- pip\n- pip:\n"
    f"  - onnx=={ONNX_FLAVOR_VERSION}\n  - onnxruntime==1.17.0\n"
)


def build_mlflow_tar_gz(
    onnx_bytes: bytes, model_name: str = "smoke-test", onnx_path: str = "model.onnx"
) -> bytes:
    """Package the graph as a `model.tar.gz` in MLflow layout.

    Three flat members at the tar root -- `MLmodel`, `model.onnx`,
    `conda.yaml` -- matching `build_fake_artifact`. Determinism comes from
    `mtime=0` on the gzip header and pinned `TarInfo` fields; without those the
    same model produces a different digest every run.

    Note this is what the *control plane* will serve, and it is built here only
    so the self-check has something real to validate. In the Session, MLflow
    writes the equivalent itself -- the point of mirroring its layout is that
    the self-check then proves the consumers before MLflow is involved.
    """
    members: list[tuple[str, bytes]] = [
        ("MLmodel", mlmodel_yaml(model_name, onnx_path).encode("utf-8")),
        (onnx_path, onnx_bytes),
        ("conda.yaml", CONDA_YAML.encode("utf-8")),
    ]

    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=_FIXED_MTIME) as gz:
        # "w", not "w:gz" -- the wrapper above already compresses, and letting
        # tarfile do it too would reintroduce its own mtime.
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for member_name, data in sorted(members):
                info = tarfile.TarInfo(name=member_name)
                info.size = len(data)
                info.mtime = _FIXED_MTIME
                info.mode = 0o644
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                tar.addfile(info, io.BytesIO(data))
    return raw.getvalue()


# --------------------------------------------------------------------------
# the self-check
# --------------------------------------------------------------------------


class CheckFailed(Exception):
    """A requirement the real consumers enforce was not met."""


def _ok(label: str, detail: str = "") -> None:
    print(f"  ok    {label}" + (f" -- {detail}" if detail else ""))


def _skip(label: str, why: str) -> None:
    print(f"  skip  {label} -- {why}")


def check_fixture() -> list[str]:
    """Validate the fixture graph and the two source copies of its base64.

    The drift assertion is the reason this exists. A base64 constant in a `.py`
    is invisible in review -- nobody reads 15 lines of base64 and nobody would
    notice if the builder below it started producing something else. So the
    relationship is checked rather than trusted: if the graph changes and the
    constants are not regenerated, this fails and names the command to run.

    Returns the requirements that could not be checked here.
    """
    skipped: list[str] = []
    fixture_bytes = build_fashion_onnx()

    if build_fashion_onnx() != fixture_bytes:
        raise CheckFailed("build_fashion_onnx() is not byte-reproducible across calls")
    _ok(
        "fixture graph builds reproducibly",
        f"{len(fixture_bytes)} bytes, {len(fixture_base64(fixture_bytes).splitlines())} "
        "lines of base64",
    )

    expected = base64.b64encode(fixture_bytes).decode("ascii")
    for path in _FIXTURE_SITES:
        # Repo-relative for the same reason the write lines are: an absolute
        # path in pasted output is a small environment leak in a public repo.
        where = path.relative_to(_REPO) if path.is_relative_to(_REPO) else path.name
        embedded = _embedded_fixture(path)
        if embedded is None:
            # Not a failure *yet*: this check lands one commit before the two
            # constants do, deliberately, so that the commit which adds them
            # is already guarded when it is written.
            skipped.append(f"the embedded fixture in {where} (no FASHION_ONNX_BASE64 there yet)")
            _skip(f"fixture embedded in {where}", "constant not present")
            continue
        if embedded != expected:
            # The two failures look different and want different reactions: a
            # length mismatch is usually a half-pasted literal, while equal
            # lengths mean the builder changed under a stale constant.
            detail = (
                f"{len(embedded)} base64 chars embedded vs {len(expected)} built"
                if len(embedded) != len(expected)
                else "same length, different content -- the graph changed"
            )
            raise CheckFailed(
                f"{where}'s FASHION_ONNX_BASE64 is not what build_fashion_onnx() "
                f"produces ({detail}). Regenerate it with:\n"
                "    python scripts/build_model.py --emit-fixture"
            )
        _ok(f"fixture embedded in {where}", "matches the builder")

    try:
        import numpy
        import onnxruntime
    except ImportError:
        skipped.append("the fixture graph loading and running under onnxruntime")
        _skip("fixture load", "onnxruntime/numpy not installed")
        return skipped

    session = onnxruntime.InferenceSession(fixture_bytes, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].name != INPUT_NAME:
        raise CheckFailed(
            f"the fixture declares {[i.name for i in inputs]}; the agent needs exactly "
            f"one input named {INPUT_NAME!r} to key a bare tensor by"
        )
    if list(inputs[0].shape) != list(FIXTURE_INPUT_SHAPE):
        raise CheckFailed(
            f"the fixture reports {inputs[0].shape}, but fake.py's MLmodel signature "
            f"claims {list(FIXTURE_INPUT_SHAPE)} -- the fixture and the descriptor "
            "that ships with it must agree"
        )

    # Batch 3, not 1: the declared `N` is only proven dynamic by feeding
    # something that is not 1, and the fixture's callers do not agree on a batch.
    batch = 3
    probe = numpy.zeros((batch, 1, 28, 28), dtype=numpy.float32)
    (result,) = session.run(None, {INPUT_NAME: probe})
    if tuple(result.shape) != (batch, _FIXTURE_CLASSES):
        raise CheckFailed(
            f"fixture output shape is {tuple(result.shape)}, expected "
            f"{(batch, _FIXTURE_CLASSES)}"
        )
    _ok(
        "fixture loads and runs",
        f"{inputs[0].shape} -> {tuple(result.shape)} on {session.get_providers()[0]}",
    )
    return skipped


def self_check(onnx_bytes: bytes, tar_bytes: bytes, model_name: str) -> list[str]:
    """Validate the artifact by calling the code that will consume it.

    Returns the list of requirements that could not be checked here, so the
    caller can say so out loud rather than letting a skipped check read as a
    passed one.
    """
    from lighthouse_contracts import ModelRef, Packaging

    from keeper.artifact_manager import ArtifactManager, _safe_extract
    from lighthouse.services.artifact_service import _read_entrypoint

    skipped: list[str] = []

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        tar_path = tmp / "model.tar.gz"
        tar_path.write_bytes(tar_bytes)

        # -- 3 and 4: gzip tar, with an MLmodel carrying flavors.onnx.data ---
        #
        # `_read_entrypoint` is the control plane's only reader of the
        # artifact's internals. It returns None for anything it cannot parse,
        # and None is the silent-permanent-failure path -- so a None here is
        # the single most important thing this check can catch.
        entrypoint = _read_entrypoint(tar_path, Packaging.MLFLOW_TAR_GZ)
        if entrypoint is None:
            raise CheckFailed(
                "_read_entrypoint returned None: the control plane could not find "
                "flavors.onnx.data in the MLmodel. This is requirement 4, and it "
                "is the failure that strands a model forever rather than erroring."
            )
        if entrypoint != "model.onnx":
            raise CheckFailed(f"entrypoint is {entrypoint!r}, expected 'model.onnx'")
        _ok("3+4 gzip tar with a readable MLmodel", f"entrypoint = {entrypoint}")

        # -- 6: the tar is safe to unpack --------------------------------
        #
        # These bytes will have crossed a network by the time the agent sees
        # them, and `_safe_extract` is the agent's own guard. Running the real
        # one proves the packaging above does not trip it (a PAX header or an
        # absolute member name would).
        staging = tmp / "staging"
        staging.mkdir()
        with tarfile.open(tar_path, "r:gz") as tar:
            names = sorted(m.name for m in tar.getmembers())
            _safe_extract(tar, staging)
        _ok("6 _safe_extract accepted the tar", f"members = {names}")

        # -- 5: an .onnx exists where the agent looks for it ---------------
        #
        # `_resolve_entrypoint` touches no instance state, so it is called
        # unbound rather than standing up a whole ArtifactManager (which wants
        # a config and a cache directory). If it ever starts using `self` this
        # line fails loudly, which is the right outcome.
        model_ref = ModelRef(
            name=model_name,
            version="1",
            artifact_uri="/api/v1/devices/self-check/artifact",
            # The real digest of the real bytes. `ModelRef` requires it, and
            # passing the true value costs nothing while a placeholder would be
            # one more thing in this file that is not what it claims to be.
            sha256=hashlib.sha256(tar_bytes).hexdigest(),
            size_bytes=len(tar_bytes),
            packaging=Packaging.MLFLOW_TAR_GZ,
            entrypoint=entrypoint,
        )
        resolved = ArtifactManager._resolve_entrypoint(None, staging, model_ref)
        if resolved is None:
            raise CheckFailed(
                "the agent found no .onnx in the unpacked tree (requirement 5): "
                "_activate would raise ArtifactError here"
            )
        _ok("5 agent resolved the entrypoint", str(resolved.relative_to(staging)))

        # -- 7 and 8: the bytes load, with exactly one input ---------------
        try:
            import numpy
            import onnxruntime
        except ImportError:
            skipped.append(
                "7+8 the graph loading under onnxruntime (pip install onnxruntime numpy)"
            )
            _skip("7+8 onnxruntime load", "onnxruntime/numpy not installed")
        else:
            session = onnxruntime.InferenceSession(
                str(resolved), providers=["CPUExecutionProvider"]
            )
            inputs = session.get_inputs()
            if len(inputs) != 1:
                raise CheckFailed(
                    f"the graph declares {len(inputs)} inputs; runtime/onnx.py can only "
                    "key a bare tensor when there is exactly one (requirement 8)"
                )
            if inputs[0].name != INPUT_NAME:
                raise CheckFailed(
                    f"input is named {inputs[0].name!r}, expected {INPUT_NAME!r}"
                )
            _ok(
                "8 a single declared input",
                f"{inputs[0].name} {inputs[0].shape} {inputs[0].type}",
            )

            # Actually run it. Loading proves the file parses; running proves
            # the shapes compose, which is what a MatMul typo breaks.
            probe = numpy.zeros(INPUT_SHAPE, dtype=numpy.float32)
            (result,) = session.run(None, {INPUT_NAME: probe})
            if tuple(result.shape) != OUTPUT_SHAPE:
                raise CheckFailed(
                    f"output shape is {tuple(result.shape)}, expected {OUTPUT_SHAPE}"
                )
            _ok("7 inference ran", f"{INPUT_SHAPE} -> {tuple(result.shape)}")

    # -- the shape fake.py claims, against the shape we just built ---------
    #
    # `fake.py` is a hand-built guess at MLflow's output that 387 tests are
    # written against. If its member names drift from what a real artifact
    # carries, those tests pass while production fails -- so the comparison is
    # worth making explicitly rather than assuming.
    try:
        from lighthouse.registry.fake import build_fake_artifact
    except ImportError:  # pragma: no cover - control-plane always present here
        skipped.append("the fake-vs-real layout comparison")
    else:
        with tarfile.open(fileobj=io.BytesIO(build_fake_artifact("x", "1")), mode="r:gz") as t:
            fake_names = sorted(m.name for m in t.getmembers())
        if fake_names != names:
            raise CheckFailed(
                f"fake.py builds {fake_names} but this builds {names}; the test "
                "suite's fixture no longer mirrors the real artifact layout"
            )
        _ok("fake.py mirrors this layout", f"{fake_names}")

    # -- the fixture graph, and the two copies of it in the source tree -----
    skipped.extend(check_fixture())

    # -- determinism -------------------------------------------------------
    #
    # Asserted rather than described: the registry's version lineage assumes a
    # given version's bytes do not change, and `cache_key` is built from it.
    if build_minimal_onnx() != onnx_bytes:
        raise CheckFailed("build_minimal_onnx() is not byte-reproducible across calls")
    if build_mlflow_tar_gz(onnx_bytes, model_name) != tar_bytes:
        raise CheckFailed("build_mlflow_tar_gz() is not byte-reproducible across calls")
    _ok("bytes are reproducible", f"sha256 = {hashlib.sha256(tar_bytes).hexdigest()[:16]}...")

    return skipped


# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the minimal ONNX artifact for M3, and optionally validate it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        default="smoke-test",
        help="name recorded in the MLmodel descriptor (default: smoke-test)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_OUT_DIR,
        help=f"where to write model.onnx and model.tar.gz (default: {_OUT_DIR})",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="validate the artifact against the real control-plane and agent code",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="build in memory only; useful with --self-check",
    )
    parser.add_argument(
        "--emit-fixture",
        action="store_true",
        help="print the fixture graph as a base64 Python constant and exit "
        "(paste into registry/fake.py and edge-agent/tests/conftest.py)",
    )
    parser.add_argument("--quiet", action="store_true", help="only report failures")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        import onnx  # noqa: F401
    except ImportError:
        print(
            "onnx is not installed. It is deliberately not a dependency of any\n"
            "package here -- install it ad hoc:\n\n    pip install onnx\n",
            file=sys.stderr,
        )
        return 2

    if args.emit_fixture:
        # Nothing else on stdout, so the output can be redirected or piped into
        # a clipboard without a header to strip.
        print(emit_fixture(build_fashion_onnx()), end="")
        return 0

    onnx_bytes = build_minimal_onnx()
    tar_bytes = build_mlflow_tar_gz(onnx_bytes, args.model_name)

    if not args.quiet:
        print(
            f"built {args.model_name}: {len(onnx_bytes)} bytes of ONNX, "
            f"{len(tar_bytes)} bytes packaged"
        )

    if not args.no_write:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        onnx_file = args.out_dir / "model.onnx"
        tar_file = args.out_dir / "model.tar.gz"
        onnx_file.write_bytes(onnx_bytes)
        tar_file.write_bytes(tar_bytes)
        if not args.quiet:
            # Relative to the repo so the line is the same on every machine --
            # an absolute path here would be a small environment leak in a
            # public repo's pasted output.
            print(f"  wrote {onnx_file.relative_to(_REPO)}")
            print(f"  wrote {tar_file.relative_to(_REPO)}")

    if not args.self_check:
        if not args.quiet:
            print("\nrun with --self-check to validate it against the consuming code")
        return 0

    print("\nself-check -- calling the real consumers, not a restatement of their rules")
    try:
        skipped = self_check(onnx_bytes, tar_bytes, args.model_name)
    except CheckFailed as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # The consumers raise their own types -- the agent's `ArtifactError` for
        # an unsafe tar, onnxruntime's `InvalidProtobuf` for bytes that are not
        # a model. Those are real failures, not bugs in this script, and a bare
        # traceback in a CAI Session's output is just noise around the one line
        # that matters.
        print(f"\nFAILED [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 1

    if skipped:
        print("\nnot checked here:")
        for item in skipped:
            print(f"  - {item}")
    print(
        "\nrequirements 1 and 2 (MLflow metadata, status READY) are properties of\n"
        "the registry, not of these bytes. Only register_model.py settles them."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
