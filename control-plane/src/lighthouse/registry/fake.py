"""An in-process ModelRegistry that exercises the real artifact path.

A fake that returns a URI string would leave the entire ingest path -- stream,
hash, unpack, cache -- untested until the first live CAI session. So this one
synthesizes genuine gzipped MLflow-layout tarballs.

The bytes are **byte-for-byte reproducible across processes and machines**. That
is not fastidiousness: the artifact cache is keyed by content lineage and the
device verifies a SHA-256, so a fake whose digest changed per run would make
every cache and checksum test intermittently fail, and intermittent checksum
failures are genuinely horrible to debug. Determinism needs three things, all of
which are easy to lose by accident:

  * `gzip.GzipFile(mtime=0)` -- the gzip header embeds a timestamp by default;
  * every TarInfo's mtime/uid/gid/uname/gname/mode pinned;
  * members added in a fixed order with a deterministic payload generator.

The payload standing in for the ONNX graph is deterministic pseudo-random data,
not a loadable model. The mock runtime verifies presence, size and digest, which
is all the simulated device can meaningfully assert. M4 swaps in a real ONNX
fixture for the real onnxruntime.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import tarfile
from datetime import datetime, timedelta, timezone

from lighthouse_contracts import ArtifactFormat, Packaging

from .base import (
    ArtifactStream,
    ModelNotFound,
    ModelRegistry,
    RegistryModelVersion,
    RegistryUnavailable,
    UnsupportedFlavor,
    VersionNotReady,
)

# Fixed epoch for all tar member timestamps. Any constant works; it just must not
# be "now".
_FIXED_MTIME = 0

# Base instant for synthetic created_at values, so model listings are stable too.
_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _deterministic_payload(seed: str, size: int) -> bytes:
    """Reproducible pseudo-random bytes.

    Chained SHA-256 over a counter: no RNG state, no seeding API differences
    between Python versions, identical output everywhere.
    """
    out = bytearray()
    counter = 0
    seed_bytes = seed.encode("utf-8")
    while len(out) < size:
        out += hashlib.sha256(seed_bytes + counter.to_bytes(8, "big")).digest()
        counter += 1
    return bytes(out[:size])


def _mlmodel_yaml(model_name: str, version: str, onnx_path: str) -> str:
    """An MLflow `MLmodel` descriptor with an onnx flavor.

    Mirrors what `mlflow.onnx.log_model()` writes, because the artifact service
    reads `flavors.onnx.data` out of this to learn the entrypoint. If the shape
    here drifted from reality, M2 would be the first thing to notice.
    """
    return (
        "artifact_path: model\n"
        "flavors:\n"
        "  onnx:\n"
        f"    data: {onnx_path}\n"
        "    onnx_version: 1.15.0\n"
        "    providers:\n"
        "    - CUDAExecutionProvider\n"
        "    - CPUExecutionProvider\n"
        "  python_function:\n"
        "    env: conda.yaml\n"
        "    loader_module: mlflow.onnx\n"
        f"model_uuid: {hashlib.sha256(f'{model_name}:{version}'.encode()).hexdigest()[:32]}\n"
        "mlflow_version: 2.14.1\n"
        f"utc_time_created: '2026-01-01 00:00:00.000000'\n"
        "signature:\n"
        '  inputs: ''[{"name": "input", "type": "tensor", "tensor-spec": {"dtype":'
        ' "float32", "shape": [-1, 1, 28, 28]}}]''\n'
        '  outputs: ''[{"name": "output", "type": "tensor", "tensor-spec": {"dtype":'
        ' "float32", "shape": [-1, 10]}}]''\n'
    )


def build_fake_artifact(
    model_name: str,
    version: str,
    payload_size: int = 64 * 1024,
    onnx_path: str = "model.onnx",
) -> bytes:
    """Build a deterministic `model.tar.gz` in MLflow layout.

    Exposed as a module-level function so tests can assert the digest is stable
    without standing up a registry.
    """
    raw = io.BytesIO()
    # mtime=0 so the gzip header carries no wall-clock timestamp.
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=_FIXED_MTIME) as gz:
        # "w" (not "w:gz") -- gzip is already handled by the wrapper above, and
        # letting tarfile compress too would reintroduce its own mtime.
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            members: list[tuple[str, bytes]] = [
                ("MLmodel", _mlmodel_yaml(model_name, version, onnx_path).encode("utf-8")),
                (onnx_path, _deterministic_payload(f"{model_name}:{version}", payload_size)),
                (
                    "conda.yaml",
                    b"channels:\n- conda-forge\ndependencies:\n- python=3.11\n"
                    b"- pip\n- pip:\n  - onnx==1.15.0\n  - onnxruntime==1.17.0\n",
                ),
            ]
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


class FakeModelRegistry(ModelRegistry):
    """Local stand-in for the CAI registry, with fault injection.

    Fault injection covers the registry-side failures the spec's test matrix
    needs (SS22) without mocking a network. Note that artifact *corruption* is
    injected in the serving layer, not here: the control plane hashes whatever it
    reads, so a registry cannot make it compute a wrong digest -- only the bytes
    served to the device afterwards can disagree with it.
    """

    def __init__(
        self,
        models: dict[str, list[str]] | None = None,
        *,
        payload_size: int = 64 * 1024,
        unavailable: bool = False,
        not_ready: set[tuple[str, str]] | None = None,
        no_onnx_flavor: set[tuple[str, str]] | None = None,
    ) -> None:
        self._models: dict[str, list[str]] = models or {
            "fashion-cnn": ["1", "2", "3"],
            "fraud-detector": ["6", "7"],
        }
        self._payload_size = payload_size
        self.unavailable = unavailable
        self.not_ready = not_ready or set()
        self.no_onnx_flavor = no_onnx_flavor or set()

    @property
    def name(self) -> str:
        return "fake"

    # -- fault injection -------------------------------------------------

    def _guard(self) -> None:
        if self.unavailable:
            raise RegistryUnavailable("fake registry marked unavailable")

    def set_unavailable(self, value: bool) -> None:
        """Flip reachability at runtime, for the dev harness."""
        self.unavailable = value

    def add_version(self, model_name: str, version: str) -> None:
        """Register a new version, so the dashboard can show one appearing."""
        self._models.setdefault(model_name, []).append(version)

    # -- ModelRegistry ---------------------------------------------------

    def list_models(self) -> list[str]:
        self._guard()
        return sorted(self._models)

    def list_versions(self, model_name: str) -> list[RegistryModelVersion]:
        self._guard()
        if model_name not in self._models:
            raise ModelNotFound(f"no such model: {model_name}")
        return [self._build(model_name, v) for v in self._models[model_name]]

    def get_version(self, model_name: str, version: str) -> RegistryModelVersion:
        self._guard()
        if model_name not in self._models:
            raise ModelNotFound(f"no such model: {model_name}")
        if version not in self._models[model_name]:
            raise ModelNotFound(f"no version {version} of {model_name}")
        if (model_name, version) in self.not_ready:
            raise VersionNotReady(f"{model_name} v{version} is still building")
        if (model_name, version) in self.no_onnx_flavor:
            raise UnsupportedFlavor(
                f"{model_name} v{version} has no onnx flavor; the edge cannot run it"
            )
        return self._build(model_name, version)

    def open_artifact(self, mv: RegistryModelVersion) -> ArtifactStream:
        self._guard()
        data = build_fake_artifact(mv.name, mv.version, self._payload_size)
        return ArtifactStream(
            fileobj=io.BytesIO(data),
            packaging=Packaging.MLFLOW_TAR_GZ,
            size_bytes=len(data),
            source_uri=mv.artifact_uri,
        )

    def get_artifact_uri(self, mv: RegistryModelVersion) -> str:
        return mv.artifact_uri

    def ping(self) -> bool:
        return not self.unavailable

    # -- internals -------------------------------------------------------

    def _build(self, model_name: str, version: str) -> RegistryModelVersion:
        # Stable synthetic lineage ids shaped like the registry's real ones
        # (four dash-separated 4-character groups).
        model_id = _fake_id(model_name)
        version_uuid = _fake_id(f"{model_name}:{version}")
        ready = (model_name, version) not in self.not_ready
        return RegistryModelVersion(
            name=model_name,
            version=version,
            model_id=model_id,
            version_uuid=version_uuid,
            # Mirrors the live registry: a prefix, with no object name. The
            # artifact service must resolve it, so the fake keeps that honest.
            artifact_uri=f"s3a://fake-lighthouse-bucket/data/modelregistry/{model_id}/{version_uuid}",
            status="READY" if ready else "PENDING",
            format=ArtifactFormat.ONNX,
            packaging=Packaging.MLFLOW_TAR_GZ,
            created_at=_EPOCH + timedelta(days=int(version) if version.isdigit() else 0),
            entrypoint="model.onnx",
            size_bytes=None,  # unknown until read, as with the real registry
        )


def _fake_id(seed: str) -> str:
    """A stable identifier shaped like a registry id (`abcd-efgh-ijkl-mnop`)."""
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return "-".join(h[i : i + 4] for i in range(0, 16, 4))
