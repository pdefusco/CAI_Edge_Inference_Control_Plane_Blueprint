"""Test harness for the agent.

Deliberately free of any `lighthouse` import. The agent must be installable and
testable on a Jetson that has never heard of the control plane package, and a test
suite that quietly depends on the server would hide a real layering mistake. The
full two-sided loop is tested in `control-plane/tests/test_end_to_end.py`, which is
allowed to import both.

`StubClient` is a fake *transport*, not a fake reconciler input: it serves genuine
gzipped tar bytes with a genuine SHA-256, so the download, verify, unpack and
activate path under test is the same code that runs against CAI.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import tarfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pytest
from lighthouse_contracts import (
    DesiredState,
    DesiredStateResponse,
    HeartbeatRequest,
    HeartbeatResponse,
    ModelRef,
    Packaging,
)

from keeper.artifact_manager import ArtifactManager
from keeper.client import (
    ArtifactForbiddenError,
    ArtifactNotReadyError,
    GenerationStaleError,
    TransientError,
)
from keeper.config import AgentSettings
from keeper.reconciler import Reconciler
from keeper.runtime.mock import MockRuntime
from keeper.state import StateStore

DEVICE_ID = "test-device-01"


def make_archive(model_name: str, version: str, *, size: int = 4096) -> bytes:
    """A reproducible gzipped MLflow-layout tarball.

    Byte-for-byte stable across runs and machines: the gzip header's timestamp is
    pinned to 0 and every tar member's mtime/uid/gid/mode is fixed. Without that,
    the digest changes per run and every checksum assertion in this file becomes
    intermittent -- which is a genuinely miserable class of test failure.
    """
    payload = bytearray()
    counter = 0
    seed = f"{model_name}:{version}".encode()
    while len(payload) < size:
        payload += hashlib.sha256(seed + counter.to_bytes(8, "big")).digest()
        counter += 1
    onnx_bytes = bytes(payload[:size])

    mlmodel = (
        "artifact_path: model\n"
        "flavors:\n"
        "  onnx:\n"
        "    data: model.onnx\n"
        "    onnx_version: 1.15.0\n"
        f"model_uuid: {model_name}-{version}\n"
    ).encode()

    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for name, data in (("MLmodel", mlmodel), ("model.onnx", onnx_bytes)):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(data))
    return raw.getvalue()


@dataclass
class FakeResponse:
    """Just enough of `httpx.Response` for the artifact manager."""

    status_code: int
    body: bytes
    force_chunk: int | None = None

    def iter_bytes(self, chunk_size: int = 65536):
        # `force_chunk` lets a test make the transfer arrive in several pieces.
        # Without it the artifact manager's 1 MiB read size swallows a small test
        # archive whole, and any assertion about *partial* transfer state is
        # really an assertion about zero bytes having been written.
        size = self.force_chunk or chunk_size
        for start in range(0, len(self.body), size):
            yield self.body[start : start + size]


@dataclass
class StubClient:
    """Stands in for `ControlPlaneClient`.

    Fault injection is by attribute rather than by subclass so a test can flip a
    failure on mid-scenario, which is how the mid-download supersede case is
    arranged deterministically instead of by racing a real poll.
    """

    archives: dict[str, bytes] = field(default_factory=dict)
    desired: DesiredStateResponse | None = None
    heartbeats: list[HeartbeatRequest] = field(default_factory=list)
    server_generation: int = 0

    # Fault injection.
    corrupt_bytes: bool = False
    raise_transient: bool = False
    raise_not_ready: bool = False
    raise_forbidden: bool = False
    raise_stale: bool = False
    truncate_after: int | None = None
    chunk_size: int | None = None

    download_count: int = 0

    def fetch_desired_state(self) -> DesiredStateResponse:
        if self.desired is None:
            raise AssertionError("test did not set a desired state")
        return self.desired

    def send_heartbeat(self, heartbeat: HeartbeatRequest) -> HeartbeatResponse:
        self.heartbeats.append(heartbeat)
        return HeartbeatResponse(
            accepted=True,
            generation=max(self.server_generation, heartbeat.observed_generation),
            desired_state=self.desired.desired_state if self.desired else None,
            server_time=datetime.now(timezone.utc),
        )

    @contextmanager
    def stream_artifact(self, artifact_uri, *, offset=0, if_match=None):
        self.download_count += 1
        if self.raise_transient:
            raise TransientError("stub: network down")
        if self.raise_not_ready:
            raise ArtifactNotReadyError("stub: still materializing")
        if self.raise_forbidden:
            raise ArtifactForbiddenError("stub: revoked")
        if self.raise_stale:
            raise GenerationStaleError("stub: superseded")

        body = self.archives[artifact_uri]
        if self.corrupt_bytes:
            flipped = bytearray(body)
            flipped[0] ^= 0xFF
            body = bytes(flipped)
        if self.truncate_after is not None:
            body = body[: self.truncate_after]
        status = 206 if offset else 200
        yield FakeResponse(
            status_code=status, body=body[offset:], force_chunk=self.chunk_size
        )


@dataclass
class Harness:
    settings: AgentSettings
    client: StubClient
    runtime: MockRuntime
    reconciler: Reconciler
    artifacts: ArtifactManager

    def model(self, name: str = "fashion-cnn", version: str = "1", *, size: int = 4096) -> ModelRef:
        """Register an artifact with the stub and return the matching `ModelRef`."""
        uri = f"/api/v1/devices/{DEVICE_ID}/artifact?model={name}&version={version}"
        body = make_archive(name, version, size=size)
        self.client.archives[uri] = body
        return ModelRef(
            name=name,
            version=version,
            sha256=hashlib.sha256(body).hexdigest(),
            artifact_uri=uri,
            packaging=Packaging.MLFLOW_TAR_GZ,
            entrypoint="model.onnx",
            size_bytes=len(body),
        )

    def desire(
        self,
        generation: int,
        desired_state: DesiredState,
        model: ModelRef | None = None,
        *,
        artifact_ready: bool = True,
    ) -> DesiredStateResponse:
        desired = DesiredStateResponse(
            device_id=DEVICE_ID,
            generation=generation,
            desired_state=desired_state,
            model=model,
            artifact_ready=artifact_ready,
            server_time=datetime.now(timezone.utc),
        )
        self.client.desired = desired
        self.client.server_generation = generation
        return desired

    def reconcile(self, desired: DesiredStateResponse):
        return self.reconciler.reconcile(desired)

    @property
    def state(self):
        return self.reconciler.state

    def model_dir(self, name: str, version: str) -> Path:
        return self.settings.model_dir / name / version


@pytest.fixture
def clock():
    """A monotonic clock the test drives by hand, so backoff assertions do not
    sleep. `advance` is the only way time moves."""

    class Clock:
        def __init__(self) -> None:
            self.now = 1000.0

        def __call__(self) -> float:
            return self.now

        def advance(self, seconds: float) -> None:
            self.now += seconds

    return Clock()


@pytest.fixture
def harness(tmp_path, clock) -> Harness:
    settings = AgentSettings(
        device_id=DEVICE_ID,
        control_plane_url="http://control-plane.invalid",
        token="lhd_0123456789abcdef.secret",
        data_dir=tmp_path / "keeper",
        poll_interval_seconds=1,
        retry_backoff_initial_seconds=10,
        retry_backoff_max_seconds=100,
    )
    client = StubClient()
    runtime = MockRuntime()
    artifacts = ArtifactManager(settings, client)  # type: ignore[arg-type]
    reconciler = Reconciler(
        settings,
        client,  # type: ignore[arg-type]
        artifacts,
        runtime,
        StateStore(settings.state_path),
        clock=clock,
    )
    return Harness(
        settings=settings,
        client=client,
        runtime=runtime,
        reconciler=reconciler,
        artifacts=artifacts,
    )
