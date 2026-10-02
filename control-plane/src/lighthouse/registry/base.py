"""The ModelRegistry seam (spec SS11 / SS18).

Everything the control plane knows about Cloudera AI enters through this
protocol. The point is that `FakeModelRegistry` and `CAIModelRegistry` are
substitutable: nothing above this layer may import boto3, httpx or mlflow.

One deliberate departure from the spec's sketch: `get_artifact_uri` is demoted to
a display/audit accessor and `open_artifact()` is the real method. The control
plane needs artifact *bytes* -- it has to hash them to satisfy SS17's
verify-before-activate requirement, and it has to serve them to a device that has
no object-store identity. Handing an `s3a://` URI up to the HTTP layer would just
relocate boto3 into the API module, which is exactly the leak this seam exists to
prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import IO, Iterator, Protocol, runtime_checkable

from lighthouse_contracts import ArtifactFormat, Packaging


class RegistryError(Exception):
    """Base for every registry failure.

    Implementations normalize their transport exceptions into this hierarchy so
    the API layer can map failures to status codes without importing botocore or
    httpx exception types.
    """


class ModelNotFound(RegistryError):
    """No model, or no such version of it."""


class VersionNotReady(RegistryError):
    """The version exists but the registry has not finished building it.

    Raised rather than returned so a half-written version can never become
    desired state -- a device would download a truncated artifact and fail its
    checksum, which looks like corruption rather than a race.

    Means *retry later*. For a version that will never become ready, see
    `VersionFailed`.
    """


class VersionFailed(RegistryError):
    """The registry tried to build this version and gave up.

    Separate from `VersionNotReady` because the two need opposite advice. The
    real registry's status is not a ready/not-ready pair: `UPLOAD_FAILED` and
    `DELETE_FAILED` are terminal, and telling an operator to retry a terminal
    state sends them back to a dashboard that will never change. The remedy is to
    register a new version.
    """


class ArtifactUnavailable(RegistryError):
    """Metadata resolved but the bytes could not be read."""


class RegistryUnavailable(RegistryError):
    """The registry could not be reached at all."""


class RegistryAuthError(RegistryError):
    """Credentials were rejected. Distinct from unavailable because the remedy is
    different: refresh a token, not retry."""


class UnsupportedFlavor(RegistryError):
    """The version carries no artifact in a format the edge can run.

    Surfaced at deployment-request time so the operator gets a clear rejection,
    instead of the device discovering it after a download.
    """


@dataclass(frozen=True, slots=True)
class RegistryModelVersion:
    """One version of a registered model, as the control plane needs it.

    `model_id` and `version_uuid` together identify *one specific set of bytes*.
    They key the artifact cache, because they are immutable -- unlike the
    `(name, version)` label, which an operator can repoint at different bytes.

    `version_uuid` is not necessarily a UUID, despite the name. It is whatever
    string the adapter can offer that changes when the bytes change. The real CAI
    registry has no such field: a version there is just an integer, and integers
    can be deleted and reissued. So `CAIModelRegistry` composes the version with
    its creation timestamp, which is what makes a reissued version number read as
    different bytes rather than silently reusing the cached ones. The fake, which
    does have stable synthetic ids, uses those. Renaming the field to match that
    looser meaning is a pending follow-up; it touches two SQLite columns, three
    independent reconstructions of `cache_key`, and ~30 test references.
    """

    name: str
    version: str
    model_id: str
    version_uuid: str
    artifact_uri: str
    status: str = "READY"
    format: ArtifactFormat = ArtifactFormat.ONNX
    packaging: Packaging = Packaging.MLFLOW_TAR_GZ
    created_at: datetime | None = None
    entrypoint: str | None = None
    size_bytes: int | None = None
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def cache_key(self) -> str:
        """Content-lineage identity, not label identity."""
        return f"{self.model_id}/{self.version_uuid}"


@dataclass(slots=True)
class ArtifactStream:
    """An open, readable artifact plus whatever the registry knew about it.

    `size_bytes` may be None: the live registry's artifact_uri is a prefix, and
    resolving it to a concrete object does not always yield a length up front.
    Callers must not depend on it -- the authoritative size is what was actually
    read and hashed.
    """

    fileobj: IO[bytes]
    packaging: Packaging
    size_bytes: int | None = None
    source_uri: str | None = None
    etag: str | None = None

    def chunks(self, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        """Iterate the stream in bounded chunks.

        Deliberately the only read path offered: artifacts are multi-megabyte and
        may grow, so nothing in the ingest path should ever call `.read()` with
        no argument.
        """
        while True:
            chunk = self.fileobj.read(chunk_size)
            if not chunk:
                return
            yield chunk

    def close(self) -> None:
        try:
            self.fileobj.close()
        except Exception:  # pragma: no cover - best-effort cleanup
            pass

    def __enter__(self) -> ArtifactStream:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


@runtime_checkable
class ModelRegistry(Protocol):
    """The abstraction the rest of the control plane codes against (spec SS11)."""

    @property
    def name(self) -> str:
        """Short identifier for health output and logs, e.g. "fake" or "cai"."""
        ...

    def list_models(self) -> list[str]:
        """Registered model names."""
        ...

    def list_versions(self, model_name: str) -> list[RegistryModelVersion]:
        """All versions of a model, oldest first.

        Oldest first is load-bearing: `ModelCatalog` reverses this to present
        newest-first, and nothing anywhere sorts versions, so an implementation
        must impose the order rather than pass through whatever the registry
        returned.

        Raises ModelNotFound if the model does not exist. A version that is
        unbuilt, failed or unrunnable is *reported* here via `status` and
        `format`, never raised -- an operator needs to see the broken version in
        the catalog. Only `get_version` refuses.
        """
        ...

    def get_version(self, model_name: str, version: str) -> RegistryModelVersion:
        """Resolve one version.

        Raises ModelNotFound if absent, VersionNotReady if it is still being
        built, VersionFailed if the registry gave up building it, and
        UnsupportedFlavor if it carries nothing the edge can execute.

        This is the deployment-time gate, so implementations must not serve it
        from a cache: a stale READY here puts bytes into desired state that the
        registry may already have deleted.
        """
        ...

    def open_artifact(self, mv: RegistryModelVersion) -> ArtifactStream:
        """Open the artifact bytes for reading.

        The caller streams and hashes these; it must close the stream.
        """
        ...

    def get_artifact_uri(self, mv: RegistryModelVersion) -> str:
        """The registry-side location, for operator display and audit only.

        Never given to a device: it points into object storage the device has no
        credentials for.
        """
        ...

    def ping(self) -> bool:
        """Cheap reachability check for the health endpoint."""
        ...
