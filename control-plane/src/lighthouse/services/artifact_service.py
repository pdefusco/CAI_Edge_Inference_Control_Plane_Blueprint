"""Artifact materialization, caching and serving.

This exists because of one hard constraint discovered against the live cluster:
**the device can never read the registry's artifact URI itself.** Registry
metadata is an HTTPS API reachable with a workload JWT, but the artifact bytes sit
in object storage that a Jetson at home has no identity for. So the control plane
-- which runs inside CAI, where credentials are vended -- has to broker the bytes.

Given that, hashing is nearly free. Spec SS17 requires SHA-256 verification before
activation, over exactly the bytes the device receives; only this process can
compute that. Since it must read every byte anyway, it writes them to a local
cache in the same pass and serves subsequent devices from disk. The "a proxy
wastes bandwidth" and "hashing costs a download" objections collapse into a single
cost paid once per model version.

Three details that are easy to get wrong:

  * **One pass, bounded memory.** The hash is computed from the same chunks that
    get written. Reading the artifact twice (once to hash, once to store) would
    double the transfer; buffering it whole would put a multi-GB object in RAM
    inside a memory-capped Application container.
  * **Atomic publish.** Bytes land in a `.tmp` and are `os.replace()`d into place
    only after the digest is known. A reader can therefore never observe a
    partially written cache file, which would hand a device bytes that fail
    verification and look like corruption.
  * **Eviction respects references.** An entry a live desired deployment points at
    is never evicted, or a device returning after a long absence would be told to
    fetch something just deleted.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import tarfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml
from lighthouse_contracts import Packaging

from ..config import Settings
from ..repositories import ArtifactCacheRow, Store
from ..registry import (
    ArtifactUnavailable,
    ModelRegistry,
    RegistryError,
    RegistryModelVersion,
)
from ..util import now_utc

log = logging.getLogger(__name__)


class ArtifactNotReady(RuntimeError):
    """The artifact is still being materialized.

    Distinct from a failure: the caller should tell the device to wait and
    re-poll, never to give up.
    """


class ArtifactFailed(RuntimeError):
    """Materialization failed permanently for this attempt."""


@dataclass(frozen=True, slots=True)
class ArtifactInfo:
    """A ready artifact, as the rest of the control plane sees it."""

    cache_key: str
    sha256: str
    size_bytes: int
    packaging: Packaging
    entrypoint: str | None
    path: Path


class ArtifactService:
    """Owns the artifact cache directory and the cache table."""

    def __init__(
        self,
        store: Store,
        registry: ModelRegistry,
        settings: Settings,
    ) -> None:
        self._store = store
        self._registry = registry
        self._settings = settings
        self._dir = settings.artifact_cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        # Guards eviction so two concurrent materializations cannot both decide to
        # delete the same entry.
        self._evict_lock = threading.Lock()
        self._threads: set[threading.Thread] = set()
        self._threads_lock = threading.Lock()
        self._closing = False

    # -- public API ------------------------------------------------------

    def get_ready(self, cache_key: str) -> ArtifactInfo | None:
        """Return the artifact if it is materialized and its file still exists.

        The on-disk check is not paranoia: the cache directory is a filesystem
        someone can clear, and a READY row pointing at a missing file would
        otherwise make the control plane advertise a digest it cannot serve.
        """
        row = self._store.get_artifact(cache_key)
        if row is None or not row.is_ready or not row.cache_path:
            return None
        path = Path(row.cache_path)
        if not path.exists():
            log.warning("cache row %s is READY but %s is missing; re-materializing", cache_key, path)
            self._store.delete_artifact(cache_key)
            return None
        return ArtifactInfo(
            cache_key=cache_key,
            sha256=row.sha256,  # type: ignore[arg-type]
            size_bytes=row.size_bytes or path.stat().st_size,
            packaging=Packaging(row.packaging or Packaging.MLFLOW_TAR_GZ),
            entrypoint=row.entrypoint,
            path=path,
        )

    def request(self, mv: RegistryModelVersion, *, background: bool = True) -> ArtifactInfo | None:
        """Ensure an artifact is (being) materialized.

        Returns ArtifactInfo on a cache hit, or None if materialization is now in
        flight -- in which case the caller reports `artifact_ready: false` and the
        device waits. Called from PUT /deployment, never from a device poll: a
        poll must stay cheap, and the operator action is the natural moment to
        pay the cost.
        """
        ready = self.get_ready(mv.cache_key)
        if ready is not None:
            self._store.touch_artifact(mv.cache_key)
            return ready

        if self._closing:
            # Shutting down. Starting work now would either be abandoned mid-write
            # or outlive the store it writes to; the caller reports
            # `artifact_ready: false` and the device re-polls the next process.
            return None

        existing = self._store.get_artifact(mv.cache_key)
        if existing is not None and existing.status == "PENDING":
            # Someone else is already on it.
            return None

        if existing is not None and existing.status == "FAILED":
            # Allow a retry: a transient registry outage should not poison a
            # version forever.
            self._store.delete_artifact(mv.cache_key)

        claimed = self._store.claim_artifact(
            ArtifactCacheRow(
                cache_key=mv.cache_key,
                model_name=mv.name,
                model_version=mv.version,
                model_id=mv.model_id,
                version_uuid=mv.version_uuid,
                packaging=mv.packaging.value,
                source_uri=mv.artifact_uri,
                created_at=now_utc(),
            )
        )
        if not claimed:
            # Lost the race; the winner is materializing.
            return None

        if background:
            thread = threading.Thread(
                target=self._materialize_guarded,
                args=(mv,),
                name=f"materialize-{mv.name}-{mv.version}",
                daemon=True,
            )
            with self._threads_lock:
                self._threads.add(thread)
            thread.start()
            return None

        return self._materialize(mv)

    def materialize_now(self, mv: RegistryModelVersion) -> ArtifactInfo:
        """Synchronous materialization, for tests and CLI use.

        Raises ArtifactFailed if the bytes could not be fetched.
        """
        ready = self.get_ready(mv.cache_key)
        if ready is not None:
            return ready
        existing = self._store.get_artifact(mv.cache_key)
        if existing is not None:
            self._store.delete_artifact(mv.cache_key)
        self._store.claim_artifact(
            ArtifactCacheRow(
                cache_key=mv.cache_key,
                model_name=mv.name,
                model_version=mv.version,
                model_id=mv.model_id,
                version_uuid=mv.version_uuid,
                packaging=mv.packaging.value,
                source_uri=mv.artifact_uri,
                created_at=now_utc(),
            )
        )
        return self._materialize(mv)

    def wait_until_ready(self, cache_key: str, timeout: float = 30.0) -> ArtifactInfo | None:
        """Block until an in-flight materialization finishes. Tests only.

        Request handlers must never call this -- a device that polls while an
        artifact is downloading gets `artifact_ready: false` and comes back,
        rather than holding a worker thread.
        """
        deadline = now_utc().timestamp() + timeout
        while now_utc().timestamp() < deadline:
            ready = self.get_ready(cache_key)
            if ready is not None:
                return ready
            row = self._store.get_artifact(cache_key)
            if row is not None and row.status == "FAILED":
                return None
            threading.Event().wait(0.05)
        return None

    # -- serving ---------------------------------------------------------

    def iter_range(
        self, info: ArtifactInfo, start: int = 0, end: int | None = None
    ) -> Iterator[bytes]:
        """Yield bytes [start, end] inclusive, as HTTP Range semantics require.

        Resumable download off a local file is one `seek`. This is the other
        reason the proxy design wins: CML's ingress behaviour on long transfers is
        undocumented, so a cut connection has to be cheap to resume, and
        re-ranging into object storage on every retry would not be.
        """
        last = (info.size_bytes - 1) if end is None else end
        remaining = last - start + 1
        if remaining <= 0:
            return
        chunk_size = self._settings.artifact_chunk_size
        corrupt = self._settings.dev_corrupt_artifacts
        position = start
        with open(info.path, "rb") as fh:
            fh.seek(start)
            while remaining > 0:
                chunk = fh.read(min(chunk_size, remaining))
                if not chunk:
                    return
                if corrupt and position == 0 and chunk:
                    # Local-only test hook: flip one byte so the device's
                    # verify-before-activate step must reject the artifact.
                    chunk = bytes([chunk[0] ^ 0xFF]) + chunk[1:]
                position += len(chunk)
                remaining -= len(chunk)
                yield chunk

    # -- internals -------------------------------------------------------

    def _materialize_guarded(self, mv: RegistryModelVersion) -> None:
        try:
            self._materialize(mv)
        except Exception:  # pragma: no cover - thread boundary
            log.exception("materialization thread failed for %s", mv.cache_key)
        finally:
            with self._threads_lock:
                self._threads.discard(threading.current_thread())

    def _materialize(self, mv: RegistryModelVersion) -> ArtifactInfo:
        """Stream, hash and publish one artifact."""
        final = self._dir / _cache_filename(mv)
        # Include the pid so two processes sharing a data dir cannot collide on
        # the temp name and corrupt each other's download.
        tmp = final.with_suffix(final.suffix + f".{os.getpid()}.tmp")
        digest = hashlib.sha256()
        total = 0

        try:
            with self._registry.open_artifact(mv) as stream:
                tmp.parent.mkdir(parents=True, exist_ok=True)
                with open(tmp, "wb") as out:
                    # One pass: the same chunk feeds the hash and the file. Never
                    # stream.fileobj.read() with no size -- artifacts are large.
                    for chunk in stream.chunks(self._settings.artifact_chunk_size):
                        digest.update(chunk)
                        out.write(chunk)
                        total += len(chunk)
                    out.flush()
                    # fsync before the rename: without it a crash can leave the
                    # directory entry pointing at unwritten data, i.e. a READY
                    # row whose file fails its own checksum.
                    os.fsync(out.fileno())
                packaging = stream.packaging

            if total == 0:
                raise ArtifactUnavailable(f"registry returned no bytes for {mv.cache_key}")

            sha256 = digest.hexdigest()
            entrypoint = _read_entrypoint(tmp, packaging) or mv.entrypoint
            os.replace(tmp, final)

            row = ArtifactCacheRow(
                cache_key=mv.cache_key,
                model_name=mv.name,
                model_version=mv.version,
                model_id=mv.model_id,
                version_uuid=mv.version_uuid,
                status="READY",
                sha256=sha256,
                size_bytes=total,
                packaging=packaging.value,
                entrypoint=entrypoint,
                cache_path=str(final),
                source_uri=mv.artifact_uri,
                completed_at=now_utc(),
                last_access=now_utc(),
            )
            self._store.update_artifact(row)
            log.info(
                "materialized %s v%s (%s bytes, sha256=%s)", mv.name, mv.version, total, sha256[:12]
            )
            self._evict_if_needed()
            return ArtifactInfo(
                cache_key=mv.cache_key,
                sha256=sha256,
                size_bytes=total,
                packaging=packaging,
                entrypoint=entrypoint,
                path=final,
            )

        except (RegistryError, OSError) as exc:
            tmp.unlink(missing_ok=True)
            self._store.update_artifact(
                ArtifactCacheRow(
                    cache_key=mv.cache_key,
                    model_name=mv.name,
                    model_version=mv.version,
                    model_id=mv.model_id,
                    version_uuid=mv.version_uuid,
                    status="FAILED",
                    error=f"{type(exc).__name__}: {exc}",
                    source_uri=mv.artifact_uri,
                    completed_at=now_utc(),
                )
            )
            log.error("materialization failed for %s: %s", mv.cache_key, exc)
            raise ArtifactFailed(str(exc)) from exc

    def _evict_if_needed(self) -> None:
        """Drop least-recently-used artifacts until under the byte cap."""
        cap = self._settings.artifact_cache_max_bytes
        if cap <= 0:
            return
        with self._evict_lock:
            rows = self._store.list_artifacts()  # oldest access first
            total = sum(r.size_bytes or 0 for r in rows if r.status == "READY")
            if total <= cap:
                return
            protected = self._store.referenced_cache_keys()
            for row in rows:
                if total <= cap:
                    break
                if row.status != "READY" or row.cache_key in protected:
                    continue
                if row.cache_path:
                    Path(row.cache_path).unlink(missing_ok=True)
                self._store.delete_artifact(row.cache_key)
                total -= row.size_bytes or 0
                log.info("evicted cached artifact %s", row.cache_key)

    def remove(self, cache_key: str) -> None:
        """Delete one cache entry and its file, regardless of references.

        Only for operator-driven cleanup; eviction uses the reference-respecting
        path above.
        """
        row = self._store.get_artifact(cache_key)
        if row is None:
            return
        if row.cache_path:
            Path(row.cache_path).unlink(missing_ok=True)
        self._store.delete_artifact(cache_key)

    def close(self, timeout: float = 10.0) -> None:
        """Stop accepting new work and wait for in-flight materializations.

        This has to happen before the store closes, and the reason is sharper than
        tidiness. A materialization thread ends by writing its cache row and then
        evicting -- both inside the SQLite connection. Closing that connection
        underneath a thread that is in it is not an exception that gets logged, it
        is a **segfault**: the C handle is freed while another thread holds it. The
        process dies with no traceback and no audit event, mid-download.

        Not hypothetical, and not only a test artifact: a CAI Application restart
        while an operator's `PUT /deployment` is still fetching bytes takes exactly
        this path.

        Bounded, because the opposite failure is worse. A thread wedged in a
        registry read that never returns must not stop the process from exiting, so
        after the timeout this gives up and says so; `daemon=True` then lets the
        interpreter abandon it. That reintroduces the race in the one case where
        there is no better option, which is why the warning names the threads.
        """
        self._closing = True
        deadline = time.monotonic() + timeout
        while True:
            with self._threads_lock:
                alive = [t for t in self._threads if t.is_alive()]
            if not alive:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                log.warning(
                    "artifact materialization still in flight at shutdown: %s",
                    ", ".join(t.name for t in alive),
                )
                return
            alive[0].join(remaining)


def _cache_filename(mv: RegistryModelVersion) -> str:
    """Flat, collision-free filename derived from lineage ids.

    The ids come from the registry, so they are sanitized rather than trusted:
    anything outside the known-safe set is replaced, which keeps a hostile or
    merely odd id from escaping the cache directory via `../`.
    """
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in f"{mv.model_id}_{mv.version_uuid}")
    return f"{safe}.tar.gz"


def _read_entrypoint(path: Path, packaging: Packaging) -> str | None:
    """Pull the ONNX file path out of the artifact's MLmodel descriptor.

    Knowing this server-side means the device does not have to guess which file
    in the tarball to load. Best-effort: a malformed or absent MLmodel is not a
    reason to fail an otherwise good artifact, so the agent falls back to its own
    discovery.
    """
    if packaging is not Packaging.MLFLOW_TAR_GZ:
        return None
    try:
        with tarfile.open(path, mode="r:gz") as tar:
            member = next(
                (m for m in tar.getmembers() if Path(m.name).name == "MLmodel"), None
            )
            if member is None:
                return None
            extracted = tar.extractfile(member)
            if extracted is None:
                return None
            doc = yaml.safe_load(io.BytesIO(extracted.read()))
        if not isinstance(doc, dict):
            return None
        data = doc.get("flavors", {}).get("onnx", {}).get("data")
        if not isinstance(data, str):
            return None
        # MLmodel paths are relative to the artifact root; keep the prefix the
        # MLmodel sits under so the agent can resolve it after unpacking.
        prefix = str(Path(member.name).parent)
        if prefix not in {"", "."}:
            return f"{prefix}/{data}"
        return data
    except (tarfile.TarError, yaml.YAMLError, OSError) as exc:
        log.warning("could not read MLmodel from %s: %s", path, exc)
        return None
