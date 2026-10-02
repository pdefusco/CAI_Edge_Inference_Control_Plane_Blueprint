"""Download, verify, unpack and delete model artifacts on the device.

Three rules govern this file, and all three come from the spec:

1. **Verify before activate.** The digest is checked over the assembled file
   before anything is unpacked. A mismatch deletes the bytes and raises; it never
   falls through to "probably fine".
2. **Resume, don't restart.** A home uplink drops. A partial download is kept as
   `<name>.part` with a `.meta` sidecar, and resumed with `Range` + `If-Match` so
   the server rejects a resume against changed bytes rather than letting us splice
   two artifacts together.
3. **Revoke means gone.** Removal deletes the unpacked model, the downloaded
   archive, any partial, and any derived runtime artifacts (TensorRT engines in
   M6) -- not just the ONNX file.

Unpacking a tarball from the network is the one genuinely dangerous operation
here, so `_safe_extract` rejects absolute paths, `..` traversal, symlinks and
hard links before any member is written.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path

from lighthouse_contracts import ModelRef, Packaging

from .client import ControlPlaneClient, GenerationStaleError
from .config import AgentSettings

log = logging.getLogger(__name__)

_CHUNK = 1024 * 1024


class ArtifactError(RuntimeError):
    """Download or unpack failed in a way the operator needs to see."""


class ChecksumMismatch(ArtifactError):
    """The assembled bytes do not match the digest the control plane published.

    This is the one failure that must *never* degrade into a deployment. The spec
    is explicit: a failed checksum prevents deployment.
    """


class DownloadAborted(ArtifactError):
    """A newer generation arrived mid-download, so this one was abandoned.

    Not a failure: the reconciler catches it, leaves state alone, and picks up the
    current instruction on the next pass.
    """


@dataclass(frozen=True, slots=True)
class LocalModel:
    """A verified, unpacked model ready for a runtime to load."""

    name: str
    version: str
    sha256: str
    path: Path
    entrypoint: Path | None

    @property
    def load_target(self) -> Path:
        """What to hand a runtime: the ONNX file if we know it, else the dir."""
        return self.entrypoint or self.path


class ArtifactManager:
    def __init__(self, settings: AgentSettings, client: ControlPlaneClient) -> None:
        self._settings = settings
        self._client = client
        self._artifacts = settings.artifact_dir
        self._models = settings.model_dir

    # -- public API --------------------------------------------------------

    def ensure(self, model: ModelRef, *, should_abort=None) -> LocalModel:
        """Make `model` present and verified locally, downloading if needed.

        `should_abort` is polled between chunks and lets the reconciler abandon a
        long download the moment a newer generation arrives -- otherwise a device
        on a slow link would finish fetching a version the operator already
        replaced, then deploy it.
        """
        existing = self._verified_model_dir(model)
        if existing is not None:
            log.debug("artifact %s/%s already unpacked at %s", model.name, model.version, existing.path)
            return existing

        archive = self._download(model, should_abort=should_abort)
        return self._activate(model, archive)

    def remove(self, name: str, version: str) -> None:
        """Delete everything derived from this model version.

        Spec SS5: revocation removes the model artifacts *and* generated runtime
        artifacts. Deliberately tolerant of missing paths -- revoke must succeed
        on a device that never finished downloading, or the device would be stuck
        reporting REVOKE_PENDING forever.
        """
        model_dir = self._model_dir(name, version)
        for path in (model_dir, self._engines_dir(name, version)):
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
                log.info("removed %s", path)
        for leftover in self._archive_candidates(name, version):
            if leftover.exists():
                leftover.unlink(missing_ok=True)
                log.info("removed %s", leftover)

    def prune(self, name: str, keep_versions: tuple[str, ...], *, keep: int = 2) -> None:
        """Delete old versions of `name`, retaining `keep` most recently used.

        Without this, every upgrade leaves the previous version's archive and
        unpacked tree behind forever -- fine on a laptop, a slow disk leak on a
        device with 32GB of eMMC.

        `keep` is 2 rather than 1 on purpose: the most likely next operator action
        after a bad upgrade is a rollback, and keeping one generation back makes
        that instant instead of a cold re-download over a home uplink.
        """
        model_root = self._models / _slug(name)
        if not model_root.is_dir():
            return
        protected = {_slug(v) for v in keep_versions}
        candidates = [
            d for d in model_root.iterdir() if d.is_dir() and d.name not in protected
        ]
        # Most recently touched first; activation rewrites the directory, so mtime
        # tracks "when this version was last deployed" closely enough.
        candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        for stale in candidates[max(0, keep - len(protected)) :]:
            log.info("pruning superseded model %s/%s", name, stale.name)
            self.remove(name, stale.name)

    def remove_all(self) -> None:
        """Delete every model artifact on the device. Used by revoke.

        The engines root is included, and that inclusion is load-bearing: `remove()`
        clears the engines of one version, so a cleanup that skipped the root here
        would leave a compiled TensorRT engine -- a *runnable* derivative of the
        model -- on a device whose authorization was just withdrawn. Rule 3 at the
        top of this file would be false.
        """
        for root in (self._models, self._artifacts, self._settings.data_dir / "engines"):
            if root.exists():
                shutil.rmtree(root, ignore_errors=True)
                log.info("removed %s", root)

    # -- download ----------------------------------------------------------

    def _download(self, model: ModelRef, *, should_abort=None) -> Path:
        self._artifacts.mkdir(parents=True, exist_ok=True)
        target = self._archive_path(model)
        if target.is_file() and _digest_of(target) == model.sha256:
            log.info("archive for %s/%s already downloaded and verified", model.name, model.version)
            return target

        part = target.with_suffix(target.suffix + ".part")
        meta_path = part.with_suffix(".meta")
        offset = self._resume_offset(part, meta_path, model)

        if offset:
            log.info(
                "resuming %s/%s at byte %d of %s",
                model.name,
                model.version,
                offset,
                model.size_bytes if model.size_bytes is not None else "unknown",
            )

        digest = hashlib.sha256()
        if offset:
            # Re-hash what we already have. Cheaper than re-downloading it, and it
            # means the final digest covers the resumed file end to end rather
            # than only the newly-fetched tail.
            with open(part, "rb") as fh:
                for chunk in iter(lambda: fh.read(_CHUNK), b""):
                    digest.update(chunk)

        mode = "ab" if offset else "wb"
        try:
            with self._client.stream_artifact(
                model.artifact_uri, offset=offset, if_match=model.sha256 if offset else None
            ) as response:
                if offset and response.status_code != 206:
                    # Server ignored the Range and is sending the whole file.
                    # Start over rather than appending a full body to a partial.
                    log.warning("server ignored Range; restarting download from zero")
                    part.unlink(missing_ok=True)
                    meta_path.unlink(missing_ok=True)
                    offset = 0
                    digest = hashlib.sha256()
                    mode = "wb"

                self._write_meta(meta_path, model)
                written = offset
                with open(part, mode) as fh:
                    for chunk in response.iter_bytes(_CHUNK):
                        if should_abort is not None and should_abort():
                            fh.flush()
                            os.fsync(fh.fileno())
                            raise DownloadAborted(
                                f"download of {model.name}/{model.version} abandoned at {written} bytes"
                            )
                        fh.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                    fh.flush()
                    os.fsync(fh.fileno())
        except DownloadAborted:
            # The .part and .meta stay on disk on purpose: if this same version
            # comes back around, the next attempt resumes instead of restarting.
            raise
        except GenerationStaleError:
            # 412 on an `If-Match` resume means the server's bytes are not the ones
            # our partial came from, so the partial is worthless -- keeping it would
            # make every subsequent resume fail the same way. 409/416 mean the
            # generation is gone entirely. Either way, discard and let the next
            # pass start clean.
            part.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            raise

        actual = digest.hexdigest()
        if actual != model.sha256:
            # Delete rather than keep: a mismatch means we cannot tell which bytes
            # are wrong, so resuming from this file would never converge.
            part.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            raise ChecksumMismatch(
                f"{model.name}/{model.version}: expected sha256 {model.sha256}, got {actual}"
            )

        if model.size_bytes is not None and written != model.size_bytes:
            part.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            raise ArtifactError(
                f"{model.name}/{model.version}: expected {model.size_bytes} bytes, got {written}"
            )

        os.replace(part, target)
        meta_path.unlink(missing_ok=True)
        log.info("downloaded and verified %s/%s (%d bytes)", model.name, model.version, written)
        return target

    def _resume_offset(self, part: Path, meta_path: Path, model: ModelRef) -> int:
        """How many bytes of `part` are safe to keep.

        Zero unless the sidecar proves the partial belongs to *these* exact bytes.
        Without the sidecar check, a partial from an earlier version of the same
        label would be silently resumed into a corrupt archive.
        """
        if not part.is_file():
            meta_path.unlink(missing_ok=True)
            return 0
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            log.info("partial download has no usable sidecar; restarting")
            part.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            return 0
        if meta.get("sha256") != model.sha256:
            log.info("partial download is for a different digest; discarding")
            part.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            return 0
        size = part.stat().st_size
        if model.size_bytes is not None and size >= model.size_bytes:
            # Complete-looking but never verified (killed between write and
            # rename). Re-verify from scratch instead of trusting the length.
            part.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            return 0
        return size

    def _write_meta(self, meta_path: Path, model: ModelRef) -> None:
        meta_path.write_text(
            json.dumps(
                {
                    "name": model.name,
                    "version": model.version,
                    "sha256": model.sha256,
                    "size_bytes": model.size_bytes,
                    "artifact_uri": model.artifact_uri,
                },
                indent=2,
            )
        )

    # -- activation --------------------------------------------------------

    def _activate(self, model: ModelRef, archive: Path) -> LocalModel:
        """Unpack a *verified* archive into its model directory.

        Unpacks to a staging dir and renames, so an interrupted extraction never
        leaves a half-populated directory that looks deployable.
        """
        final = self._model_dir(model.name, model.version)
        staging = final.with_name(final.name + ".staging")
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)

        try:
            if model.packaging is Packaging.MLFLOW_TAR_GZ:
                with tarfile.open(archive, "r:gz") as tar:
                    _safe_extract(tar, staging)
            else:
                shutil.copy2(archive, staging / (model.entrypoint or archive.name))
        except (tarfile.TarError, OSError) as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise ArtifactError(f"could not unpack {model.name}/{model.version}: {exc}") from exc

        entrypoint = self._resolve_entrypoint(staging, model)
        if entrypoint is None:
            shutil.rmtree(staging, ignore_errors=True)
            raise ArtifactError(
                f"{model.name}/{model.version}: no .onnx file found in the unpacked artifact"
            )

        if final.exists():
            shutil.rmtree(final, ignore_errors=True)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)

        resolved = final / entrypoint.relative_to(staging)
        self._write_receipt(final, model, resolved)
        log.info("activated %s/%s at %s", model.name, model.version, resolved)
        return LocalModel(
            name=model.name,
            version=model.version,
            sha256=model.sha256,
            path=final,
            entrypoint=resolved,
        )

    def _resolve_entrypoint(self, root: Path, model: ModelRef) -> Path | None:
        """Find the ONNX file, trusting the server's hint but not requiring it.

        The control plane reads `entrypoint` out of the tarball's `MLmodel`, which
        is right when MLflow wrote it. The glob fallback keeps a hand-built
        artifact working rather than failing on a missing metadata field.
        """
        if model.entrypoint:
            candidate = root / model.entrypoint
            if candidate.is_file():
                return candidate
            matches = sorted(root.rglob(Path(model.entrypoint).name))
            if matches:
                return matches[0]
        onnx = sorted(root.rglob("*.onnx"))
        return onnx[0] if onnx else None

    def _write_receipt(self, model_dir: Path, model: ModelRef, entrypoint: Path) -> None:
        """A human-readable record next to the model.

        Not read by the agent -- `state.json` is the source of truth. This is for
        whoever SSHes into the Jetson at 1am and needs to know what is on disk.
        """
        try:
            (model_dir / ".keeper.json").write_text(
                json.dumps(
                    {
                        "name": model.name,
                        "version": model.version,
                        "sha256": model.sha256,
                        "entrypoint": str(entrypoint),
                        "packaging": model.packaging.value,
                    },
                    indent=2,
                )
            )
        except OSError:  # pragma: no cover
            pass

    def _verified_model_dir(self, model: ModelRef) -> LocalModel | None:
        """Is this exact version already unpacked and still intact?

        The receipt's digest must match the desired digest. A directory whose
        receipt says a different sha256 is a stale deployment of the same label
        and is treated as absent, so the next reconcile rebuilds it -- from the
        verified local archive if that is still on disk, and only otherwise
        from the network. The two caches are deliberately independent layers.
        """
        final = self._model_dir(model.name, model.version)
        receipt = final / ".keeper.json"
        if not receipt.is_file():
            return None
        try:
            data = json.loads(receipt.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if data.get("sha256") != model.sha256:
            return None
        entrypoint = Path(data["entrypoint"]) if data.get("entrypoint") else None
        if entrypoint is not None and not entrypoint.is_file():
            # Someone deleted the model file but left the directory. Treat as
            # absent so the next reconcile repairs it.
            return None
        return LocalModel(
            name=model.name,
            version=model.version,
            sha256=model.sha256,
            path=final,
            entrypoint=entrypoint,
        )

    # -- paths -------------------------------------------------------------

    def _model_dir(self, name: str, version: str) -> Path:
        return self._models / _slug(name) / _slug(version)

    def _engines_dir(self, name: str, version: str) -> Path:
        """Where M6 will compile TensorRT engines. Created by nobody yet, but
        named here so `remove()` already deletes it."""
        return self._settings.data_dir / "engines" / _slug(name) / _slug(version)

    def _archive_path(self, model: ModelRef) -> Path:
        return self._artifacts / f"{_slug(model.name)}-{_slug(model.version)}.tar.gz"

    def _archive_candidates(self, name: str, version: str) -> list[Path]:
        stem = f"{_slug(name)}-{_slug(version)}.tar.gz"
        return [self._artifacts / stem, self._artifacts / f"{stem}.part", self._artifacts / f"{stem}.meta"]


def _digest_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    """Make a registry label safe as a single path segment.

    Both a traversal guard and a portability one: a model named `../../etc` must
    not escape the data directory, and a version label with a slash must not
    silently create a nested tree.
    """
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    cleaned = cleaned.strip("._") or "unnamed"
    return cleaned[:128]


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract only regular files and directories, strictly inside `dest`.

    These bytes arrived over the network. Python's own `extractall` historically
    happily wrote through `../` members and symlinks; 3.12 added filters but this
    agent targets 3.10 on the Jetson, so the check is explicit here rather than
    delegated to a parameter that may not exist.
    """
    root = dest.resolve()
    for member in tar.getmembers():
        if member.issym() or member.islnk():
            raise ArtifactError(f"refusing to extract link member {member.name!r}")
        if not (member.isfile() or member.isdir()):
            raise ArtifactError(f"refusing to extract special member {member.name!r}")
        if member.name.startswith("/") or ".." in Path(member.name).parts:
            raise ArtifactError(f"refusing to extract unsafe path {member.name!r}")
        target = (root / member.name).resolve()
        if root != target and root not in target.parents:
            raise ArtifactError(f"member {member.name!r} escapes the extraction directory")

    # The checks above are the actual guarantee; `filter="data"` is a second,
    # interpreter-maintained opinion layered on top, and from 3.14 it is the
    # default anyway. It is passed defensively because the parameter does not
    # exist on every 3.10 patch level the Jetson might be running, and refusing to
    # unpack a verified artifact because of a missing keyword would be absurd.
    try:
        tar.extractall(dest, filter="data")
    except TypeError:  # pragma: no cover - older tarfile without filters
        tar.extractall(dest)
