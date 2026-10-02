"""Download, verify, unpack and delete, tested at the unit level.

The reconciler suite reaches this code through the happy path. These tests go at
the parts a reconciler test cannot reach without contrivance: resume arithmetic,
the sidecar's job, and the extraction guards -- which are the only place in either
package where bytes from the network are turned into files on disk.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile

import pytest
from lighthouse_contracts import ModelRef, Packaging

from keeper.artifact_manager import (
    ArtifactError,
    ArtifactManager,
    ChecksumMismatch,
    DownloadAborted,
    _safe_extract,
    _slug,
)

from conftest import DEVICE_ID, make_archive


@pytest.fixture
def manager(harness):
    return harness.artifacts


# --------------------------------------------------------------------------
# Determinism of the fixture itself
# --------------------------------------------------------------------------


def test_archive_bytes_are_reproducible():
    """If this fails, every checksum assertion in the suite becomes intermittent.

    Worth its own test because the failure mode is "a different test goes flaky
    next Tuesday", which is miserable to trace back to a gzip timestamp.
    """
    assert make_archive("m", "1") == make_archive("m", "1")
    assert make_archive("m", "1") != make_archive("m", "2")


# --------------------------------------------------------------------------
# ensure(): verify before activate
# --------------------------------------------------------------------------


def test_ensure_downloads_verifies_and_unpacks(harness, manager):
    model = harness.model("fashion-cnn", "1")

    local = manager.ensure(model)

    assert local.sha256 == model.sha256
    assert local.path.is_dir()
    assert local.entrypoint is not None and local.entrypoint.name == "model.onnx"
    assert local.load_target == local.entrypoint
    receipt = json.loads((local.path / ".keeper.json").read_text())
    assert receipt["sha256"] == model.sha256


def test_ensure_is_cached_on_the_second_call(harness, manager):
    model = harness.model("fashion-cnn", "1")
    manager.ensure(model)

    manager.ensure(model)

    assert harness.client.download_count == 1


def test_checksum_mismatch_leaves_nothing_behind(harness, manager):
    """Spec SS17: a failed checksum must prevent deployment. "Prevent" has to mean
    no unpacked directory and no partial that a later resume would build on -- a
    mismatch gives no way to tell which bytes are wrong, so resuming from them
    would never converge."""
    model = harness.model("fashion-cnn", "1")
    harness.client.corrupt_bytes = True

    with pytest.raises(ChecksumMismatch) as exc:
        manager.ensure(model)

    assert model.sha256 in str(exc.value)
    assert not harness.model_dir("fashion-cnn", "1").exists()
    assert list(harness.settings.artifact_dir.glob("*.part")) == []
    assert list(harness.settings.artifact_dir.glob("*.meta")) == []


def test_short_body_is_rejected_on_size(harness, manager):
    """A truncated transfer fails the digest first, but the explicit size check is
    the one that produces a diagnosable message."""
    model = harness.model("fashion-cnn", "1")
    harness.client.truncate_after = 64

    with pytest.raises(ArtifactError):
        manager.ensure(model)

    assert not harness.model_dir("fashion-cnn", "1").exists()


def test_stale_receipt_is_repaired_from_the_local_archive(harness, manager):
    """A directory whose receipt names a different digest is a stale deployment of
    the same mutable label, not a cache hit -- so it is rebuilt.

    Rebuilt, not re-fetched: the verified archive is still on disk, and there is no
    reason to pull several MB over a home uplink to fix a bad receipt. The two
    caches are deliberately independent layers.
    """
    model = harness.model("fashion-cnn", "1")
    local = manager.ensure(model)
    receipt = local.path / ".keeper.json"
    data = json.loads(receipt.read_text())
    data["sha256"] = "0" * 64
    receipt.write_text(json.dumps(data))

    repaired = manager.ensure(model)

    assert json.loads((repaired.path / ".keeper.json").read_text())["sha256"] == model.sha256
    assert harness.client.download_count == 1, "re-fetched bytes it had already verified"


def test_deleted_model_file_is_repaired(harness, manager):
    """Someone cleaning up disk by hand removed the ONNX but left the directory."""
    model = harness.model("fashion-cnn", "1")
    local = manager.ensure(model)
    local.entrypoint.unlink()

    repaired = manager.ensure(model)

    assert repaired.entrypoint.is_file()
    assert harness.client.download_count == 1


def test_missing_archive_and_tree_redownloads(harness, manager):
    """Both caches gone -- the revoke case -- is the one that must hit the network."""
    model = harness.model("fashion-cnn", "1")
    manager.ensure(model)
    manager.remove("fashion-cnn", "1")

    manager.ensure(model)

    assert harness.client.download_count == 2


# --------------------------------------------------------------------------
# Resume
# --------------------------------------------------------------------------


def _part_paths(harness):
    return (
        sorted(harness.settings.artifact_dir.glob("*.part")),
        sorted(harness.settings.artifact_dir.glob("*.meta")),
    )


def test_resume_continues_from_the_partial(harness, manager):
    model = harness.model("fashion-cnn", "1", size=256 * 1024)
    harness.client.chunk_size = 32 * 1024

    calls = {"n": 0}

    def abort_after_two() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(DownloadAborted):
        manager.ensure(model, should_abort=abort_after_two)

    parts, metas = _part_paths(harness)
    assert parts and metas
    first_pass = parts[0].stat().st_size
    assert 0 < first_pass < len(harness.client.archives[model.artifact_uri])

    local = manager.ensure(model)

    # The digest covers the whole file, resumed head included -- not just the tail
    # fetched on the second attempt.
    assert local.sha256 == model.sha256
    assert _part_paths(harness) == ([], [])


def test_resume_discards_a_partial_from_different_bytes(harness, manager):
    """The sidecar's entire reason for existing.

    A version label repointed at new bytes would otherwise splice the old partial
    onto the new tail and produce an archive that belongs to neither.
    """
    old = harness.model("fashion-cnn", "1", size=256 * 1024)
    harness.client.chunk_size = 32 * 1024
    calls = {"n": 0}
    with pytest.raises(DownloadAborted):
        manager.ensure(
            old,
            should_abort=lambda: (calls.__setitem__("n", calls["n"] + 1), calls["n"] > 2)[1],
        )
    parts, _ = _part_paths(harness)
    assert parts

    # Same name and version, different content.
    repointed = harness.model("fashion-cnn", "1", size=200 * 1024)
    assert repointed.sha256 != old.sha256
    harness.client.chunk_size = None

    local = manager.ensure(repointed)

    assert local.sha256 == repointed.sha256


def test_partial_without_a_sidecar_restarts(harness, manager):
    model = harness.model("fashion-cnn", "1")
    harness.settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    orphan = harness.settings.artifact_dir / "fashion-cnn-1.tar.gz.part"
    orphan.write_bytes(b"bytes from who knows where")

    local = manager.ensure(model)

    assert local.sha256 == model.sha256


def test_server_ignoring_range_restarts_from_zero(harness, manager, monkeypatch):
    """Some proxies answer a Range request with the whole body and a 200.

    Appending that to a partial would silently produce a corrupt archive, so the
    only safe response is to throw the partial away. CML's ingress behaviour on
    ranged requests is undocumented, which is exactly why this path exists.
    """
    model = harness.model("fashion-cnn", "1", size=256 * 1024)
    harness.client.chunk_size = 32 * 1024
    calls = {"n": 0}
    with pytest.raises(DownloadAborted):
        manager.ensure(
            model,
            should_abort=lambda: (calls.__setitem__("n", calls["n"] + 1), calls["n"] > 2)[1],
        )
    assert _part_paths(harness)[0]

    # Second attempt: the stub replies 200 with the full body despite the offset.
    real_stream = harness.client.stream_artifact

    def ignores_range(artifact_uri, *, offset=0, if_match=None):
        return real_stream(artifact_uri, offset=0, if_match=None)

    monkeypatch.setattr(harness.client, "stream_artifact", ignores_range)
    harness.client.chunk_size = None

    local = manager.ensure(model)

    assert local.sha256 == model.sha256


def test_complete_but_unverified_partial_is_rehashed(harness, manager):
    """Killed between the final write and the rename.

    The file is the right length, so a length-only resume check would treat it as
    done and skip verification entirely.
    """
    model = harness.model("fashion-cnn", "1")
    body = harness.client.archives[model.artifact_uri]
    harness.settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    part = harness.settings.artifact_dir / "fashion-cnn-1.tar.gz.part"
    part.write_bytes(b"\x00" * len(body))
    (harness.settings.artifact_dir / "fashion-cnn-1.tar.gz.meta").write_text(
        json.dumps({"sha256": model.sha256, "size_bytes": len(body)})
    )

    local = manager.ensure(model)

    assert local.sha256 == model.sha256


def test_verified_archive_on_disk_is_not_refetched(harness, manager):
    """Unpacked tree gone, archive intact -- e.g. someone cleared `models/`."""
    model = harness.model("fashion-cnn", "1")
    manager.ensure(model)
    import shutil

    shutil.rmtree(harness.model_dir("fashion-cnn", "1"))

    manager.ensure(model)

    assert harness.client.download_count == 1, "re-downloaded bytes it already had verified"


# --------------------------------------------------------------------------
# remove / prune / remove_all
# --------------------------------------------------------------------------


def test_remove_deletes_archive_and_model(harness, manager):
    model = harness.model("fashion-cnn", "1")
    manager.ensure(model)

    manager.remove("fashion-cnn", "1")

    assert not harness.model_dir("fashion-cnn", "1").exists()
    assert list(harness.settings.artifact_dir.glob("fashion-cnn-1*")) == []


def test_remove_tolerates_a_model_that_was_never_downloaded(manager):
    """Revoke must succeed on a device that never finished downloading, or the
    device is stuck reporting REVOKE_PENDING forever."""
    manager.remove("never-existed", "7")  # must not raise


def test_remove_deletes_derived_engine_artifacts(harness, manager):
    """SS5: revocation removes generated runtime artifacts too. M6 compiles
    TensorRT engines into this directory; revoke has to already know about it,
    because the alternative is a revoked model still being served from a cached
    engine."""
    model = harness.model("fashion-cnn", "1")
    manager.ensure(model)
    engines = harness.settings.data_dir / "engines" / "fashion-cnn" / "1"
    engines.mkdir(parents=True)
    (engines / "model.plan").write_bytes(b"fake tensorrt engine")

    manager.remove("fashion-cnn", "1")

    assert not engines.exists()


def test_prune_keeps_one_version_back_for_rollback(harness, manager):
    for version in ("1", "2", "3"):
        manager.ensure(harness.model("fashion-cnn", version))

    manager.prune("fashion-cnn", keep_versions=("3",))

    kept = sorted(p.name for p in (harness.settings.model_dir / "fashion-cnn").iterdir())
    # 3 is live and 2 is the rollback candidate; 1 is the one that has to go, or a
    # device with 32GB of eMMC accumulates every version it ever ran.
    assert "3" in kept
    assert "1" not in kept
    assert len(kept) == 2


def test_prune_never_removes_the_live_version(harness, manager):
    manager.ensure(harness.model("fashion-cnn", "1"))

    manager.prune("fashion-cnn", keep_versions=("1",))

    assert harness.model_dir("fashion-cnn", "1").exists()


def test_prune_is_a_noop_for_an_unknown_model(manager):
    manager.prune("never-existed", keep_versions=("1",))


def test_remove_all_clears_everything(harness, manager):
    manager.ensure(harness.model("fashion-cnn", "1"))
    manager.ensure(harness.model("other-model", "4"))

    manager.remove_all()

    assert not harness.settings.model_dir.exists()
    assert not harness.settings.artifact_dir.exists()


# --------------------------------------------------------------------------
# Extraction guards. These bytes arrived over the network.
# --------------------------------------------------------------------------


def _tar_with(members, *, mode="w") -> io.BytesIO:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode=mode) as tar:
        for info, data in members:
            tar.addfile(info, io.BytesIO(data) if data is not None else None)
    raw.seek(0)
    return raw


def _regular(name: str, data: bytes = b"x") -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    return info, data


def _symlink(name: str, target: str) -> tuple[tarfile.TarInfo, None]:
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.linkname = target
    return info, None


def _unpack(raw: io.BytesIO, dest):
    with tarfile.open(fileobj=raw, mode="r") as tar:
        _safe_extract(tar, dest)


@pytest.mark.parametrize(
    "name",
    [
        "../escaped.onnx",
        "/absolute.onnx",
        "nested/../../escaped.onnx",
    ],
)
def test_traversal_members_are_refused(tmp_path, name):
    dest = tmp_path / "out"
    dest.mkdir()

    with pytest.raises(ArtifactError):
        _unpack(_tar_with([_regular(name)]), dest)


def test_symlink_members_are_refused(tmp_path):
    """A symlink named `model.onnx` pointing at `/etc/shadow` would make the next
    read of "the model" a read of whatever the link targets."""
    dest = tmp_path / "out"
    dest.mkdir()

    with pytest.raises(ArtifactError):
        _unpack(_tar_with([_symlink("model.onnx", "/etc/passwd")]), dest)


def test_device_members_are_refused(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    info = tarfile.TarInfo("dev/null")
    info.type = tarfile.CHRTYPE

    with pytest.raises(ArtifactError):
        _unpack(_tar_with([(info, None)]), dest)


def test_nothing_is_written_when_a_member_is_refused(tmp_path):
    """The guard runs over every member before any is written, so a tarball whose
    last member is malicious cannot leave its earlier members on disk."""
    dest = tmp_path / "out"
    dest.mkdir()

    with pytest.raises(ArtifactError):
        _unpack(_tar_with([_regular("good.onnx"), _regular("../bad.onnx")]), dest)

    assert list(dest.iterdir()) == []


def test_ordinary_members_extract(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()

    _unpack(_tar_with([_regular("model/model.onnx", b"onnx bytes")]), dest)

    assert (dest / "model" / "model.onnx").read_bytes() == b"onnx bytes"


def test_archive_without_onnx_is_rejected(harness, manager):
    """Better to fail at activation with a clear message than to hand a runtime a
    directory and let onnxruntime produce something cryptic."""
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            info = tarfile.TarInfo("MLmodel")
            payload = b"flavors:\n  sklearn: {}\n"
            info.size = len(payload)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(payload))
    body = raw.getvalue()

    uri = f"/api/v1/devices/{DEVICE_ID}/artifact?model=sk&version=1"
    harness.client.archives[uri] = body
    model = ModelRef(
        name="sk",
        version="1",
        sha256=hashlib.sha256(body).hexdigest(),
        artifact_uri=uri,
        packaging=Packaging.MLFLOW_TAR_GZ,
        size_bytes=len(body),
    )

    with pytest.raises(ArtifactError, match="no .onnx"):
        manager.ensure(model)

    assert not harness.model_dir("sk", "1").exists()


def test_corrupt_gzip_is_an_artifact_error(harness, manager):
    """Bytes that match the published digest but are not a valid archive. The
    control plane hashed them, so this is a registry or packaging fault, and it
    must surface as a deployment failure rather than a traceback."""
    body = b"this is definitely not gzip" * 100
    uri = f"/api/v1/devices/{DEVICE_ID}/artifact?model=junk&version=1"
    harness.client.archives[uri] = body
    model = ModelRef(
        name="junk",
        version="1",
        sha256=hashlib.sha256(body).hexdigest(),
        artifact_uri=uri,
        packaging=Packaging.MLFLOW_TAR_GZ,
        size_bytes=len(body),
    )

    with pytest.raises(ArtifactError):
        manager.ensure(model)

    assert not harness.model_dir("junk", "1").exists()


# --------------------------------------------------------------------------
# Path safety for registry labels
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("fashion-cnn", "fashion-cnn"),
        ("../../etc/passwd", "_.._etc_passwd".replace("_.._", "_.._")),
        ("v1.2.3", "v1.2.3"),
        ("", "unnamed"),
        ("...", "unnamed"),
    ],
)
def test_slug_never_escapes_a_single_segment(value, expected):
    result = _slug(value)
    assert "/" not in result
    assert result not in ("", ".", "..")
    assert len(result) <= 128


def test_a_malicious_model_name_stays_inside_the_data_dir(harness, manager):
    """A registry is an upstream system. A model named `../../..` must not let it
    write outside the agent's data directory."""
    uri = f"/api/v1/devices/{DEVICE_ID}/artifact?model=evil&version=1"
    body = make_archive("evil", "1")
    harness.client.archives[uri] = body
    model = ModelRef(
        name="../../../../tmp/escaped",
        version="1",
        sha256=hashlib.sha256(body).hexdigest(),
        artifact_uri=uri,
        packaging=Packaging.MLFLOW_TAR_GZ,
        entrypoint="model.onnx",
        size_bytes=len(body),
    )

    local = manager.ensure(model)

    assert harness.settings.data_dir.resolve() in local.path.resolve().parents
