"""Hash-on-ingest, caching, eviction and range serving.

This is the component the plan called load-bearing, and the reason is that it is
the only place where a single wrong decision breaks the spec's central security
guarantee. SS17 requires SHA-256 verification before activation over *exactly* the
bytes the device receives -- so the digest this service publishes and the bytes it
later serves have to be the same bytes, under restart, under eviction, under
concurrency, and under a registry that fails halfway through.

The tests therefore care about three things more than functionality: that a digest
is never advertised for bytes that are not on disk, that nothing a live deployment
depends on can be evicted, and that `iter_range` returns exactly HTTP's inclusive
byte range -- an off-by-one there is a corrupted download on the device, reported
as a checksum failure with no clue where it came from.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import replace

import pytest
from lighthouse_contracts import DesiredState, Packaging

from lighthouse.config import Settings
from lighthouse.registry import RegistryUnavailable
from lighthouse.registry.base import ArtifactStream
from lighthouse.registry.fake import FakeModelRegistry, build_fake_artifact
from lighthouse.repositories import DesiredDeploymentRow, SqliteStore
from lighthouse.services.artifact_service import (
    ArtifactFailed,
    ArtifactService,
    _read_entrypoint,
)

from conftest import ADMIN_TOKEN

# The ONNX payload inside the tarball. The artifact itself is slightly *larger*:
# the payload is deterministic pseudo-random data, so gzip cannot compress it, and
# MLmodel plus conda.yaml are added on top. Eviction caps have to be expressed in
# terms of the real artifact size -- a cap of PAYLOAD would evict every artifact
# during its own materialization.
PAYLOAD = 64 * 1024
ARTIFACT_SIZE = len(build_fake_artifact("fashion-cnn", "1", PAYLOAD))


@pytest.fixture
def registry():
    return FakeModelRegistry(payload_size=PAYLOAD)


@pytest.fixture
def svc_settings(tmp_path):
    return Settings(
        env="local",
        data_dir=tmp_path / "lighthouse",
        registry_impl="fake",
        admin_token=ADMIN_TOKEN,
    )


@pytest.fixture
def db(tmp_path):
    store = SqliteStore(tmp_path / "lighthouse.db")
    yield store
    store.close()


@pytest.fixture
def service(db, registry, svc_settings):
    return ArtifactService(db, registry, svc_settings)


def version(registry, name: str = "fashion-cnn", v: str = "1"):
    return registry.get_version(name, v)


def expected_digest(name: str = "fashion-cnn", v: str = "1") -> str:
    return hashlib.sha256(build_fake_artifact(name, v, PAYLOAD)).hexdigest()


# --------------------------------------------------------------------------
# The fake's determinism, which every other assertion rests on
# --------------------------------------------------------------------------


def test_the_fixture_digest_is_stable_across_builds():
    """If this ever fails, every cache and checksum test below becomes flaky, and
    an intermittent checksum failure is about the worst thing to debug in this
    system -- it looks exactly like real corruption."""
    first = build_fake_artifact("fashion-cnn", "1", PAYLOAD)
    second = build_fake_artifact("fashion-cnn", "1", PAYLOAD)

    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()


def test_different_versions_have_different_bytes():
    assert expected_digest(v="1") != expected_digest(v="2")


# --------------------------------------------------------------------------
# Materialization
# --------------------------------------------------------------------------


def test_materializing_computes_the_digest_of_the_bytes_on_disk(service, registry):
    """The one assertion the whole design exists to support: what the control plane
    advertises is the hash of the file it will actually serve."""
    info = service.materialize_now(version(registry))

    assert info.sha256 == expected_digest()
    assert info.path.is_file()
    assert hashlib.sha256(info.path.read_bytes()).hexdigest() == info.sha256
    assert info.size_bytes == info.path.stat().st_size


def test_the_entrypoint_is_read_out_of_the_mlmodel(service, registry):
    """Resolved server-side so the device never has to guess which file in the
    tarball to load."""
    info = service.materialize_now(version(registry))

    assert info.entrypoint == "model.onnx"
    assert info.packaging is Packaging.MLFLOW_TAR_GZ


def test_a_second_request_is_a_cache_hit(service, registry):
    mv = version(registry)
    first = service.materialize_now(mv)

    second = service.request(mv)

    assert second is not None
    assert second.path == first.path
    assert second.sha256 == first.sha256


def test_a_cache_hit_updates_the_access_time(service, registry, db):
    """Eviction order comes off `last_access`, so a hit that did not record itself
    would make a hot artifact look cold and get it evicted under pressure."""
    mv = version(registry)
    service.materialize_now(mv)
    before = db.get_artifact(mv.cache_key).last_access

    service.request(mv)

    assert db.get_artifact(mv.cache_key).last_access >= before


def test_the_cache_is_keyed_by_lineage_not_by_the_version_label(service, registry):
    """The nastiest registry behaviour this guards against: a version label
    repointed at new bytes. Keying on (name, version) would serve the old bytes
    forever under the new label."""
    one = version(registry, v="1")
    two = version(registry, v="2")

    assert one.cache_key != two.cache_key
    assert service.materialize_now(one).path != service.materialize_now(two).path


def test_no_temporary_file_survives_a_successful_publish(service, registry, svc_settings):
    service.materialize_now(version(registry))

    leftovers = list(svc_settings.artifact_cache_dir.glob("*.tmp"))

    assert leftovers == [], f"temporary files left behind: {leftovers}"


def test_a_hostile_lineage_id_cannot_escape_the_cache_directory(service, registry, svc_settings):
    """Lineage ids come from the registry, so they are sanitized rather than
    trusted. A `../` in one must not write outside the cache directory."""
    mv = version(registry)
    hostile = replace(mv, model_id="../../etc", version_uuid="../passwd")

    info = service.materialize_now(hostile)

    assert info.path.parent == svc_settings.artifact_cache_dir
    assert info.path.is_file()


# --------------------------------------------------------------------------
# Readiness is never advertised for bytes that are not there
# --------------------------------------------------------------------------


def test_nothing_is_ready_before_it_is_materialized(service, registry):
    assert service.get_ready(version(registry).cache_key) is None


def test_a_ready_row_with_a_missing_file_is_not_ready(service, registry, db):
    """The cache directory is a filesystem someone can clear -- or, in CAI, a
    project volume that can be restored from an older snapshot than the database.

    Returning the row anyway would make the control plane advertise a digest it
    cannot serve, and the device would report a checksum failure for an artifact
    that simply is not there.
    """
    mv = version(registry)
    info = service.materialize_now(mv)
    info.path.unlink()

    assert service.get_ready(mv.cache_key) is None
    assert db.get_artifact(mv.cache_key) is None, "the stale row was not cleared"


def test_a_cleared_cache_recovers_on_the_next_request(service, registry):
    """Self-healing, because the alternative is an operator noticing."""
    mv = version(registry)
    service.materialize_now(mv).path.unlink()

    recovered = service.materialize_now(mv)

    assert recovered.sha256 == expected_digest()
    assert recovered.path.is_file()


def test_a_pending_row_is_not_ready(service, registry, db):
    from lighthouse.repositories import ArtifactCacheRow
    from lighthouse.util import now_utc

    mv = version(registry)
    db.claim_artifact(
        ArtifactCacheRow(
            cache_key=mv.cache_key,
            model_name=mv.name,
            model_version=mv.version,
            model_id=mv.model_id,
            version_uuid=mv.version_uuid,
            created_at=now_utc(),
        )
    )

    assert service.get_ready(mv.cache_key) is None


# --------------------------------------------------------------------------
# Failure handling
# --------------------------------------------------------------------------


def test_an_unavailable_registry_fails_the_artifact_not_the_process(service, registry, db):
    registry.set_unavailable(True)

    with pytest.raises(ArtifactFailed):
        service.materialize_now(version(FakeModelRegistry(payload_size=PAYLOAD)))

    rows = db.list_artifacts()
    assert len(rows) == 1
    assert rows[0].status == "FAILED"
    assert rows[0].is_ready is False


def test_a_failure_records_why(service, registry, db):
    """The device will never know why; the operator has to be able to find out."""
    registry.set_unavailable(True)
    mv = version(FakeModelRegistry(payload_size=PAYLOAD))

    with pytest.raises(ArtifactFailed):
        service.materialize_now(mv)

    assert "RegistryUnavailable" in db.get_artifact(mv.cache_key).error


def test_a_failed_attempt_leaves_no_partial_file(service, registry, svc_settings):
    """A half-written `.tmp` promoted by a later bug would be an artifact whose
    digest does not match its bytes -- the one outcome SS17 exists to prevent."""
    registry.set_unavailable(True)

    with pytest.raises(ArtifactFailed):
        service.materialize_now(version(FakeModelRegistry(payload_size=PAYLOAD)))

    assert list(svc_settings.artifact_cache_dir.glob("*")) == []


def test_a_failure_can_be_retried(service, registry, db):
    """A transient registry outage must not poison a version permanently. The
    retry path deletes the FAILED row so the claim can be taken again."""
    mv = version(FakeModelRegistry(payload_size=PAYLOAD))
    registry.set_unavailable(True)
    with pytest.raises(ArtifactFailed):
        service.materialize_now(mv)

    registry.set_unavailable(False)
    recovered = service.materialize_now(mv)

    assert recovered.sha256 == expected_digest()
    assert db.get_artifact(mv.cache_key).status == "READY"


def test_a_registry_that_returns_nothing_is_a_failure(db, svc_settings):
    """Zero bytes would otherwise hash successfully -- to the digest of the empty
    string -- and publish as a perfectly valid, perfectly empty artifact."""
    import io

    class EmptyRegistry(FakeModelRegistry):
        def open_artifact(self, mv):
            return ArtifactStream(
                fileobj=io.BytesIO(b""),
                packaging=Packaging.MLFLOW_TAR_GZ,
                size_bytes=0,
                source_uri=mv.artifact_uri,
            )

    registry = EmptyRegistry(payload_size=PAYLOAD)
    service = ArtifactService(db, registry, svc_settings)

    with pytest.raises(ArtifactFailed):
        service.materialize_now(registry.get_version("fashion-cnn", "1"))


def test_a_stream_that_dies_midway_is_a_failure(db, svc_settings):
    """The realistic object-store failure: the connection drops partway through.

    It must not publish the truncated bytes, because their digest is internally
    consistent and the device would verify them happily.
    """
    import io

    class TruncatingStream(ArtifactStream):
        def chunks(self, chunk_size: int = 65536):
            yield b"partial data"
            raise RegistryUnavailable("connection reset by peer")

    class FlakyRegistry(FakeModelRegistry):
        def open_artifact(self, mv):
            return TruncatingStream(
                fileobj=io.BytesIO(b""),
                packaging=Packaging.MLFLOW_TAR_GZ,
                size_bytes=None,
                source_uri=mv.artifact_uri,
            )

    registry = FlakyRegistry(payload_size=PAYLOAD)
    service = ArtifactService(db, registry, svc_settings)
    mv = registry.get_version("fashion-cnn", "1")

    with pytest.raises(ArtifactFailed):
        service.materialize_now(mv)

    assert service.get_ready(mv.cache_key) is None
    assert list(svc_settings.artifact_cache_dir.glob("*")) == []


# --------------------------------------------------------------------------
# Concurrency: request() is the gate
# --------------------------------------------------------------------------


def test_request_returns_none_while_materialization_is_in_flight(service, registry):
    """The spec SS13 contract: accept and return, never wait for the device. None
    means the caller answers `artifact_ready: false` and the device re-polls."""
    mv = version(registry)

    first = service.request(mv)

    assert first is None
    assert service.wait_until_ready(mv.cache_key) is not None


def test_a_second_request_does_not_start_a_second_download(service, registry):
    """Two operators deploying the same new version must not both pull it."""
    mv = version(registry)
    opened: list[str] = []
    real_open = registry.open_artifact

    def counting_open(mv_):
        opened.append(mv_.cache_key)
        return real_open(mv_)

    registry.open_artifact = counting_open  # type: ignore[method-assign]

    service.request(mv)
    service.request(mv)
    service.wait_until_ready(mv.cache_key)

    assert len(opened) == 1, f"artifact was fetched {len(opened)} times"


def test_many_concurrent_requests_produce_one_download(service, registry):
    mv = version(registry)
    opened: list[str] = []
    lock = threading.Lock()
    real_open = registry.open_artifact

    def counting_open(mv_):
        with lock:
            opened.append(mv_.cache_key)
        return real_open(mv_)

    registry.open_artifact = counting_open  # type: ignore[method-assign]
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        service.request(mv)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    service.wait_until_ready(mv.cache_key)

    assert len(opened) == 1, f"artifact was fetched {len(opened)} times"
    assert service.get_ready(mv.cache_key).sha256 == expected_digest()


def test_wait_until_ready_gives_up_on_a_failure_rather_than_timing_out(db, svc_settings):
    registry = FakeModelRegistry(payload_size=PAYLOAD, unavailable=True)
    service = ArtifactService(db, registry, svc_settings)
    mv = FakeModelRegistry(payload_size=PAYLOAD).get_version("fashion-cnn", "1")

    service.request(mv)

    assert service.wait_until_ready(mv.cache_key, timeout=5.0) is None


# --------------------------------------------------------------------------
# Range serving
# --------------------------------------------------------------------------


def test_a_full_read_reproduces_the_file(service, registry):
    info = service.materialize_now(version(registry))

    body = b"".join(service.iter_range(info))

    assert hashlib.sha256(body).hexdigest() == info.sha256
    assert len(body) == info.size_bytes


def test_a_range_is_inclusive_at_both_ends(service, registry):
    """HTTP Range is inclusive. An off-by-one here is a one-byte-short download
    that the device reports as a checksum failure, which is a genuinely horrible
    bug to trace back to its cause."""
    info = service.materialize_now(version(registry))
    whole = info.path.read_bytes()

    assert b"".join(service.iter_range(info, 0, 0)) == whole[0:1]
    assert b"".join(service.iter_range(info, 10, 19)) == whole[10:20]
    assert b"".join(service.iter_range(info, info.size_bytes - 1, info.size_bytes - 1)) == whole[-1:]


def test_resuming_from_an_offset_returns_the_remainder(service, registry):
    """What the agent's resume path actually asks for."""
    info = service.materialize_now(version(registry))
    whole = info.path.read_bytes()
    offset = 40_000

    tail = b"".join(service.iter_range(info, offset))

    assert tail == whole[offset:]


def test_a_prefix_plus_its_suffix_is_the_whole_file(service, registry):
    """The resume invariant stated directly: an interrupted transfer plus its
    continuation must hash to what the control plane advertised."""
    info = service.materialize_now(version(registry))
    cut = 12_345

    head = b"".join(service.iter_range(info, 0, cut - 1))
    tail = b"".join(service.iter_range(info, cut))

    assert len(head) == cut
    assert hashlib.sha256(head + tail).hexdigest() == info.sha256


def test_an_empty_range_yields_nothing(service, registry):
    info = service.materialize_now(version(registry))

    assert b"".join(service.iter_range(info, 10, 9)) == b""


def test_a_start_past_the_end_yields_nothing_rather_than_raising(service, registry):
    info = service.materialize_now(version(registry))

    assert b"".join(service.iter_range(info, info.size_bytes + 100)) == b""


def test_small_chunks_still_reproduce_the_file(db, registry, svc_settings):
    """CML's ingress behaviour on long transfers is undocumented, so the chunk size
    is configurable. Changing it must not change the bytes."""
    svc_settings.artifact_chunk_size = 1024
    service = ArtifactService(db, registry, svc_settings)
    info = service.materialize_now(version(registry))

    body = b"".join(service.iter_range(info))

    assert hashlib.sha256(body).hexdigest() == expected_digest()


def test_the_corruption_hook_changes_the_bytes_without_changing_the_digest(
    db, registry, svc_settings
):
    """The dev-only hook behind `make dev`'s checksum-failure demo.

    It must corrupt the *served* bytes only. If it touched the cache file the
    advertised digest would move with it and the device would verify happily --
    proving nothing.
    """
    svc_settings.dev_corrupt_artifacts = True
    service = ArtifactService(db, registry, svc_settings)
    info = service.materialize_now(version(registry))

    served = b"".join(service.iter_range(info))

    assert info.sha256 == expected_digest(), "the stored digest was affected"
    assert hashlib.sha256(served).hexdigest() != info.sha256
    assert len(served) == info.size_bytes, "corruption changed the length, so size would catch it"
    assert hashlib.sha256(info.path.read_bytes()).hexdigest() == info.sha256


# --------------------------------------------------------------------------
# Eviction
# --------------------------------------------------------------------------


def enroll_and_desire(db, mv, device_id: str = "jetson-orin-01"):
    """Pin an artifact by making it some device's current desired state.

    Deliberately called *before* materializing, which is also the production
    order: `PUT /deployment` writes desired state to allocate a generation and only
    then asks for the bytes. Pinning afterwards would be too late -- eviction runs
    at the end of materialization, so an artifact can be evicted by its own
    materialization before anything has had a chance to reference it.
    """
    db.create_device(device_id, None, None)
    db.set_desired(
        DesiredDeploymentRow(
            device_id=device_id,
            generation=0,
            desired_state=DesiredState.RUNNING,
            model_name=mv.name,
            model_version=mv.version,
            model_id=mv.model_id,
            version_uuid=mv.version_uuid,
        )
    )


def test_nothing_is_evicted_under_the_cap(service, registry, db):
    for v in ("1", "2", "3"):
        service.materialize_now(version(registry, v=v))

    assert len(db.list_artifacts()) == 3


def test_the_least_recently_used_entry_goes_first(db, registry, svc_settings):
    """Room for two and a bit, so adding a third evicts exactly one -- and it must
    be the one nobody has touched, not simply the oldest by creation."""
    svc_settings.artifact_cache_max_bytes = ARTIFACT_SIZE * 2 + ARTIFACT_SIZE // 2
    service = ArtifactService(db, registry, svc_settings)

    first = version(registry, v="1")
    cold = version(registry, v="2")
    service.materialize_now(first)
    service.materialize_now(cold)
    service.request(first)  # a cache hit, which makes v2 the coldest
    service.materialize_now(version(registry, v="3"))

    remaining = {r.cache_key for r in db.list_artifacts()}
    assert first.cache_key in remaining, "evicted the entry that was just used"
    assert cold.cache_key not in remaining, "kept the coldest entry instead"


def test_eviction_deletes_the_file_too(db, registry, svc_settings):
    """A row-only delete would leak the cache directory until the volume filled,
    which is the failure eviction exists to prevent."""
    svc_settings.artifact_cache_max_bytes = ARTIFACT_SIZE + ARTIFACT_SIZE // 2
    service = ArtifactService(db, registry, svc_settings)
    doomed = service.materialize_now(version(registry, v="1"))
    service.materialize_now(version(registry, v="2"))

    assert not doomed.path.exists(), "evicted the row but left the bytes on disk"
    assert list(svc_settings.artifact_cache_dir.glob("*.tar.gz")) != []


def test_a_live_deployment_pins_its_artifact_against_eviction(db, registry, svc_settings):
    """The sharp edge. A device offline for a week comes back, polls, and is told to
    fetch an artifact the control plane evicted -- and the device has no other way
    to obtain those bytes, so it can never converge again.
    """
    svc_settings.artifact_cache_max_bytes = ARTIFACT_SIZE + ARTIFACT_SIZE // 2
    service = ArtifactService(db, registry, svc_settings)
    pinned = version(registry, v="1")
    enroll_and_desire(db, pinned)
    info = service.materialize_now(pinned)

    for v in ("2", "3"):
        service.materialize_now(version(registry, v=v))

    assert service.get_ready(pinned.cache_key) is not None, "evicted a live deployment"
    assert info.path.is_file()


def test_pinning_does_not_protect_the_rest_of_the_cache(db, registry, svc_settings):
    """The converse, so the test above cannot pass by eviction being broken."""
    svc_settings.artifact_cache_max_bytes = ARTIFACT_SIZE + ARTIFACT_SIZE // 2
    service = ArtifactService(db, registry, svc_settings)
    pinned = version(registry, v="1")
    enroll_and_desire(db, pinned)
    service.materialize_now(pinned)
    service.materialize_now(version(registry, v="2"))

    service.materialize_now(version(registry, v="3"))

    keys = {r.cache_key for r in db.list_artifacts()}
    assert pinned.cache_key in keys
    assert len(keys) < 3, "nothing was evicted at all, so pinning proves nothing"


def test_eviction_stops_once_it_is_under_the_cap(db, registry, svc_settings):
    """It must not clear the cache wholesale: every eviction beyond what is needed
    is a multi-MB re-download over someone's home uplink."""
    svc_settings.artifact_cache_max_bytes = ARTIFACT_SIZE * 2 + ARTIFACT_SIZE // 2
    service = ArtifactService(db, registry, svc_settings)
    for v in ("1", "2", "3"):
        service.materialize_now(version(registry, v=v))

    assert len(db.list_artifacts()) == 2


def test_a_cap_of_zero_disables_eviction(db, registry, svc_settings):
    svc_settings.artifact_cache_max_bytes = 0
    service = ArtifactService(db, registry, svc_settings)
    for v in ("1", "2", "3"):
        service.materialize_now(version(registry, v=v))

    assert len(db.list_artifacts()) == 3


def test_everything_pinned_means_nothing_is_evicted_even_over_the_cap(db, registry, svc_settings):
    """Over the cap and unable to act is the correct outcome: serving a device
    beats respecting a soft byte budget. Running out of disk is an operator
    problem; a device that can never converge is a broken fleet.
    """
    svc_settings.artifact_cache_max_bytes = 1
    service = ArtifactService(db, registry, svc_settings)
    pinned = []
    for index, v in enumerate(("1", "2", "3")):
        mv = version(registry, v=v)
        enroll_and_desire(db, mv, device_id=f"device-{index}")
        service.materialize_now(mv)
        pinned.append(mv)

    service.materialize_now(version(registry, name="fraud-detector", v="6"))

    keys = {r.cache_key for r in db.list_artifacts()}
    assert {mv.cache_key for mv in pinned} <= keys, "evicted a live deployment under pressure"


# --------------------------------------------------------------------------
# Operator removal
# --------------------------------------------------------------------------


def test_remove_deletes_the_row_and_the_file(service, registry, db):
    mv = version(registry)
    info = service.materialize_now(mv)

    service.remove(mv.cache_key)

    assert db.get_artifact(mv.cache_key) is None
    assert not info.path.exists()


def test_remove_ignores_an_unknown_key(service):
    service.remove("nonexistent/key")  # must not raise


def test_remove_overrides_pinning(service, registry, db):
    """Deliberate asymmetry: eviction is automatic and must respect references,
    removal is an operator saying "delete this" and is allowed to win."""
    mv = version(registry)
    service.materialize_now(mv)
    enroll_and_desire(db, mv)

    service.remove(mv.cache_key)

    assert db.get_artifact(mv.cache_key) is None


# --------------------------------------------------------------------------
# MLmodel parsing
# --------------------------------------------------------------------------


def test_the_entrypoint_comes_from_the_onnx_flavor(tmp_path):
    path = tmp_path / "a.tar.gz"
    path.write_bytes(build_fake_artifact("m", "1", 1024, onnx_path="classifier.onnx"))

    assert _read_entrypoint(path, Packaging.MLFLOW_TAR_GZ) == "classifier.onnx"


def test_a_corrupt_archive_does_not_raise(tmp_path):
    """Best-effort by design: an unreadable MLmodel is not a reason to reject an
    artifact whose digest is perfectly good. The agent falls back to its own
    discovery."""
    path = tmp_path / "bad.tar.gz"
    path.write_bytes(b"this is not a gzip stream")

    assert _read_entrypoint(path, Packaging.MLFLOW_TAR_GZ) is None


def test_a_missing_file_does_not_raise(tmp_path):
    assert _read_entrypoint(tmp_path / "absent.tar.gz", Packaging.MLFLOW_TAR_GZ) is None


def test_raw_packaging_is_not_parsed(tmp_path):
    """A raw single file has no MLmodel to read, and trying to open one as a
    tarball would log a warning on every materialization for no reason."""
    path = tmp_path / "a.tar.gz"
    path.write_bytes(build_fake_artifact("m", "1", 1024))

    assert _read_entrypoint(path, Packaging.RAW_FILE) is None


def test_the_cache_directory_is_created_on_construction(db, registry, tmp_path):
    """The CAI project filesystem will not have it pre-made."""
    settings = Settings(
        env="local",
        data_dir=tmp_path / "deep" / "nested",
        registry_impl="fake",
        admin_token=ADMIN_TOKEN,
    )

    ArtifactService(db, registry, settings)

    assert settings.artifact_cache_dir.is_dir()


def test_the_cache_survives_a_restart(db, registry, svc_settings):
    """Both halves have to survive: the row in SQLite and the file on the project
    volume. A restart that re-downloaded everything would be slow; one that served
    a stale digest would be wrong."""
    mv = version(registry)
    first = ArtifactService(db, registry, svc_settings).materialize_now(mv)

    second = ArtifactService(db, registry, svc_settings).get_ready(mv.cache_key)

    assert second is not None
    assert second.sha256 == first.sha256
    assert second.path == first.path
