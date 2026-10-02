"""The persistence layer, tested where it carries weight.

Most of this file is about **generation allocation**, because generation is the
only ordering mechanism in the system (spec SS4: ordering never uses timestamps) and
a lost or reused generation is not a cosmetic bug -- it is a device that silently
keeps running a revoked model because the instruction to drop it shared a number
with one it had already acknowledged.

The concurrency tests use a file-backed store on purpose. `:memory:` shares one
connection behind a lock, so it would exercise the lock rather than SQLite's
`BEGIN IMMEDIATE`, and the lock is not what runs in production.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import timezone

import pytest
from lighthouse_contracts import ActualState, DesiredState, EventType

from lighthouse.repositories import (
    ActualDeploymentRow,
    ArtifactCacheRow,
    AuditEventRow,
    DesiredDeploymentRow,
    SqliteStore,
)
from lighthouse.repositories.base import DeviceExists, DeviceUnknown
from lighthouse.util import now_utc

DEVICE = "jetson-orin-01"


@pytest.fixture
def db(tmp_path):
    """A file-backed store: real connections, real write locking."""
    store = SqliteStore(tmp_path / "lighthouse.db")
    yield store
    store.close()


def enrolled(store: SqliteStore, device_id: str = DEVICE):
    return store.create_device(device_id, "Bench Jetson", "jetson-orin")


def deploy(device_id: str = DEVICE, *, version: str = "1", state=DesiredState.RUNNING):
    return DesiredDeploymentRow(
        device_id=device_id,
        generation=0,  # ignored; the store allocates
        desired_state=state,
        model_name="fashion-cnn",
        model_version=version,
        model_id="m-1",
        version_uuid=f"v-{version}",
        artifact_sha256="a" * 64,
    )


# --------------------------------------------------------------------------
# Enrollment
# --------------------------------------------------------------------------


def test_registering_a_device_seeds_both_sides(db):
    """A new device is explicitly "told to run nothing" rather than having no
    desired row, which keeps every read path free of a None special case."""
    enrolled(db)

    desired = db.get_desired(DEVICE)
    actual = db.get_actual(DEVICE)

    assert desired is not None and desired.generation == 0
    assert desired.desired_state is DesiredState.STOPPED
    assert actual is not None and actual.actual_state is ActualState.UNKNOWN
    assert actual.observed_generation == 0


def test_duplicate_registration_is_refused(db):
    enrolled(db)

    with pytest.raises(DeviceExists):
        enrolled(db)


def test_a_failed_duplicate_leaves_the_original_intact(db):
    """The insert is three statements in one transaction. A partial rollback would
    leave a device with no desired row, which every read path assumes exists."""
    enrolled(db)
    db.set_desired(deploy())

    with pytest.raises(DeviceExists):
        db.create_device(DEVICE, "impostor", "x")

    assert db.get_device(DEVICE).display_name == "Bench Jetson"
    assert db.get_desired(DEVICE).generation == 1


def test_setting_desired_state_for_an_unknown_device_is_refused(db):
    with pytest.raises(DeviceUnknown):
        db.set_desired(deploy("never-registered"))


def test_timestamps_come_back_timezone_aware(db):
    """Stored as ISO-8601 text. If they reloaded as naive datetimes, every
    connectivity comparison against an aware `now` would raise at runtime -- in the
    dashboard, not here."""
    enrolled(db)

    registered = db.get_device(DEVICE).registered_at

    assert registered.tzinfo is not None
    assert registered.utcoffset() == timezone.utc.utcoffset(None)


# --------------------------------------------------------------------------
# Generation allocation
# --------------------------------------------------------------------------


def test_each_change_advances_the_generation(db):
    enrolled(db)

    generations = [db.set_desired(deploy(version=v)).generation for v in ("1", "2", "3")]

    assert generations == [1, 2, 3]


def test_the_caller_cannot_choose_the_generation(db):
    """The row carries a generation field, and the store must ignore it. Honoring
    it would let a caller rewind the counter and make a new instruction look like
    one the device already acknowledged."""
    enrolled(db)
    db.set_desired(deploy())

    row = deploy(version="2")
    row.generation = 1  # a stale value a caller might carry in

    assert db.set_desired(row).generation == 2


def test_the_returned_row_carries_the_allocated_generation(db):
    """Callers audit and respond from the returned row. If it still held the input
    value, the audit event and the API response would disagree with the database."""
    enrolled(db)

    returned = db.set_desired(deploy())

    assert returned.generation == db.get_desired(DEVICE).generation
    assert returned.updated_at is not None


def test_recording_the_digest_does_not_advance_the_generation(db):
    """A deployment is accepted before the bytes are cached, so the digest lands
    later. That is new knowledge about an unchanged instruction -- bumping the
    generation for it would make every device reconcile again for nothing.
    """
    enrolled(db)
    generation = db.set_desired(deploy()).generation

    db.update_desired_digest(DEVICE, "b" * 64)

    row = db.get_desired(DEVICE)
    assert row.generation == generation
    assert row.artifact_sha256 == "b" * 64


def test_recording_a_digest_for_an_unknown_device_is_refused(db):
    with pytest.raises(DeviceUnknown):
        db.update_desired_digest("never-registered", "c" * 64)


def test_generation_is_never_reused_even_if_history_runs_ahead(db):
    """Defensive: allocation takes MAX over the current row *and* history.

    The skew is forced here with raw SQL because no supported sequence produces it
    today. It is guarded anyway because the consequence is the worst one available
    -- two different instructions sharing a generation, where a device that
    acknowledged the first would ignore the second forever.
    """
    enrolled(db)
    db.set_desired(deploy())
    db._conn.execute(
        "INSERT INTO deployment_history (device_id, generation, desired_state, created_at) "
        "VALUES (?, 99, 'STOPPED', ?)",
        (DEVICE, "2026-10-01T00:00:00+00:00"),
    )
    db._conn.commit()

    assert db.set_desired(deploy(version="2")).generation == 100


def test_history_records_every_change_newest_first(db):
    enrolled(db)
    for version in ("1", "2", "3"):
        db.set_desired(deploy(version=version))
    db.set_desired(deploy(version="1"))  # a rollback

    history = db.list_history(DEVICE)

    assert [h.generation for h in history] == [4, 3, 2, 1]
    assert [h.model_version for h in history] == ["1", "3", "2", "1"]


def test_history_is_what_makes_rollback_detectable(db):
    """The classification input: an operator cannot mislabel a rollback because the
    server compares against where the device has actually been."""
    enrolled(db)
    db.set_desired(deploy(version="1"))
    db.set_desired(deploy(version="2"))

    previous = {h.model_version for h in db.list_history(DEVICE)}

    assert "1" in previous


def test_concurrent_changes_all_get_distinct_generations(db):
    """The lost-update test, and the reason writes use BEGIN IMMEDIATE.

    Under SQLite's default deferred transactions two operators could both read
    generation N and both write N+1 -- one desired-state change vanishing with no
    error anywhere.
    """
    enrolled(db)
    seen: list[int] = []
    lock = threading.Lock()
    barrier = threading.Barrier(8)

    def worker(n: int):
        barrier.wait()
        for _ in range(5):
            row = db.set_desired(deploy(version=str(n)))
            with lock:
                seen.append(row.generation)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(seen) == 40
    assert len(set(seen)) == 40, "a generation was handed out twice"
    assert sorted(seen) == list(range(1, 41)), "the sequence has gaps"
    assert db.get_desired(DEVICE).generation == 40


# --------------------------------------------------------------------------
# Actual state
# --------------------------------------------------------------------------


def test_heartbeats_overwrite_rather_than_accumulate(db):
    """Only the latest report is kept. History of actual state is the audit log's
    job, not this table's -- a row per heartbeat would be 8,640 rows per device per
    day on a 10-second interval."""
    enrolled(db)

    for generation, state in ((1, ActualState.DOWNLOADING), (1, ActualState.RUNNING)):
        db.set_actual(
            ActualDeploymentRow(
                device_id=DEVICE,
                observed_generation=generation,
                actual_state=state,
                model_name="fashion-cnn",
                model_version="1",
                artifact_sha256="a" * 64,
                inference_running=state is ActualState.RUNNING,
                updated_at=now_utc(),
            )
        )

    row = db.get_actual(DEVICE)
    assert row.actual_state is ActualState.RUNNING
    assert row.inference_running is True


def test_hardware_metadata_round_trips(db):
    enrolled(db)
    db.set_actual(
        ActualDeploymentRow(
            device_id=DEVICE,
            observed_generation=1,
            actual_state=ActualState.RUNNING,
            hardware={"gpu": "Orin", "jetpack": "6.0", "temp_c": 41.5},
            updated_at=now_utc(),
        )
    )

    assert db.get_actual(DEVICE).hardware["gpu"] == "Orin"


def test_a_failure_message_is_preserved(db):
    enrolled(db)
    db.set_actual(
        ActualDeploymentRow(
            device_id=DEVICE,
            observed_generation=2,
            actual_state=ActualState.FAILED,
            message="expected sha256 aaa, got bbb",
            updated_at=now_utc(),
        )
    )

    assert "expected sha256" in db.get_actual(DEVICE).message


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------


def event(event_type: EventType, *, device_id: str | None = DEVICE, generation: int | None = None):
    return AuditEventRow(
        event_id=str(uuid.uuid4()),
        timestamp=now_utc(),
        device_id=device_id,
        event_type=event_type,
        generation=generation,
        details={"actor": "operator"},
    )


def test_events_come_back_newest_first_with_a_stable_tie_break(db):
    """Several events share a timestamp on any fast operation. Without the rowid
    tie-break the dashboard would reshuffle them between polls, which reads as
    events appearing and disappearing."""
    enrolled(db)
    types = [
        EventType.DEVICE_REGISTERED,
        EventType.DEPLOYMENT_REQUESTED,
        EventType.STOP_REQUESTED,
        EventType.REVOKE_REQUESTED,
    ]
    for event_type in types:
        db.append_event(event(event_type))

    first = [e.event_type for e in db.list_events(DEVICE)]
    second = [e.event_type for e in db.list_events(DEVICE)]

    assert first == second, "ordering is not stable across reads"
    assert first[0] is EventType.REVOKE_REQUESTED


def test_events_filter_by_device(db):
    enrolled(db)
    enrolled(db, "other-device")
    db.append_event(event(EventType.DEPLOYMENT_REQUESTED))
    db.append_event(event(EventType.DEPLOYMENT_REQUESTED, device_id="other-device"))

    assert len(db.list_events(DEVICE)) == 1
    assert len(db.list_events()) == 2


def test_events_filter_by_type(db):
    enrolled(db)
    db.append_event(event(EventType.DEPLOYMENT_REQUESTED))
    db.append_event(event(EventType.DEVICE_STATE_CHANGED))

    found = db.list_events(DEVICE, event_types=[EventType.DEPLOYMENT_REQUESTED])

    assert [e.event_type for e in found] == [EventType.DEPLOYMENT_REQUESTED]


def test_the_limit_is_honoured(db):
    enrolled(db)
    for _ in range(10):
        db.append_event(event(EventType.DEVICE_STATE_CHANGED))

    assert len(db.list_events(DEVICE, limit=3)) == 3


def test_audit_survives_the_device_it_describes(db):
    """`audit_event.device_id` deliberately carries no foreign key. An audit trail
    that is erased by deleting its subject is not an audit trail -- and "who
    deleted this device" is exactly the question it has to answer afterwards.
    """
    enrolled(db)
    db.append_event(event(EventType.REVOKE_REQUESTED, generation=3))

    db.delete_device(DEVICE)

    remaining = db.list_events(DEVICE)
    assert len(remaining) == 1
    assert remaining[0].generation == 3


def test_deleting_a_device_clears_its_operational_rows(db):
    """The converse of the rule above: state cascades, the record does not."""
    enrolled(db)
    db.set_desired(deploy())
    db.set_actual(
        ActualDeploymentRow(
            device_id=DEVICE, observed_generation=1, actual_state=ActualState.RUNNING
        )
    )

    db.delete_device(DEVICE)

    assert db.get_device(DEVICE) is None
    assert db.get_desired(DEVICE) is None
    assert db.get_actual(DEVICE) is None
    assert db.list_history(DEVICE) == []


# --------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------


def token_row(token_id: str = "t1", *, device_id: str = DEVICE):
    from lighthouse.repositories import DeviceTokenRow

    return DeviceTokenRow(
        token_id=token_id,
        device_id=device_id,
        token_sha256="d" * 64,
        created_at=now_utc(),
        label="bench",
    )


def test_a_token_is_found_by_its_public_id(db):
    enrolled(db)
    db.create_token(token_row())

    found = db.get_token("t1")

    assert found is not None and found.device_id == DEVICE
    assert found.active is True


def test_an_unknown_token_id_is_none_not_an_error(db):
    assert db.get_token("nope") is None


def test_revocation_is_reported_once(db):
    """The second call returning False is what lets the API answer 404 instead of
    confirming a revocation it did not perform."""
    enrolled(db)
    db.create_token(token_row())

    assert db.revoke_token("t1") is True
    assert db.revoke_token("t1") is False


def test_a_revoked_token_is_still_readable_but_inactive(db):
    """Kept rather than deleted, so audit can distinguish a revoked credential
    from one that never existed."""
    enrolled(db)
    db.create_token(token_row())
    db.revoke_token("t1")

    found = db.get_token("t1")
    assert found is not None
    assert found.active is False
    assert found.revoked_at is not None


def test_several_tokens_can_be_active_at_once(db):
    """Rotation with no downtime depends on this."""
    enrolled(db)
    db.create_token(token_row("t1"))
    db.create_token(token_row("t2"))

    assert sum(1 for t in db.list_tokens(DEVICE) if t.active) == 2


def test_marking_use_does_not_disturb_anything_else(db):
    enrolled(db)
    db.create_token(token_row())

    db.mark_token_used("t1")

    found = db.get_token("t1")
    assert found.last_used_at is not None
    assert found.active is True


def test_tokens_cannot_outlive_their_device(db):
    """Cascade, enforced by `PRAGMA foreign_keys = ON` -- which SQLite defaults to
    *off*, so this asserts the pragma is actually applied on every connection."""
    enrolled(db)
    db.create_token(token_row())

    db.delete_device(DEVICE)

    assert db.get_token("t1") is None


def test_a_token_for_an_unknown_device_is_refused(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.create_token(token_row(device_id="never-registered"))


# --------------------------------------------------------------------------
# Artifact cache
# --------------------------------------------------------------------------


def cache_row(cache_key: str = "m-1/v-1", *, version: str = "1"):
    return ArtifactCacheRow(
        cache_key=cache_key,
        model_name="fashion-cnn",
        model_version=version,
        model_id=cache_key.split("/")[0],
        version_uuid=cache_key.split("/")[1],
        packaging="mlflow-onnx",
        source_uri="s3a://bucket/prefix",
        created_at=now_utc(),
    )


def test_claiming_a_new_artifact_succeeds_once(db):
    """The concurrency gate on materialization. Whoever wins the insert streams the
    bytes; the losers wait. Without it two deployment requests for the same new
    version would both pull several MB from object storage."""
    assert db.claim_artifact(cache_row()) is True
    assert db.claim_artifact(cache_row()) is False


def test_a_claimed_artifact_starts_pending_and_unready(db):
    db.claim_artifact(cache_row())

    row = db.get_artifact("m-1/v-1")

    assert row.status == "PENDING"
    assert row.is_ready is False
    assert row.sha256 is None


def test_only_one_of_many_concurrent_claims_wins(db):
    """The same race as generation allocation, from the other direction."""
    barrier = threading.Barrier(8)
    won: list[bool] = []
    lock = threading.Lock()

    def worker():
        barrier.wait()
        result = db.claim_artifact(cache_row())
        with lock:
            won.append(result)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert won.count(True) == 1, f"{won.count(True)} threads all thought they had the claim"


def test_completing_an_artifact_makes_it_ready(db):
    db.claim_artifact(cache_row())
    row = db.get_artifact("m-1/v-1")
    row.status = "READY"
    row.sha256 = "e" * 64
    row.size_bytes = 4096
    row.entrypoint = "model.onnx"
    row.cache_path = "/data/cache/m-1/v-1.tar.gz"
    row.completed_at = now_utc()

    db.update_artifact(row)

    stored = db.get_artifact("m-1/v-1")
    assert stored.is_ready is True
    assert stored.sha256 == "e" * 64
    assert stored.entrypoint == "model.onnx"


def test_a_failed_artifact_keeps_its_error(db):
    """The operator has to learn *why* a deployment never became ready, and the
    device will never know."""
    db.claim_artifact(cache_row())
    row = db.get_artifact("m-1/v-1")
    row.status = "FAILED"
    row.error = "registry unavailable: 503"

    db.update_artifact(row)

    stored = db.get_artifact("m-1/v-1")
    assert stored.is_ready is False
    assert "503" in stored.error


def test_artifacts_are_listed_least_recently_used_first(db):
    """Eviction order comes straight off this query, so the ordering is the
    policy."""
    for key in ("m-1/v-1", "m-1/v-2", "m-1/v-3"):
        db.claim_artifact(cache_row(key))
    db.touch_artifact("m-1/v-1")

    order = [r.cache_key for r in db.list_artifacts()]

    assert order[-1] == "m-1/v-1", f"most recently used is not last: {order}"


def test_live_desired_state_pins_its_artifact(db):
    """Eviction must never remove one of these. A device returning after a long
    absence would otherwise be told to fetch bytes the control plane just
    deleted -- and it has no other way to get them."""
    enrolled(db)
    db.set_desired(deploy(version="7"))

    assert db.referenced_cache_keys() == {"m-1/v-7"}


def test_nothing_is_pinned_before_a_deployment(db):
    enrolled(db)

    assert db.referenced_cache_keys() == set()


def test_deleting_an_artifact_row_is_idempotent(db):
    db.claim_artifact(cache_row())
    db.delete_artifact("m-1/v-1")
    db.delete_artifact("m-1/v-1")

    assert db.get_artifact("m-1/v-1") is None


def test_the_cache_outlives_the_device_that_needed_it(db):
    """No foreign key on purpose: the bytes are shared fleet-wide and keyed by
    registry lineage, so deleting one device must not invalidate another's cache
    hit."""
    enrolled(db)
    db.set_desired(deploy())
    db.claim_artifact(cache_row())

    db.delete_device(DEVICE)

    assert db.get_artifact("m-1/v-1") is not None


# --------------------------------------------------------------------------
# Durability
# --------------------------------------------------------------------------


def test_state_survives_reopening_the_database(tmp_path):
    """A CAI Application restarts. If enrollment did not survive it, every device
    in the fleet would need re-registering by hand."""
    path = tmp_path / "lighthouse.db"
    first = SqliteStore(path)
    enrolled(first)
    first.set_desired(deploy(version="4"))
    first.close()

    second = SqliteStore(path)
    try:
        assert second.get_device(DEVICE) is not None
        assert second.get_desired(DEVICE).model_version == "4"
        assert second.get_desired(DEVICE).generation == 1
    finally:
        second.close()


def test_opening_an_existing_database_does_not_reset_it(tmp_path):
    """Schema init runs on every open, so every statement in it has to be
    IF NOT EXISTS -- otherwise a restart either crashes or, far worse, succeeds by
    dropping the fleet."""
    path = tmp_path / "lighthouse.db"
    first = SqliteStore(path)
    enrolled(first)
    first.close()

    for _ in range(3):
        store = SqliteStore(path)
        assert store.device_count() == 1
        store.close()


def test_the_parent_directory_is_created(tmp_path):
    """The CAI project filesystem will not have `/data/lighthouse` pre-made."""
    store = SqliteStore(tmp_path / "deep" / "nested" / "lighthouse.db")
    try:
        assert store.device_count() == 0
    finally:
        store.close()
