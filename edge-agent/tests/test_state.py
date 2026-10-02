"""The agent's only memory, tested for the thing that actually happens to it.

A Jetson on a shelf loses power mid-write. The requirement is not that no state is
lost -- it is that the agent still *starts*, because an agent that refuses to boot
because its state file is truncated needs a human to drive to it.
"""

from __future__ import annotations

import json

from lighthouse_contracts import ActualState

from keeper.state import AgentState, DeployedModel, StateStore


def store_at(tmp_path):
    return StateStore(tmp_path / "nested" / "state.json")


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------


def test_missing_file_loads_a_default(tmp_path):
    state = store_at(tmp_path).load()

    assert state.observed_generation == 0
    assert state.actual_state is ActualState.UNKNOWN
    assert state.model is None


def test_round_trip_preserves_everything_that_matters(tmp_path):
    store = store_at(tmp_path)
    saved = AgentState(
        observed_generation=42,
        actual_state=ActualState.RUNNING,
        model=DeployedModel(
            name="fashion-cnn",
            version="3",
            sha256="a" * 64,
            path="/var/lib/keeper/models/fashion-cnn/3",
            entrypoint="/var/lib/keeper/models/fashion-cnn/3/model.onnx",
        ),
        inference_running=True,
        message=None,
    )

    store.save(saved)
    loaded = store.load()

    assert loaded.observed_generation == 42
    assert loaded.actual_state is ActualState.RUNNING
    assert loaded.inference_running is True
    assert loaded.model == saved.model


def test_save_creates_the_parent_directory(tmp_path):
    store = store_at(tmp_path)
    store.save(AgentState())

    assert (tmp_path / "nested" / "state.json").is_file()


# --------------------------------------------------------------------------
# Corruption tolerance. This is the point of the file.
# --------------------------------------------------------------------------


def test_truncated_file_loads_as_default_instead_of_raising(tmp_path):
    store = store_at(tmp_path)
    store.save(AgentState(observed_generation=9, actual_state=ActualState.RUNNING))
    path = tmp_path / "nested" / "state.json"
    body = path.read_text()
    path.write_text(body[: len(body) // 2])

    state = store.load()

    # Forgetting everything is the correct outcome: UNKNOWN forces a full
    # reconcile, and the next poll re-establishes the truth from the control
    # plane. Refusing to start would not.
    assert state.actual_state is ActualState.UNKNOWN
    assert state.observed_generation == 0


def test_garbage_file_loads_as_default(tmp_path):
    store = store_at(tmp_path)
    store._path.parent.mkdir(parents=True, exist_ok=True)
    store._path.write_bytes(b"\x00\x01\x02 not json at all")

    assert store.load().actual_state is ActualState.UNKNOWN


def test_unknown_actual_state_degrades_to_unknown(tmp_path):
    """A state file written by a newer agent, then downgraded.

    Mapping an unrecognised value to UNKNOWN forces a reconcile; mapping it to
    RUNNING would have the device claim to be serving something it is not.
    """
    store = store_at(tmp_path)
    store._path.parent.mkdir(parents=True, exist_ok=True)
    store._path.write_text(
        json.dumps({"observed_generation": 5, "actual_state": "TRANSCENDENT", "schema_version": 1})
    )

    state = store.load()

    assert state.actual_state is ActualState.UNKNOWN


def test_monotonic_retry_deadline_is_not_restored(tmp_path):
    """`next_retry_monotonic` is meaningless across a reboot.

    `time.monotonic()` restarts from an arbitrary origin, so a persisted deadline
    could sit far in the future and silently block retries for the uptime of the
    device.
    """
    store = store_at(tmp_path)
    store.save(
        AgentState(
            actual_state=ActualState.FAILED,
            failed_generation=4,
            failure_count=3,
            next_retry_monotonic=9_999_999.0,
        )
    )

    state = store.load()

    assert state.next_retry_monotonic == 0.0
    # The failure history itself is kept -- backoff should not reset to zero just
    # because the agent restarted, or a crash-looping device hammers the server.
    assert state.failure_count == 3
    assert state.failed_generation == 4


# --------------------------------------------------------------------------
# matches()
# --------------------------------------------------------------------------


def _running(sha: str = "a" * 64) -> AgentState:
    return AgentState(
        observed_generation=1,
        actual_state=ActualState.RUNNING,
        model=DeployedModel(name="fashion-cnn", version="1", sha256=sha, path="/x"),
        inference_running=True,
    )


def test_matches_requires_the_digest_too():
    """Name and version are mutable labels in a registry; the digest is the only
    identity that cannot be repointed underneath the device."""
    state = _running()

    assert state.matches("fashion-cnn", "1", "a" * 64)
    assert not state.matches("fashion-cnn", "1", "b" * 64)
    assert not state.matches("fashion-cnn", "2", "a" * 64)
    assert not state.matches("other", "1", "a" * 64)


def test_matches_is_false_with_no_model():
    assert not AgentState().matches("fashion-cnn", "1", "a" * 64)


# --------------------------------------------------------------------------
# Atomicity
# --------------------------------------------------------------------------


def test_save_leaves_no_temporary_files(tmp_path):
    store = store_at(tmp_path)
    store.save(AgentState(observed_generation=1))
    store.save(AgentState(observed_generation=2))

    entries = sorted(p.name for p in (tmp_path / "nested").iterdir())
    assert entries == ["state.json"], f"temporary files left behind: {entries}"


def test_an_interrupted_save_cannot_destroy_the_previous_state(tmp_path, monkeypatch):
    """The reason the write goes tmp -> fsync -> rename rather than straight to the
    real path: a crash during serialization must leave the last good state intact,
    not a half-written file where the state used to be."""
    store = store_at(tmp_path)
    store.save(AgentState(observed_generation=7, actual_state=ActualState.RUNNING))

    real_replace = __import__("os").replace

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr("os.replace", boom)
    try:
        store.save(AgentState(observed_generation=8))
    except OSError:
        pass
    monkeypatch.setattr("os.replace", real_replace)

    recovered = store.load()
    assert recovered.observed_generation == 7
    assert recovered.actual_state is ActualState.RUNNING
