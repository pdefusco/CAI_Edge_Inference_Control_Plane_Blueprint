"""SQLite implementation of the Store protocol.

Two things here are load-bearing rather than incidental:

1. **Thread-local connections.** FastAPI runs sync endpoint functions in a
   threadpool, and a SQLite connection may not be shared across threads. One
   connection per thread, opened lazily.

2. **`BEGIN IMMEDIATE` for writes.** Generation allocation is a read-modify-write
   (`SELECT MAX`, then insert N+1). SQLite's default deferred transactions would
   let two operator requests both read N and both write N+1, silently losing one
   desired-state change -- the exact failure the generation counter exists to
   prevent. IMMEDIATE takes the write lock up front so they serialize.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any, Iterator

from lighthouse_contracts import ActualState, DesiredState, EventType

from ..util import from_iso, now_utc, to_iso
from .base import DeviceExists, DeviceUnknown, Store
from .rows import (
    ActualDeploymentRow,
    ArtifactCacheRow,
    AuditEventRow,
    DeploymentHistoryRow,
    DesiredDeploymentRow,
    DeviceRow,
    DeviceTokenRow,
)


def _load_schema() -> str:
    return resources.files("lighthouse.repositories").joinpath("schema.sql").read_text()


class SqliteStore(Store):
    """Store backed by a single SQLite file."""

    def __init__(self, db_path: Path | str) -> None:
        self._path = str(db_path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # An in-memory database lives only as long as its connection, so a
        # thread-local pool would give each thread an empty database. Tests use
        # :memory:, so share one connection and serialize with a lock instead.
        self._shared_conn: sqlite3.Connection | None = None
        self._shared_lock = threading.RLock()
        if self._path == ":memory:":
            self._shared_conn = self._connect()
        self._init_schema()

    # -- connection management -------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        if self._path != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")
        return conn

    @property
    def _conn(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        with self._write() as conn:
            conn.executescript(_load_schema())

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """A serialized write transaction.

        IMMEDIATE acquires the write lock at BEGIN rather than at first write, so
        concurrent generation allocation cannot interleave.
        """
        conn = self._conn
        with self._shared_lock:
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _read(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        if self._shared_conn is not None:
            self._shared_conn.close()
            self._shared_conn = None
            return
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # -- devices ---------------------------------------------------------

    def create_device(
        self, device_id: str, display_name: str | None, platform: str | None
    ) -> DeviceRow:
        ts = now_utc()
        try:
            with self._write() as conn:
                conn.execute(
                    "INSERT INTO device (device_id, display_name, platform, registered_at) "
                    "VALUES (?, ?, ?, ?)",
                    (device_id, display_name, platform, to_iso(ts)),
                )
                # Seed desired state at generation 0 / STOPPED. A freshly
                # enrolled device is explicitly "told to run nothing" rather
                # than having no desired row at all, which keeps every read path
                # free of a None special case.
                conn.execute(
                    "INSERT INTO desired_deployment "
                    "(device_id, generation, desired_state, updated_at) VALUES (?, 0, ?, ?)",
                    (device_id, DesiredState.STOPPED.value, to_iso(ts)),
                )
                conn.execute(
                    "INSERT INTO actual_deployment "
                    "(device_id, observed_generation, actual_state, updated_at) "
                    "VALUES (?, 0, ?, ?)",
                    (device_id, ActualState.UNKNOWN.value, to_iso(ts)),
                )
        except sqlite3.IntegrityError as exc:
            raise DeviceExists(f"device already registered: {device_id}") from exc
        return DeviceRow(
            device_id=device_id,
            display_name=display_name,
            platform=platform,
            registered_at=ts,
        )

    def get_device(self, device_id: str) -> DeviceRow | None:
        row = self._read().execute(
            "SELECT * FROM device WHERE device_id = ?", (device_id,)
        ).fetchone()
        return _device(row) if row else None

    def list_devices(self) -> list[DeviceRow]:
        rows = self._read().execute("SELECT * FROM device ORDER BY device_id").fetchall()
        return [_device(r) for r in rows]

    def touch_device(self, device_id: str) -> None:
        with self._write() as conn:
            conn.execute(
                "UPDATE device SET last_seen = ? WHERE device_id = ?",
                (to_iso(now_utc()), device_id),
            )

    def delete_device(self, device_id: str) -> None:
        with self._write() as conn:
            conn.execute("DELETE FROM device WHERE device_id = ?", (device_id,))

    def device_count(self) -> int:
        return int(self._read().execute("SELECT COUNT(*) FROM device").fetchone()[0])

    # -- desired state ---------------------------------------------------

    def get_desired(self, device_id: str) -> DesiredDeploymentRow | None:
        row = self._read().execute(
            "SELECT * FROM desired_deployment WHERE device_id = ?", (device_id,)
        ).fetchone()
        return _desired(row) if row else None

    def set_desired(self, row: DesiredDeploymentRow) -> DesiredDeploymentRow:
        ts = now_utc()
        with self._write() as conn:
            exists = conn.execute(
                "SELECT generation FROM desired_deployment WHERE device_id = ?",
                (row.device_id,),
            ).fetchone()
            if exists is None:
                raise DeviceUnknown(f"no such device: {row.device_id}")

            # Allocate inside the transaction. MAX over both the current row and
            # history, so a generation is never reused even if a history row
            # outlives its desired row.
            current = int(exists["generation"])
            hist = conn.execute(
                "SELECT COALESCE(MAX(generation), 0) FROM deployment_history WHERE device_id = ?",
                (row.device_id,),
            ).fetchone()[0]
            generation = max(current, int(hist)) + 1

            conn.execute(
                "UPDATE desired_deployment SET generation = ?, desired_state = ?, "
                "model_name = ?, model_version = ?, registry_artifact_uri = ?, "
                "artifact_sha256 = ?, artifact_format = ?, model_id = ?, version_uuid = ?, "
                "updated_at = ? WHERE device_id = ?",
                (
                    generation,
                    row.desired_state.value,
                    row.model_name,
                    row.model_version,
                    row.registry_artifact_uri,
                    row.artifact_sha256,
                    row.artifact_format,
                    row.model_id,
                    row.version_uuid,
                    to_iso(ts),
                    row.device_id,
                ),
            )
            conn.execute(
                "INSERT INTO deployment_history "
                "(device_id, generation, desired_state, model_name, model_version, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    row.device_id,
                    generation,
                    row.desired_state.value,
                    row.model_name,
                    row.model_version,
                    to_iso(ts),
                ),
            )
        row.generation = generation
        row.updated_at = ts
        return row

    def update_desired_digest(self, device_id: str, sha256: str) -> None:
        with self._write() as conn:
            cur = conn.execute(
                "UPDATE desired_deployment SET artifact_sha256 = ? WHERE device_id = ?",
                (sha256, device_id),
            )
            if cur.rowcount == 0:
                raise DeviceUnknown(f"no such device: {device_id}")

    def list_history(self, device_id: str, limit: int = 50) -> list[DeploymentHistoryRow]:
        rows = self._read().execute(
            "SELECT * FROM deployment_history WHERE device_id = ? "
            "ORDER BY generation DESC LIMIT ?",
            (device_id, limit),
        ).fetchall()
        return [
            DeploymentHistoryRow(
                id=r["id"],
                device_id=r["device_id"],
                generation=r["generation"],
                desired_state=DesiredState(r["desired_state"]),
                model_name=r["model_name"],
                model_version=r["model_version"],
                created_at=from_iso(r["created_at"]),  # type: ignore[arg-type]
            )
            for r in rows
        ]

    # -- actual state ----------------------------------------------------

    def get_actual(self, device_id: str) -> ActualDeploymentRow | None:
        row = self._read().execute(
            "SELECT * FROM actual_deployment WHERE device_id = ?", (device_id,)
        ).fetchone()
        return _actual(row) if row else None

    def set_actual(self, row: ActualDeploymentRow) -> None:
        with self._write() as conn:
            conn.execute(
                "INSERT INTO actual_deployment (device_id, observed_generation, actual_state, "
                "model_name, model_version, artifact_sha256, inference_running, message, "
                "hardware_json, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(device_id) DO UPDATE SET observed_generation = excluded.observed_generation, "
                "actual_state = excluded.actual_state, model_name = excluded.model_name, "
                "model_version = excluded.model_version, artifact_sha256 = excluded.artifact_sha256, "
                "inference_running = excluded.inference_running, message = excluded.message, "
                "hardware_json = excluded.hardware_json, updated_at = excluded.updated_at",
                (
                    row.device_id,
                    row.observed_generation,
                    row.actual_state.value,
                    row.model_name,
                    row.model_version,
                    row.artifact_sha256,
                    1 if row.inference_running else 0,
                    row.message,
                    json.dumps(row.hardware or {}),
                    to_iso(row.updated_at or now_utc()),
                ),
            )

    # -- audit -----------------------------------------------------------

    def append_event(self, row: AuditEventRow) -> None:
        with self._write() as conn:
            conn.execute(
                "INSERT INTO audit_event (event_id, timestamp, device_id, event_type, "
                "generation, details) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    row.event_id,
                    to_iso(row.timestamp),
                    row.device_id,
                    row.event_type.value,
                    row.generation,
                    json.dumps(row.details or {}),
                ),
            )

    def list_events(
        self,
        device_id: str | None = None,
        limit: int = 100,
        event_types: list[EventType] | None = None,
    ) -> list[AuditEventRow]:
        sql = "SELECT * FROM audit_event"
        clauses: list[str] = []
        params: list[Any] = []
        if device_id is not None:
            clauses.append("device_id = ?")
            params.append(device_id)
        if event_types:
            placeholders = ",".join("?" for _ in event_types)
            clauses.append(f"event_type IN ({placeholders})")
            params.extend(t.value for t in event_types)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        # Tie-break on rowid: several events can share a timestamp, and without a
        # stable secondary sort the dashboard would shuffle them between polls.
        sql += " ORDER BY timestamp DESC, rowid DESC LIMIT ?"
        params.append(limit)
        rows = self._read().execute(sql, params).fetchall()
        return [
            AuditEventRow(
                event_id=r["event_id"],
                timestamp=from_iso(r["timestamp"]),  # type: ignore[arg-type]
                device_id=r["device_id"],
                event_type=EventType(r["event_type"]),
                generation=r["generation"],
                details=json.loads(r["details"] or "{}"),
            )
            for r in rows
        ]

    # -- device tokens ---------------------------------------------------

    def create_token(self, row: DeviceTokenRow) -> None:
        with self._write() as conn:
            conn.execute(
                "INSERT INTO device_token (token_id, device_id, token_sha256, created_at, label) "
                "VALUES (?, ?, ?, ?, ?)",
                (row.token_id, row.device_id, row.token_sha256, to_iso(row.created_at), row.label),
            )

    def get_token(self, token_id: str) -> DeviceTokenRow | None:
        row = self._read().execute(
            "SELECT * FROM device_token WHERE token_id = ?", (token_id,)
        ).fetchone()
        return _token(row) if row else None

    def list_tokens(self, device_id: str) -> list[DeviceTokenRow]:
        rows = self._read().execute(
            "SELECT * FROM device_token WHERE device_id = ? ORDER BY created_at DESC",
            (device_id,),
        ).fetchall()
        return [_token(r) for r in rows]

    def mark_token_used(self, token_id: str) -> None:
        with self._write() as conn:
            conn.execute(
                "UPDATE device_token SET last_used_at = ? WHERE token_id = ?",
                (to_iso(now_utc()), token_id),
            )

    def revoke_token(self, token_id: str) -> bool:
        with self._write() as conn:
            cur = conn.execute(
                "UPDATE device_token SET revoked_at = ? WHERE token_id = ? AND revoked_at IS NULL",
                (to_iso(now_utc()), token_id),
            )
            return cur.rowcount > 0

    # -- artifact cache --------------------------------------------------

    def get_artifact(self, cache_key: str) -> ArtifactCacheRow | None:
        row = self._read().execute(
            "SELECT * FROM artifact_cache WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        return _artifact(row) if row else None

    def claim_artifact(self, row: ArtifactCacheRow) -> bool:
        try:
            with self._write() as conn:
                conn.execute(
                    "INSERT INTO artifact_cache (cache_key, model_name, model_version, model_id, "
                    "version_uuid, status, packaging, source_uri, created_at, last_access) "
                    "VALUES (?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?)",
                    (
                        row.cache_key,
                        row.model_name,
                        row.model_version,
                        row.model_id,
                        row.version_uuid,
                        row.packaging,
                        row.source_uri,
                        to_iso(row.created_at or now_utc()),
                        to_iso(now_utc()),
                    ),
                )
            return True
        except sqlite3.IntegrityError:
            # Someone else is already materializing this version.
            return False

    def update_artifact(self, row: ArtifactCacheRow) -> None:
        with self._write() as conn:
            conn.execute(
                "UPDATE artifact_cache SET status = ?, sha256 = ?, size_bytes = ?, "
                "packaging = ?, entrypoint = ?, cache_path = ?, source_uri = ?, error = ?, "
                "completed_at = ?, last_access = ? WHERE cache_key = ?",
                (
                    row.status,
                    row.sha256,
                    row.size_bytes,
                    row.packaging,
                    row.entrypoint,
                    row.cache_path,
                    row.source_uri,
                    row.error,
                    to_iso(row.completed_at),
                    to_iso(row.last_access or now_utc()),
                    row.cache_key,
                ),
            )

    def list_artifacts(self) -> list[ArtifactCacheRow]:
        rows = self._read().execute(
            "SELECT * FROM artifact_cache ORDER BY last_access ASC"
        ).fetchall()
        return [_artifact(r) for r in rows]

    def touch_artifact(self, cache_key: str) -> None:
        with self._write() as conn:
            conn.execute(
                "UPDATE artifact_cache SET last_access = ? WHERE cache_key = ?",
                (to_iso(now_utc()), cache_key),
            )

    def delete_artifact(self, cache_key: str) -> None:
        with self._write() as conn:
            conn.execute("DELETE FROM artifact_cache WHERE cache_key = ?", (cache_key,))

    def referenced_cache_keys(self) -> set[str]:
        rows = self._read().execute(
            "SELECT model_id, version_uuid FROM desired_deployment "
            "WHERE model_id IS NOT NULL AND version_uuid IS NOT NULL"
        ).fetchall()
        return {f"{r['model_id']}/{r['version_uuid']}" for r in rows}


# -- row mapping ---------------------------------------------------------


def _device(r: sqlite3.Row) -> DeviceRow:
    return DeviceRow(
        device_id=r["device_id"],
        display_name=r["display_name"],
        platform=r["platform"],
        registered_at=from_iso(r["registered_at"]),
        last_seen=from_iso(r["last_seen"]),
    )


def _desired(r: sqlite3.Row) -> DesiredDeploymentRow:
    return DesiredDeploymentRow(
        device_id=r["device_id"],
        generation=r["generation"],
        desired_state=DesiredState(r["desired_state"]),
        model_name=r["model_name"],
        model_version=r["model_version"],
        registry_artifact_uri=r["registry_artifact_uri"],
        artifact_sha256=r["artifact_sha256"],
        artifact_format=r["artifact_format"],
        model_id=r["model_id"],
        version_uuid=r["version_uuid"],
        updated_at=from_iso(r["updated_at"]),
    )


def _actual(r: sqlite3.Row) -> ActualDeploymentRow:
    return ActualDeploymentRow(
        device_id=r["device_id"],
        observed_generation=r["observed_generation"],
        actual_state=ActualState(r["actual_state"]),
        model_name=r["model_name"],
        model_version=r["model_version"],
        artifact_sha256=r["artifact_sha256"],
        inference_running=bool(r["inference_running"]),
        message=r["message"],
        hardware=json.loads(r["hardware_json"] or "{}"),
        updated_at=from_iso(r["updated_at"]),
    )


def _token(r: sqlite3.Row) -> DeviceTokenRow:
    return DeviceTokenRow(
        token_id=r["token_id"],
        device_id=r["device_id"],
        token_sha256=r["token_sha256"],
        created_at=from_iso(r["created_at"]),  # type: ignore[arg-type]
        last_used_at=from_iso(r["last_used_at"]),
        revoked_at=from_iso(r["revoked_at"]),
        label=r["label"],
    )


def _artifact(r: sqlite3.Row) -> ArtifactCacheRow:
    return ArtifactCacheRow(
        cache_key=r["cache_key"],
        model_name=r["model_name"],
        model_version=r["model_version"],
        model_id=r["model_id"],
        version_uuid=r["version_uuid"],
        status=r["status"],
        sha256=r["sha256"],
        size_bytes=r["size_bytes"],
        packaging=r["packaging"],
        entrypoint=r["entrypoint"],
        cache_path=r["cache_path"],
        source_uri=r["source_uri"],
        error=r["error"],
        created_at=from_iso(r["created_at"]),
        completed_at=from_iso(r["completed_at"]),
        last_access=from_iso(r["last_access"]),
    )
