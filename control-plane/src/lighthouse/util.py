"""Small shared helpers.

Time handling is centralized here because mixing naive and aware datetimes is the
classic way to get a heartbeat-age calculation that is silently wrong by the
local UTC offset -- which would make a healthy device read as OFFLINE.
Everything in this system is timezone-aware UTC, enforced at the boundary.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone


def now_utc() -> datetime:
    """Current time, timezone-aware, UTC. The only clock this codebase reads."""
    return datetime.now(timezone.utc)


def to_utc(value: datetime) -> datetime:
    """Coerce an incoming datetime to aware UTC.

    A naive datetime is assumed to be UTC rather than local: devices send ISO
    timestamps and an absent offset is far more likely to mean "UTC, formatted
    carelessly" than "the device's local wall clock".
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def to_iso(value: datetime | None) -> str | None:
    """Serialize for storage. None passes through so optional columns stay NULL."""
    if value is None:
        return None
    return to_utc(value).isoformat()


def from_iso(value: str | None) -> datetime | None:
    """Parse a stored timestamp back to aware UTC."""
    if value is None:
        return None
    return to_utc(datetime.fromisoformat(value))


def new_id(prefix: str | None = None) -> str:
    """Opaque identifier for audit events and similar.

    The optional prefix is purely for legibility when reading rows by hand --
    `evt_3f2a…` beats a bare hex blob when scanning an audit table.
    """
    raw = uuid.uuid4().hex
    return f"{prefix}_{raw}" if prefix else raw
