"""Small shared helpers."""

from __future__ import annotations

from datetime import datetime, timezone


def now_utc() -> datetime:
    """Timezone-aware UTC now.

    Always aware, never naive: heartbeat timestamps are compared against server
    time to derive connectivity, and a naive datetime there silently becomes
    whatever the device's local timezone is -- which on a Jetson with no NTP and a
    default timezone is how a healthy device reads as hours stale.
    """
    return datetime.now(timezone.utc)
