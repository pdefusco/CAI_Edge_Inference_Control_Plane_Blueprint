"""Terminal formatting for the operator Makefile targets.

A separate file rather than an inline `python -c` in the Makefile: the fleet view
nests f-strings inside shell quoting inside Make's own escaping, and that is a
place where a missing backslash costs twenty minutes.

Reads JSON on stdin, writes a table. Nothing here talks to the API -- the Makefile
already did, with curl, so what you see is exactly what the dashboard sees.

    curl .../devices | python scripts/show.py fleet
    curl .../devices/<id>/events | python scripts/show.py events
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone


def _pair(state: object, version: object) -> str:
    """Render a state/version pair, e.g. `RUNNING 2` or `STOPPED —`."""
    return f"{state or '—'} {version if version is not None else '—'}"


def _age(stamp: str | None) -> str:
    """Relative age, not a wall clock.

    The API speaks UTC and the terminal running this is usually not in it, so a
    sliced-out `07:15:41` sitting next to local-time log lines reads as a bug. The
    number an operator actually wants from `last_seen` is how stale it is, and that
    needs no timezone at all.
    """
    if not stamp:
        return "never"
    try:
        seen = datetime.fromisoformat(stamp)
    except ValueError:
        return stamp
    if seen.tzinfo is None:
        seen = seen.replace(tzinfo=timezone.utc)
    seconds = max(0, (datetime.now(timezone.utc) - seen).total_seconds())
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m ago"
    return f"{seconds / 3600:.0f}h ago"


def _local(stamp: str) -> str:
    """An ISO-8601 UTC timestamp as a local clock time, for event ordering."""
    try:
        parsed = datetime.fromisoformat(stamp)
    except ValueError:
        return stamp[11:19]
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone().strftime("%H:%M:%S")


def fleet(rows: list[dict]) -> None:
    if not rows:
        print("  no devices enrolled (make enroll)")
        return
    header = (
        f"{'DEVICE':<20}{'LINK':<11}{'GOVERNANCE':<15}"
        f"{'DESIRED':<22}{'ACTUAL':<22}{'ACCEL':<13}LAST SEEN"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        # Desired and actual stay in separate columns here for the same reason they
        # do in the dashboard: collapsing them into one "status" would delete the
        # only interesting information on the line.
        #
        # ACCEL is next to GOVERNANCE because the pair is the thing worth reading:
        # `HEALTHY` + `CPU_ONLY` is a device doing exactly what it was told while
        # failing the acceptance check this fleet exists to pass, and it is the one
        # combination that looks fine device-by-device and wrong across a fleet.
        # The full enum value, unabbreviated -- a `GPU`/`CPU` shorthand invented
        # here would be a third spelling of a judgement that has one home.
        print(
            f"{row['device_id']:<20}"
            f"{row['connectivity']:<11}"
            f"{row['governance_status']:<15}"
            f"{_pair(row.get('desired_state'), row.get('desired_model_version')):<22}"
            f"{_pair(row.get('actual_state'), row.get('actual_model_version')):<22}"
            f"{row.get('acceleration') or 'UNKNOWN':<13}"
            f"{_age(row.get('last_seen'))}"
        )


def events(rows: list[dict]) -> None:
    if not rows:
        print("  no events yet")
        return
    # The API returns newest first, which is right for a dashboard and wrong for a
    # terminal: reversed, the trail reads downward in the order things happened.
    for event in reversed(rows):
        details = event.get("details") or {}
        what = " · ".join(
            str(part)
            for part in (
                f"{details['model_name']}:{details.get('model_version')}"
                if details.get("model_name")
                else None,
                f"{details['previous_state']} → {details['actual_state']}"
                if details.get("previous_state") and details.get("actual_state")
                else details.get("actual_state") or details.get("desired_state"),
                details.get("message"),
            )
            if part
        )
        generation = event.get("generation")
        print(
            f"{_local(event['timestamp'])}  "
            f"gen {'—' if generation is None else generation:<4}  "
            f"{event['event_type']:<30}{what}"
        )


def main() -> int:
    view = sys.argv[1] if len(sys.argv) > 1 else "fleet"
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        print("  the control plane did not return JSON -- is it running?", file=sys.stderr)
        return 1
    if view == "fleet":
        fleet(payload)
    elif view == "events":
        events(payload)
    else:  # pragma: no cover - operator typo
        print(f"unknown view: {view}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
