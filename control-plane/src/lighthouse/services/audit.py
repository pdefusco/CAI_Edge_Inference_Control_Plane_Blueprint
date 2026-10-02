"""Audit trail (spec SS16).

Two jobs. The first is dull and necessary: append an immutable event for every
state change, with the generation it belongs to, so "who deployed what to which
device, when" is answerable after the fact.

The second is the interesting one. **Rollback is classified here, not declared by
the caller.** Spec SS13's deployment request has no "this is a rollback" field and
should not get one: an operator who redeploys v6 over v7 has performed a rollback
whether or not they say so, and one who mislabels a fresh deployment as a rollback
would corrupt the record. Comparing the request against `deployment_history` makes
the label a fact about what happened rather than a claim about intent.
"""

from __future__ import annotations

import logging
from typing import Any

from lighthouse_contracts import AuditEventView, DesiredState, EventType

from ..repositories import AuditEventRow, Store
from ..util import new_id, now_utc

log = logging.getLogger(__name__)

# Token material must never reach the audit table. Only the public token_id is
# ever recorded, so these keys are dropped if a caller passes them by accident.
_FORBIDDEN_DETAIL_KEYS = frozenset({"token", "secret", "password", "authorization", "private_key"})


class AuditService:
    def __init__(self, store: Store) -> None:
        self._store = store

    def record(
        self,
        event_type: EventType,
        *,
        device_id: str | None = None,
        generation: int | None = None,
        **details: Any,
    ) -> AuditEventRow:
        """Append one event. Never raises into the caller's path.

        A failed audit write must not fail the operation it was describing --
        losing the record of a successful revoke is bad, but refusing the revoke
        because the record failed is worse.
        """
        row = AuditEventRow(
            event_id=new_id("evt"),
            timestamp=now_utc(),
            event_type=event_type,
            device_id=device_id,
            generation=generation,
            details={k: v for k, v in details.items() if k not in _FORBIDDEN_DETAIL_KEYS},
        )
        try:
            self._store.append_event(row)
        except Exception:  # pragma: no cover - audit must not break callers
            log.exception("failed to append audit event %s for %s", event_type, device_id)
        return row

    def classify_deployment(
        self,
        device_id: str,
        model_name: str,
        model_version: str,
    ) -> EventType:
        """Decide what kind of deployment change this request represents.

        A rollback is "returning this device to a (model, version) pair it was
        previously told to run, which is not what it is running now". That is
        derivable, auditable, and impossible for the caller to fake.
        """
        history = self._store.list_history(device_id, limit=200)
        deployed = [
            (h.model_name, h.model_version)
            for h in history
            if h.desired_state is DesiredState.RUNNING and h.model_name is not None
        ]
        target = (model_name, model_version)

        if not deployed:
            return EventType.DEPLOYMENT_REQUESTED

        # history is newest-first, so deployed[0] is the current running intent.
        if deployed[0] == target:
            # Re-asserting the same model: still a request (generation advances,
            # the device no-ops), but not a version change and not a rollback.
            return EventType.DEPLOYMENT_REQUESTED

        if target in deployed:
            return EventType.DEPLOYMENT_ROLLED_BACK

        return EventType.MODEL_VERSION_CHANGED

    def list_events(
        self,
        device_id: str | None = None,
        limit: int = 100,
        event_types: list[EventType] | None = None,
    ) -> list[AuditEventView]:
        rows = self._store.list_events(device_id=device_id, limit=limit, event_types=event_types)
        return [
            AuditEventView(
                event_id=r.event_id,
                timestamp=r.timestamp,
                device_id=r.device_id,
                event_type=r.event_type,
                generation=r.generation,
                details=r.details,
            )
            for r in rows
        ]
