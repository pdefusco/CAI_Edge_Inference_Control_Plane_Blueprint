"""The agent's local record of what it has actually done.

This file is the agent's only memory. Its correctness matters most at the moment
it is least convenient: after a hard power cut mid-deployment, which on a device
sitting in someone's house is a routine event rather than an edge case. So it is
written atomically -- temp file, fsync, rename -- and a corrupt file is treated as
"I know nothing" rather than as a crash, because an agent that refuses to start
because its state file is truncated is an agent that needs a human in the loop to
recover from a power cut.

`observed_generation` means "the generation I have **converged to**", not "the
generation I have seen". It advances on success and on terminal failure, never
while a deployment is still in flight -- otherwise the control plane would read
mid-download as compliant.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lighthouse_contracts import ActualState

log = logging.getLogger(__name__)

_SCHEMA_VERSION = 1


@dataclass(slots=True)
class DeployedModel:
    """What is unpacked on disk right now."""

    name: str
    version: str
    sha256: str
    path: str
    entrypoint: str | None = None


@dataclass(slots=True)
class AgentState:
    observed_generation: int = 0
    actual_state: ActualState = ActualState.UNKNOWN
    model: DeployedModel | None = None
    inference_running: bool = False
    message: str | None = None

    # Backoff bookkeeping. Keyed by generation so a new instruction starts clean:
    # an operator pushing a corrected version should not wait out a penalty earned
    # by the version it replaces.
    failed_generation: int | None = None
    failure_count: int = 0
    next_retry_monotonic: float = 0.0

    schema_version: int = _SCHEMA_VERSION

    def matches(self, name: str | None, version: str | None, sha256: str | None) -> bool:
        """Is the deployed model exactly the one described?

        The digest is part of the comparison on purpose. Name and version are
        mutable labels in a registry, so "fashion-cnn v2" on disk is not
        necessarily the same bytes as "fashion-cnn v2" in today's desired state.
        """
        if self.model is None:
            return False
        if self.model.name != name or self.model.version != version:
            return False
        if sha256 is not None and self.model.sha256 != sha256:
            return False
        return True


class StateStore:
    """Load/save `AgentState` as a single JSON file."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> AgentState:
        if not self._path.is_file():
            return AgentState()
        try:
            raw = json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            # Start from UNKNOWN and re-derive by reconciling, rather than
            # refusing to run. The control plane is the authority anyway.
            log.warning("state file %s unreadable (%s); starting from UNKNOWN", self._path, exc)
            return AgentState()
        return _from_dict(raw)

    def save(self, state: AgentState) -> None:
        payload = asdict(state)
        payload["actual_state"] = state.actual_state.value
        payload["schema_version"] = _SCHEMA_VERSION
        tmp = self._path.with_suffix(".tmp")
        try:
            with open(tmp, "w") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.flush()
                # fsync before rename: without it a power cut can leave the
                # directory entry pointing at unwritten data, which is the one
                # failure this whole dance exists to prevent.
                os.fsync(fh.fileno())
            os.replace(tmp, self._path)
        except OSError as exc:  # pragma: no cover - disk failure
            log.error("could not persist state to %s: %s", self._path, exc)
            tmp.unlink(missing_ok=True)


def _from_dict(raw: dict) -> AgentState:
    model = None
    raw_model = raw.get("model")
    if isinstance(raw_model, dict) and raw_model.get("name"):
        model = DeployedModel(
            name=raw_model["name"],
            version=raw_model.get("version", ""),
            sha256=raw_model.get("sha256", ""),
            path=raw_model.get("path", ""),
            entrypoint=raw_model.get("entrypoint"),
        )
    try:
        actual = ActualState(raw.get("actual_state", ActualState.UNKNOWN.value))
    except ValueError:
        # A state name this build does not know (downgrade, or a hand-edited
        # file). UNKNOWN is the safe reading: it forces a reconcile.
        actual = ActualState.UNKNOWN
    return AgentState(
        observed_generation=int(raw.get("observed_generation", 0)),
        actual_state=actual,
        model=model,
        inference_running=bool(raw.get("inference_running", False)),
        message=raw.get("message"),
        failed_generation=raw.get("failed_generation"),
        failure_count=int(raw.get("failure_count", 0)),
        # Monotonic clocks do not survive a restart, so a pending backoff is
        # deliberately dropped: after a reboot, retry immediately.
        next_retry_monotonic=0.0,
    )
