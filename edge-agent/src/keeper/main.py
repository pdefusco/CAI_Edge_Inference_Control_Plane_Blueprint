"""`keeper` entry point: the poll loop that runs as a systemd unit on the Jetson.

Deliberately thin. Everything interesting is in `Reconciler`, which is driven here
by a loop that does three things the reconciler should not have to care about:
sleeping, catching SIGTERM, and deciding to poll early when the control plane says
it has a newer generation.

The loop never exits on a reconcile error. An agent that dies on a bad artifact is
an agent that stops heartbeating, which makes a *governance* problem look like a
*connectivity* problem on the dashboard -- the single most misleading failure this
system could have.
"""

from __future__ import annotations

import argparse
import logging
import re
import signal
import sys
import threading
import time
from types import FrameType

from .artifact_manager import ArtifactManager
from .client import ControlPlaneClient
from .config import AgentSettings, ConfigError, load_settings
from .reconciler import Reconciler
from .runtime import build_runtime
from .state import StateStore

log = logging.getLogger("keeper")


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    handler.addFilter(_RedactingFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level, logging.INFO))


class _RedactingFilter(logging.Filter):
    """Keep the device token out of the journal.

    `journalctl` on a device is readable by anyone with shell access and is often
    the first thing pasted into a bug report. The token is a permanent deployment
    credential, so it must never appear there.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover
            return True
        redacted = re.sub(r"(lhd_[A-Za-z0-9]+)\.[A-Za-z0-9_\-]+", r"\1.***", message)
        redacted = re.sub(r"(?i)(authorization:\s*bearer\s+)\S+", r"\1***", redacted)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


class Agent:
    """Owns the poll loop and the objects it needs."""

    def __init__(self, settings: AgentSettings) -> None:
        self._settings = settings
        self._stop = threading.Event()
        self._client = ControlPlaneClient(settings)
        self._artifacts = ArtifactManager(settings, self._client)
        self._runtime = build_runtime(settings.runtime_impl)
        self._reconciler = Reconciler(
            settings,
            self._client,
            self._artifacts,
            self._runtime,
            StateStore(settings.state_path),
        )
        self._reconciler.set_abort_probe(self._newer_generation_exists)
        self._last_probe = 0.0

    # -- loop --------------------------------------------------------------

    def run(self) -> int:
        settings = self._settings
        log.info(
            "keeper starting: device=%s control_plane=%s runtime=%s data_dir=%s",
            settings.device_id,
            settings.control_plane_url,
            self._runtime.name,
            settings.data_dir,
        )
        if not settings.verify_tls:
            log.warning(
                "TLS verification is DISABLED -- the device token is exposed to "
                "anyone who can intercept this connection"
            )

        interval = settings.poll_interval_seconds
        while not self._stop.is_set():
            try:
                self._reconciler.tick()
            except Exception:
                # Last-resort guard. The reconciler maps every error it expects;
                # anything reaching here is a bug, and crashing would be worse than
                # logging it and heartbeating again next tick.
                log.exception("unhandled error in reconcile tick")
            self._stop.wait(interval)

        self._shutdown()
        return 0

    def run_once(self) -> int:
        """One pass, then exit. Used by `--once` and by the dev scripts."""
        outcome = self._reconciler.tick()
        if outcome is not None:
            log.info("single pass: %s", outcome.note)
        self._shutdown()
        return 0

    def request_stop(self, signum: int, _frame: FrameType | None) -> None:
        log.info("received signal %s; finishing current pass and exiting", signum)
        self._stop.set()

    def _shutdown(self) -> None:
        """Exit without changing the model's deployment state.

        Stopping the runtime here would be wrong: a `systemctl restart` is not an
        operator asking for STOPPED, and reporting STOPPED because the agent
        bounced would make the dashboard lie. The agent leaves state alone and
        re-converges on its next start.
        """
        log.info("keeper stopping (deployment state left unchanged)")
        try:
            self._client.close()
        except Exception:  # pragma: no cover
            pass

    # -- mid-download abort probe ------------------------------------------

    def _newer_generation_exists(self, generation: int) -> bool:
        """Has the operator superseded the generation we are downloading?

        Called between chunks, so it is rate-limited to the poll interval -- a
        probe per megabyte would hammer the control plane. A failed probe returns
        False: a flaky network must never abort a download that is otherwise fine.
        """
        if self._stop.is_set():
            return True
        now = time.monotonic()
        if now - self._last_probe < self._settings.poll_interval_seconds:
            return False
        self._last_probe = now
        try:
            desired = self._client.fetch_desired_state()
        except Exception as exc:
            log.debug("abort probe failed (continuing download): %s", exc)
            return False
        return desired.generation > generation


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="keeper",
        description="Lighthouse edge agent: reconciles local model deployment "
        "against CAI desired state.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="run a single reconcile pass and exit (useful in scripts and CI)",
    )
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    configure_logging(settings.log_level)
    agent = Agent(settings)

    if args.once:
        return agent.run_once()

    signal.signal(signal.SIGTERM, agent.request_stop)
    signal.signal(signal.SIGINT, agent.request_stop)
    return agent.run()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
