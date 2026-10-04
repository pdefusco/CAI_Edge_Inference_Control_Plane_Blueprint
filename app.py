"""Entry script for the control plane running as a Cloudera AI Application.

Point the Application's script field at this file. It exists because a one-line
launcher in the project is easier to point an Application at than the
`lighthouse` console script (`control-plane/pyproject.toml:52`) on `PATH`.

Everything that decides whether the process is reachable -- the port, the bind
address, the exit-2-on-ConfigError path -- is already in `main.py:268-299`, and
this file must not duplicate or second-guess any of it.

What it does own is *reaching* `run()` at all, which took three measurements
against a real Application to get right. All three are here, each commented with
what was observed, and none changes what the server does once it is up: finding
the project root without `__file__`, starting the server when the caller already
has a running event loop, and picking the one CAI-provided port that is actually
bindable.
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys
import threading
from pathlib import Path

# The two packages whose `src` trees have to be importable, which is also what
# identifies the project root below.
_PACKAGES = ("contracts", "control-plane")


def _project_root() -> Path:
    """The project root, found WITHOUT depending on `__file__`.

    `python app.py` defines `__file__`. A CAI Application does not necessarily:
    with the Workbench editor the script is handed to an IPython-style kernel
    that executes it as numbered cells, echoes this docstring as output, and
    leaves `__file__` undefined. A bare `Path(__file__)` then dies on

        NameError: name '__file__' is not defined

    before `run()` is ever reached, and the Application exits 1 with no hint
    that the cause is the entry script rather than the config. Observed
    2026-10-04 in a deployed Application.

    Every candidate is *verified* to contain both `src` trees rather than
    trusted, so a wrong guess fails here with a readable message instead of
    resurfacing later as a `ModuleNotFoundError` for `lighthouse`.
    """
    starts: list[Path] = []
    try:
        starts.append(Path(__file__).resolve().parent)
    except NameError:
        pass  # Expected under the kernel; the candidates below cover it.
    starts.append(Path.cwd().resolve())
    # The project filesystem's conventional mount, and the same path the `cai`
    # default for `data_dir` is built on (`docs/cai-deployment.md` §4).
    starts.append(Path("/home/cdsw"))

    for start in starts:
        for directory in (start, *start.parents):
            if all((directory / pkg / "src").is_dir() for pkg in _PACKAGES):
                return directory

    print(
        "could not locate the project root. Looked at these and every parent:\n"
        + "".join(f"  {s}\n" for s in starts)
        + "None contains both "
        + " and ".join(f"{pkg}/src" for pkg in _PACKAGES)
        + ".\nPoint the Application's script field at app.py in the project"
        " root, beside those two directories.",
        file=sys.stderr,
    )
    raise SystemExit(2)


# A deliberate departure from the repo's install precedent. Everywhere else --
# `Makefile:49-53`, `scripts/_common.sh:32-43` -- the packages are pip-installed
# editable, and `scripts/build_model.py:737` assumes that outright. Here the
# repo's own `src` trees go on `sys.path` ahead of anything installed, for two
# reasons: an Application container is not the container a Session's install ran
# in (see `docs/cai-deployment.md` §3, which marks that an open question), and an
# Application should serve the project's current code rather than a copy
# installed weeks ago.
#
# This makes the *first-party* half of the import work with no install at all.
# The third-party half -- fastapi, uvicorn, jinja2, pydantic, pyyaml,
# python-multipart, and httpx for the `cai` extra -- it cannot help with.
_REPO = _project_root()
for _pkg in _PACKAGES:
    _src = _REPO / _pkg / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from lighthouse.main import run


def _pick_port() -> None:
    """Choose between the two ports CAI offers, by trying to bind them.

    `main.py:297` prefers `CDSW_APP_PORT`, which is right for an Application
    that runs as a plain process. Under the kernel it is wrong: the engine
    already holds it, and uvicorn dies with

        [Errno 98] error while attempting to bind on address
        ('0.0.0.0', 8100): address already in use

    *after* logging `Application startup complete`, so the log reads as a
    healthy boot. Observed 2026-10-04 in a deployed Application. The public URL
    for that kind of Application is proxied off `CDSW_READONLY_PORT` instead --
    the pattern a sibling blueprint (`CAI_Agentic_NBA_Observability_Blueprint`,
    `launch_app.py`) has deployed successfully.

    Hardcoding the read-only port would just move the breakage: both variables
    are set either way, so a plain-process Application would then bind a port
    nothing routes to and look healthy while being unreachable. Binding is the
    only test that distinguishes them, so that is the test -- preferring
    `CDSW_APP_PORT` and falling back only when it is genuinely taken, which
    needs no knowledge of which runtime kind this is.

    An explicit `PORT` disables the probe entirely, because silently overriding
    one would be a bug: `CDSW_READONLY_PORT` is set in a *Session* too, so the
    by-hand command in `docs/cai-deployment.md` §3 (`env -u CDSW_APP_PORT
    PORT=8900 python app.py`) would otherwise bind something other than 8900 and
    report success. Disabling the probe leaves `main.py:297`'s precedence
    untouched rather than inverting it -- so `PORT` decides only if
    `CDSW_APP_PORT` is unset, which is exactly what that `env -u` is for.

    Nothing happens on a laptop either, where no CAI variable is set.
    """
    if os.environ.get("PORT"):
        return  # Explicitly asked for; not ours to second-guess.

    tried: list[str] = []
    for name in ("CDSW_APP_PORT", "CDSW_READONLY_PORT"):
        raw = os.environ.get(name)
        if not raw:
            continue
        try:
            port = int(raw)
        except ValueError:
            tried.append(f"{name}={raw!r} (not a number)")
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            # The same option uvicorn sets, so this probe succeeds exactly when
            # uvicorn's own bind would.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("0.0.0.0", port))
            except OSError as exc:
                tried.append(f"{name}={port} ({exc.strerror})")
                continue
        if tried:
            print(
                f"{name}={port} is free; not using " + ", ".join(tried),
                file=sys.stderr,
            )
        # Handed over as `PORT` with `CDSW_APP_PORT` cleared, which is how
        # `docs/cai-deployment.md` §3 already tells you to override the port by
        # hand (`env -u CDSW_APP_PORT PORT=8900`). `main.py` stays generic and
        # keeps deciding the port in one place.
        os.environ["PORT"] = str(port)
        os.environ.pop("CDSW_APP_PORT", None)
        return

    if tried:
        print(
            "no CAI-provided port could be bound: " + ", ".join(tried)
            + ". Letting main.py choose, which will almost certainly fail too.",
            file=sys.stderr,
        )


def serve() -> None:
    """Call `run()`, tolerating a caller that already has an event loop.

    `run()` ends in `uvicorn.run()`, which ends in `asyncio.run()` and so
    requires that *no* loop is running in the calling thread. A plain
    interpreter satisfies that. The Workbench kernel does not -- it runs its
    cells on a live uvloop, and the stack ends:

        RuntimeError: Runner.run() cannot be called from a running event loop
        RuntimeError: Cannot run the event loop while another loop is running

    and the Application exits 1 *after* the app was successfully built, so the
    log shows a healthy startup followed by an asyncio traceback with no
    mention of the real cause. Observed 2026-10-04 in a deployed Application.

    The fix is a thread of uvicorn's own, where `asyncio.run()` is legal again.
    It is deliberately non-daemon and joined: the process must outlive this
    call or CAI sees the script finish and takes the Application down with it.

    `run()` is reused verbatim rather than reimplemented, so port, bind address
    and the exit-2-on-ConfigError path stay defined in exactly one place. The
    one thing a thread would otherwise swallow is `SystemExit` -- a config error
    would kill the thread quietly and leave this returning 0 -- so the
    exception is carried back out and re-raised here.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        run()  # No loop in this thread: the normal path, unchanged.
        return

    failure: list[BaseException] = []

    def _target() -> None:
        try:
            run()
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the caller
            failure.append(exc)

    thread = threading.Thread(target=_target, name="uvicorn", daemon=False)
    thread.start()
    thread.join()
    if failure:
        raise failure[0]


# Called at module scope, not under `if __name__ == "__main__"`: CAI executes
# this script rather than importing it, and a guard here would make an
# Application that starts cleanly and serves nothing.
_pick_port()
serve()
