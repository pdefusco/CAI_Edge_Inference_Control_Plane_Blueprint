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


def _listening(proc: Path) -> dict[int, set[str]]:
    """Every port in TCP_LISTEN in this container, mapped to its socket inodes.

    Reads `/proc/net/tcp{,6}` directly because `ss` and `netstat` have both
    shown nothing for this class of conflict inside a CAI container
    (`main.py:286-295`).
    """
    ports: dict[int, set[str]] = {}
    for name in ("net/tcp", "net/tcp6"):
        try:
            lines = (proc / name).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            # Field 3 is the state; 0x0A is TCP_LISTEN. A *connected* socket on
            # a port is a client of someone else's and not what blocks a bind.
            if len(fields) <= 9 or fields[3] != "0A":
                continue
            try:
                port = int(fields[1].rsplit(":", 1)[-1], 16)
            except ValueError:
                continue
            ports.setdefault(port, set()).add(fields[9])
    return ports


def _mask(hex_addr: str) -> str:
    """Decode a `/proc/net/tcp` local address, masking anything routable.

    The three answers that matter are `0.0.0.0` (nobody else can have this
    port), `127.0.0.1` (the port is only taken on loopback, so the pod's own
    interface may still be bindable) and everything else. Only the first two are
    printed literally. A pod IP is private tenant data and must not reach a log
    someone pastes into an issue, so it comes back as `<pod-ip>` --
    `_port_report` already learned that lesson the expensive way.
    """
    try:
        if len(hex_addr) == 8:
            ip = ".".join(str(b) for b in reversed(bytes.fromhex(hex_addr)))
        elif len(hex_addr) == 32:
            # Four 32-bit words, each little-endian within the word.
            raw = b"".join(
                bytes.fromhex(hex_addr[i : i + 8])[::-1] for i in range(0, 32, 8)
            )
            ip = socket.inet_ntop(socket.AF_INET6, raw)
        else:
            return "<unparsed>"
    except (ValueError, OSError):
        return "<unparsed>"
    if ip in ("0.0.0.0", "::"):
        return ip
    if ip.startswith("127.") or ip in ("::1", "::ffff:127.0.0.1"):
        return ip
    return "<pod-ip>"


def _listen_addrs(proc: Path) -> dict[int, set[str]]:
    """Every listening port mapped to the masked addresses it is bound to.

    Separate from `_listening` because the two answer different questions: that
    one finds who owns a port, this one finds *where* they bound it, and the
    second is what decides whether the port is winnable at all. Observed
    2026-10-04: a wildcard holder cannot be worked around, a loopback-only
    holder can -- on Linux, binding a specific non-loopback address succeeds
    while another socket holds a different specific address on the same port.
    """
    addrs: dict[int, set[str]] = {}
    for name in ("net/tcp", "net/tcp6"):
        try:
            lines = (proc / name).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) <= 9 or fields[3] != "0A":
                continue
            local = fields[1].rsplit(":", 1)
            if len(local) != 2:
                continue
            try:
                port = int(local[1], 16)
            except ValueError:
                continue
            addrs.setdefault(port, set()).add(_mask(local[0]))
    return addrs


def _owner(inodes: set[str], proc: Path) -> str:
    """`pid N (cmdline)` for whichever visible process holds one of `inodes`."""
    if not inodes:
        return ""
    for entry in sorted(proc.iterdir()):
        if not entry.name.isdigit():
            continue
        try:
            fds = list((entry / "fd").iterdir())
        except OSError:
            continue  # Someone else's process, or it just exited.
        for fd in fds:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if not (target.startswith("socket:[") and target[8:-1] in inodes):
                continue
            try:
                cmd = (entry / "cmdline").read_text().replace("\0", " ").strip()
            except OSError:
                cmd = ""
            return f"pid {entry.name}" + (f" ({cmd[:160]})" if cmd else "")
    return "a process outside this container's view"


def _port_holder(port: int, *, proc: Path = Path("/proc")) -> str:
    """Best-effort description of the process already listening on `port`.

    Only ever called on the failure path, where the errno alone is not
    actionable: "address already in use" on the one port CAI routes is either a
    leftover of ours (restart it) or part of the engine (change how the
    Application runs), and those have completely different fixes. Observed
    2026-10-04 in a deployed Application, where `CDSW_APP_PORT` and
    `CDSW_READONLY_PORT` were *the same port* and both were taken, so there was
    no fallback left and no way to tell the two causes apart from the log. The
    answer that line gave -- `pid 1 (... engine-init ... jupyter-wsg-launcher)`
    -- is what `docs/cai-deployment.md` §3 now records.

    Returns "" when it cannot tell, and must never raise: a diagnostic that
    replaces the real error with its own traceback is worse than no diagnostic.
    """
    try:
        holder = _owner(_listening(proc).get(port, set()), proc)
        where = ", ".join(sorted(_listen_addrs(proc).get(port, set())))
        if not holder:
            return f"bound on {where}" if where else ""
        return f"held by {holder}" + (f", bound on {where}" if where else "")
    except Exception:  # noqa: BLE001 -- never let the diagnostic mask the error
        return ""


def _port_report(*, proc: Path = Path("/proc")) -> list[str]:
    """What CAI offered and what is already listening, as indented lines.

    Printed alongside the bind failure because the two causes §3 distinguishes
    are told apart by exactly this: which ports CAI *named*, and whether any of
    them is free.

    Shown only if the name contains `PORT` *and* the value is a bare number.
    Both halves are load-bearing. The name filter keeps out the workbench domain
    and CRNs, which identify a tenant. The numeric filter was added after the
    first real dump leaked private IPs: Kubernetes service discovery sets
    `<SERVICE>_PORT=tcp://172.x.y.z:8100` and `..._PORT_8100_TCP_ADDR=172.x.y.z`
    for every service in the namespace, so a name filter alone prints the
    cluster's internal addressing into a log someone pastes into an issue. A
    port number is the whole point here and carries no tenant identity.
    Observed 2026-10-04 (`[[lighthouse-repo-is-public]]`).
    """
    lines: list[str] = []
    try:
        offered = sorted(
            (name, value)
            for name, value in os.environ.items()
            if "PORT" in name and value.isdigit()
        )
        if offered:
            lines.append(
                "  ports in the environment: "
                + " ".join(f"{name}={value}" for name, value in offered)
            )
        table = _listening(proc)
        if table:
            where = _listen_addrs(proc)
            lines.append("  listening in this container:")
            for port in sorted(table):
                owner = _owner(table[port], proc) or "owner unknown"
                on = ", ".join(sorted(where.get(port, set()))) or "?"
                lines.append(f"    {port} on {on} -- {owner}")
    except Exception:  # noqa: BLE001 -- a diagnostic must not mask the error
        return lines
    return lines


def _bind_error(port: int, host: str) -> str | None:
    """`None` if uvicorn could bind `host:port`, else the errno's text.

    `SO_REUSEADDR` matches what uvicorn sets, so this probe succeeds exactly
    when uvicorn's own bind would.

    The caller's wildcard-then-loopback fallback is only meaningful under Linux
    semantics, where a wildcard bind collides with a holder of any specific
    address on that port and vice versa. **BSD differs**: on macOS, with
    `SO_REUSEADDR`, a wildcard bind coexists with a specific-address holder, so
    this probe answers a materially different question there. That is harmless
    -- the Application is Linux and a laptop sets none of these variables -- but
    it does mean the fallback cannot be verified against real sockets on a Mac.
    Observed 2026-10-04 while testing exactly that.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError as exc:
            return exc.strerror or str(exc)
    return None


def _app_service_ports() -> list[tuple[str, str]]:
    """Ports that Kubernetes service discovery calls an **app** service port.

    A last resort that buys a *live process*, not a reachable one. Do not read
    the name as authority. Observed 2026-10-04 in a deployed Application: all
    three of `CDSW_APP_PORT`, `CDSW_PUBLIC_PORT` and `CDSW_READONLY_PORT` read
    **8100**, while the same environment says

        DS_RUNTIME_<id>_SERVICE_PORT_APP=8090
        DS_RUNTIME_<id>_SERVICE_PORT_READ_ONLY=8100
        DS_RUNTIME_<id>_SERVICE_PORT_PUBLIC=8080

    which reads like a correction and is not one. `CDSW_PUBLIC_PORT` is 8100
    against a service calling `public` 8080, and *nobody set it* -- not the
    Application's environment variables, not the Project's. **The two
    vocabularies are unrelated**: `SERVICE_PORT_*` describes the Kubernetes
    service, `CDSW_*` the single port the engine proxies a workload through.
    They share the word "app" by coincidence, and reading the mismatch as an
    override cost a wasted round trip (`docs/cai-deployment.md` §3).

    So 8100 is where CAI routes, and binding 8090 instead moves the process
    *away* from the route: measured, that yields an `istio-envoy` 502 with
    `content-length: 0` while uvicorn logs a healthy start. Kubernetes injects
    these variables for every service in the namespace, so the value found may
    belong to a sibling workload -- here 8090 did, a Session's. The bind still
    decides, and the caller says loudly when the winner came from here, because
    this is the candidate that produces an app which looks healthy and answers
    nobody.
    """
    found: dict[int, str] = {}
    for name, value in sorted(os.environ.items()):
        if name.endswith("_SERVICE_PORT_APP") and value.isdigit():
            found.setdefault(int(value), name)
    return [(name, str(port)) for port, name in sorted(found.items())]


def _pick_port() -> None:
    """Choose between the two ports CAI offers, by trying to bind them.

    `main.py:297` prefers `CDSW_APP_PORT`, which is right for an Application
    that runs as a plain process. Under the kernel it is wrong: the engine
    already holds it, and uvicorn dies with

        [Errno 98] error while attempting to bind on address
        ('0.0.0.0', 8100): address already in use

    *after* logging `Application startup complete`, so the log reads as a
    healthy boot. Observed 2026-10-04 in a deployed Application, where
    `_port_holder` named the holder as **pid 1**, the engine's own
    `jupyter-wsg-launcher`.

    **Each port is tried on `0.0.0.0` and then on `127.0.0.1`, and the port
    matters more than the address.** This is the correction to an earlier
    version of this file, which probed only the wildcard and so reported
    `CDSW_APP_PORT` "taken" without testing the address that was free. On Linux
    a wildcard bind collides with a holder of *any* specific address on that
    port, so `0.0.0.0:8100` failing is consistent with the holder sitting on one
    specific address and `127.0.0.1:8100` being bindable. The earlier version
    also claimed only `0.0.0.0` is reachable "from a proxy in another
    container", which is wrong: a sibling blueprint
    (`CAI_Agentic_NBA_Observability_Blueprint`, `deploy/cai/start.py`) sets
    `BIND_HOST=127.0.0.1` on `CDSW_APP_PORT` and is reachable in production.
    CAI proxies an Application from *inside its own pod*, which is exactly what
    `CDSW_APP_PORT` meaning "the port the engine proxies a workload through"
    implies. So winning the routed port on loopback beats winning an unrouted
    port on the wildcard, and that is the order here.

    The fallback to `CDSW_READONLY_PORT` comes from that same blueprint, which
    offers it as a selectable alternative (`CAI_SELLER_PORT_VARIABLE`) for when
    `CDSW_APP_PORT` is occupied. Observed 2026-10-04: here both variables read
    8100, so the fallback had nowhere to go until `CDSW_READONLY_PORT` was set
    by hand -- and the 8090 it then bound is still not the routed port.

    So there is a third candidate, `_app_service_ports()`, which reads the port
    Kubernetes service discovery *calls* the app port. It is last and it is not
    a repair -- measured, its 8090 is bindable and routes to nobody. It exists
    to turn a crash into a live process that can be probed. See
    `docs/cai-deployment.md` §3.

    The winner is exported as `LIGHTHOUSE_HOST` as well as `PORT`, because a
    loopback win is only correct if `main.py:303` actually binds there; its
    default for a non-local env is `0.0.0.0`, which would re-raise the very
    `EADDRINUSE` this probe just avoided.

    Hardcoding any one of these just moves the breakage, because "it bound" and
    "it is reachable" are different facts and only the first is testable from
    inside the container. Binding is still the only test available, so that is
    the test -- preferring `CDSW_APP_PORT`, which is the port CAI routes to, and
    falling back only when it is genuinely taken. That needs no knowledge of
    which runtime kind this is, and when the fallback wins it says so loudly,
    because a win there means the app is up and unreachable.

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

    candidates: list[tuple[str, str]] = [
        (name, os.environ[name])
        for name in ("CDSW_APP_PORT", "CDSW_READONLY_PORT")
        if os.environ.get(name)
    ]
    # Last, and only as a repair: see `_app_service_ports`.
    discovered = {name for name, _ in _app_service_ports()}
    candidates += _app_service_ports()

    # An explicit `LIGHTHOUSE_HOST` is honoured the way `PORT` is: probe only
    # what the caller asked for rather than silently binding somewhere else.
    chosen_host = os.environ.get("LIGHTHOUSE_HOST")
    hosts = (chosen_host,) if chosen_host else ("0.0.0.0", "127.0.0.1")

    tried: list[str] = []
    failed_ports: list[int] = []
    for name, raw in candidates:
        try:
            port = int(raw)
        except ValueError:
            tried.append(f"{name}={raw!r} (not a number)")
            continue
        for host in hosts:
            error = _bind_error(port, host)
            if error is None:
                break
            tried.append(f"{name}={port} on {host} ({error})")
            failed_ports.append(port)
        else:
            continue  # Every address refused this port; try the next candidate.

        if tried:
            print(
                f"{name}={port} is free on {host}; not using "
                + ", ".join(tried),
                file=sys.stderr,
            )
        # Anything but the routed port on the wildcard address is a compromise,
        # and the two compromises fail in opposite directions, so each says which
        # one it made and both get the full table. Observed 2026-10-04: a
        # fallback served happily on 8090 while the public URL returned a 502
        # from the ingress, and the table was unavailable precisely because
        # nothing had *failed*.
        if name in discovered or name != "CDSW_APP_PORT":
            print(
                "\n".join(
                    [
                        f"serving on {port}, taken from {name} because"
                        " CDSW_APP_PORT could not be bound on any address. CAI"
                        " routes the Application's URL to CDSW_APP_PORT, not"
                        " here, so expect a 502 from the ingress -- see"
                        " `docs/cai-deployment.md` §3.",
                        *_port_report(),
                    ]
                ),
                file=sys.stderr,
            )
        elif host != "0.0.0.0":
            # The good outcome, and still worth a line: it means the holder of
            # this port bound a *specific* address, so we took the routed port
            # on a different one. Reachable only because CAI proxies an
            # Application from inside its own pod -- which is how the sibling
            # blueprint has always worked.
            print(
                "\n".join(
                    [
                        f"serving on {port} ({name}) bound to {host} rather than"
                        " 0.0.0.0, which is already taken. This is the routed"
                        " port, so the URL should work; the table below says who"
                        " holds the wildcard.",
                        *_port_report(),
                    ]
                ),
                file=sys.stderr,
            )
        # Handed over as `PORT` with `CDSW_APP_PORT` cleared, which is how
        # `docs/cai-deployment.md` §3 already tells you to override the port by
        # hand (`env -u CDSW_APP_PORT PORT=8900`). `main.py` stays generic and
        # keeps deciding the port in one place.
        os.environ["PORT"] = str(port)
        os.environ["LIGHTHOUSE_HOST"] = host
        os.environ.pop("CDSW_APP_PORT", None)
        return

    if tried:
        lines = [
            "no CAI-provided port could be bound: " + ", ".join(tried) + ".",
        ]
        for failed in sorted(set(failed_ports)):
            holder = _port_holder(failed)
            if holder:
                lines.append(f"  port {failed} is {holder}")
        lines.extend(_port_report())
        lines.append(
            "  If the holder is an earlier instance of this app, stop the"
            " Application fully and start it again rather than restarting it."
            " If it is the engine -- pid 1, or a command line naming"
            " `engine-init`, `jupyter` or `workbench` -- then read the address"
            " column above: every port was refused on both 0.0.0.0 and"
            " 127.0.0.1, so a holder shown on 0.0.0.0 cannot be worked around"
            " from here and the fix is at creation time (the runtime's editor)."
            " See `docs/cai-deployment.md` §3."
        )
        print("\n".join(lines), file=sys.stderr)


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
