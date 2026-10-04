"""Agent configuration.

Env-var driven, because on the Jetson this runs as a systemd unit with an
`EnvironmentFile` and there is no interactive configuration step.

The device token is accepted from a *file* as well as an env var, and the file is
the recommended form: an env var is visible in `/proc/<pid>/environ` and in
`systemctl show`, while a file readable only by the service user is not.

`0640 root:keeper` is the mode `deploy/keeper.service` installs, and the reason it
is not `0600` is the unit's `User=keeper`: a 0600 root-owned file is unreadable by
the process that needs it. The property this code cares about is "not
world-readable", which that mode satisfies while staying readable by the group the
agent runs in. Getting it wrong is an ordinary mistake, so it is reported as a
`ConfigError` below rather than left to surface as a `PermissionError` traceback.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path


class ConfigError(RuntimeError):
    """Configuration is unusable. Raised at startup, never later."""


@dataclass(slots=True)
class AgentSettings:
    device_id: str
    control_plane_url: str
    token: str

    data_dir: Path = Path("/var/lib/keeper")
    poll_interval_seconds: int = 10

    # Agent-side ceiling on a single download. The control plane serves from a
    # local cache, so a stall here means the network or the ingress, not the
    # registry.
    download_timeout_seconds: int = 900
    request_timeout_seconds: int = 30

    runtime_impl: str = "mock"
    verify_tls: bool = True

    # Retry backoff after a failed reconciliation. A new generation resets it --
    # an operator pushing a fix should not wait out a backoff earned by the
    # previous instruction.
    retry_backoff_initial_seconds: int = 15
    retry_backoff_max_seconds: int = 600

    log_level: str = "INFO"

    @property
    def artifact_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def model_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def state_path(self) -> Path:
        return self.data_dir / "state.json"

    def url(self, path: str) -> str:
        """Join a control-plane-relative path onto the configured base URL.

        The server returns relative artifact URIs precisely so this join is the
        only place the base URL appears -- the same desired-state payload then
        works through the CAI domain, a tunnel, or localhost.
        """
        return f"{self.control_plane_url.rstrip('/')}/{path.lstrip('/')}"


def _runtime_is_installed(impl: str) -> bool:
    """Whether `impl`'s dependencies are importable -- without importing them.

    `find_spec` rather than a `try: import` for three reasons: importing
    onnxruntime costs hundreds of milliseconds and loads CUDA libraries on a
    device, doing it here would happen on every startup, and it leaves
    `sys.modules` clean, which `test_onnx_runtime.py`'s no-leak assertion depends
    on.

    This is a *necessary* condition, not a sufficient one. A wheel built for the
    wrong CUDA version installs fine, produces a spec, and then dies on
    `libcublas.so` at import -- which is the realistic Jetson failure. That one is
    caught in `main.py` around the runtime construction; both land on exit 2.

    Broad except: a half-installed distribution can make the finders themselves
    raise, and the right answer then is "no" rather than a traceback out of
    configuration loading.
    """
    if impl != "onnx":
        return True
    try:
        return importlib.util.find_spec("onnxruntime") is not None
    except Exception:  # pragma: no cover - requires a corrupt installation
        return False


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    """Read one integer setting.

    `env` is a parameter rather than a read of `os.environ` because it used to be
    the latter, and that silently defeated `load_settings(environ=...)` for all
    five integer settings: an injected environ configured the strings and the
    process environment configured the numbers. Nothing failed loudly -- a test
    passing `KEEPER_POLL_INTERVAL` got the default, and an ambient
    `KEEPER_POLL_INTERVAL` in a developer's shell changed the result of a test
    that had injected its own.
    """
    raw = env.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def load_settings(
    environ: Mapping[str, str] | None = None,
    *,
    runtime_is_installed: Callable[[str], bool] = _runtime_is_installed,
) -> AgentSettings:
    """Build settings from an environment, refusing anything unusable.

    Both parameters are injectable for the same reason: so the agent's startup
    decisions can be tested on a laptop that has neither the environment nor the
    wheel. `runtime_is_installed` takes the implementation name rather than being a
    bare boolean so the signature survives a third runtime (TensorRT, spec Phase 8)
    without changing.
    """
    env: Mapping[str, str] = os.environ if environ is None else environ

    device_id = (env.get("KEEPER_DEVICE_ID") or "").strip()
    if not device_id:
        raise ConfigError("KEEPER_DEVICE_ID is required")

    url = (env.get("KEEPER_CONTROL_PLANE_URL") or "").strip()
    if not url:
        raise ConfigError("KEEPER_CONTROL_PLANE_URL is required")
    if not url.startswith(("http://", "https://")):
        raise ConfigError(f"KEEPER_CONTROL_PLANE_URL must be an http(s) URL, got {url!r}")

    token = (env.get("KEEPER_TOKEN") or "").strip()
    token_file = (env.get("KEEPER_TOKEN_FILE") or "").strip()
    if not token and token_file:
        path = Path(token_file).expanduser()
        if not path.is_file():
            # `is_file()` swallows whatever OSError it hit and answers False, so
            # this branch is not only "absent": a directory, a dangling symlink and
            # a parent directory this user cannot traverse all land here too. The
            # message says so rather than claiming the file does not exist, because
            # `ls` as root would then contradict the journal.
            raise ConfigError(
                f"KEEPER_TOKEN_FILE is not a readable regular file: {path} -- it is "
                "missing, not a file, or inside a directory this user cannot enter"
            )
        try:
            token = path.read_text().strip()
        except OSError as exc:
            # The failure this exists for: a token file installed `0600 root:root`
            # while the unit runs as `User=keeper`. `is_file()` passes -- it only
            # stats -- and the read then raises `PermissionError`, which is *not* a
            # `ConfigError` and so escaped `main.py`'s exit-2 handler as a
            # traceback. An operator debugging a device over SSH got a stack trace
            # for a chmod, and systemd restarted the unit forever because the exit
            # code was 1 rather than the 2 `RestartPreventExitStatus` watches for.
            raise ConfigError(
                f"KEEPER_TOKEN_FILE cannot be read: {path} "
                f"({exc.strerror or exc.__class__.__name__}). The unit runs as "
                "User=keeper, so the file must be readable by that user -- install "
                "it 0640 root:keeper, not 0600 root:root."
            ) from exc
        except UnicodeDecodeError as exc:
            # A `ValueError`, not an `OSError`, so it needs its own branch to reach
            # exit 2. Reached by copying the wrong thing into place -- a keyfile, an
            # archive -- and the decode offset is the only safe detail to report:
            # the bytes themselves are a credential.
            raise ConfigError(
                f"KEEPER_TOKEN_FILE is not text: {path} is not valid {exc.encoding} "
                f"at byte {exc.start}"
            ) from exc
    if not token:
        raise ConfigError("one of KEEPER_TOKEN or KEEPER_TOKEN_FILE is required")

    settings = AgentSettings(device_id=device_id, control_plane_url=url, token=token)

    data_dir = env.get("KEEPER_DATA_DIR")
    if data_dir:
        settings.data_dir = Path(data_dir).expanduser()

    settings.poll_interval_seconds = _env_int(env, "KEEPER_POLL_INTERVAL", 10)
    settings.download_timeout_seconds = _env_int(env, "KEEPER_DOWNLOAD_TIMEOUT", 900)
    settings.request_timeout_seconds = _env_int(env, "KEEPER_REQUEST_TIMEOUT", 30)
    settings.retry_backoff_initial_seconds = _env_int(env, "KEEPER_RETRY_BACKOFF_INITIAL", 15)
    settings.retry_backoff_max_seconds = _env_int(env, "KEEPER_RETRY_BACKOFF_MAX", 600)

    settings.runtime_impl = (env.get("KEEPER_RUNTIME") or "mock").strip().lower()
    if settings.runtime_impl not in {"mock", "onnx"}:
        raise ConfigError(f"KEEPER_RUNTIME must be 'mock' or 'onnx', got {settings.runtime_impl!r}")
    # A missing wheel is a misconfiguration, not a runtime fault, so it is refused
    # here and becomes exit 2 -- which `RestartPreventExitStatus=2` in the systemd
    # unit turns into a stopped service with a readable journal line, rather than a
    # device that restarts every ten seconds forever. The alternative, starting up
    # and reporting FAILED, was rejected: at this point the agent has no identity to
    # report *with*, and `main.py:8-11`'s "a governance problem must not look like a
    # connectivity problem" argument is about reconcile failures, not about startup.
    if not runtime_is_installed(settings.runtime_impl):
        raise ConfigError(
            f"KEEPER_RUNTIME={settings.runtime_impl} but its runtime is not "
            "importable. Install it (on a Jetson, onnxruntime comes from NVIDIA's "
            "index rather than PyPI) or set KEEPER_RUNTIME=mock to run without "
            "inference."
        )

    # Opt-out only, and loudly: the device token travels in this request.
    verify = (env.get("KEEPER_VERIFY_TLS") or "true").strip().lower()
    settings.verify_tls = verify not in {"0", "false", "no", "off"}

    settings.log_level = (env.get("KEEPER_LOG_LEVEL") or "INFO").upper()
    return settings
