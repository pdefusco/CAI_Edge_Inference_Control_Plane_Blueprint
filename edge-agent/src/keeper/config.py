"""Agent configuration.

Env-var driven, because on the Jetson this runs as a systemd unit with an
`EnvironmentFile` and there is no interactive configuration step.

The device token is accepted from a *file* as well as an env var, and the file is
the recommended form: an env var is visible in `/proc/<pid>/environ` and in
`systemctl show`, while a `0600` file is not.
"""

from __future__ import annotations

import os
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


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def load_settings(environ: dict[str, str] | None = None) -> AgentSettings:
    env = os.environ if environ is None else environ

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
            raise ConfigError(f"KEEPER_TOKEN_FILE does not exist: {path}")
        token = path.read_text().strip()
    if not token:
        raise ConfigError("one of KEEPER_TOKEN or KEEPER_TOKEN_FILE is required")

    settings = AgentSettings(device_id=device_id, control_plane_url=url, token=token)

    data_dir = env.get("KEEPER_DATA_DIR")
    if data_dir:
        settings.data_dir = Path(data_dir).expanduser()

    settings.poll_interval_seconds = _env_int("KEEPER_POLL_INTERVAL", 10)
    settings.download_timeout_seconds = _env_int("KEEPER_DOWNLOAD_TIMEOUT", 900)
    settings.request_timeout_seconds = _env_int("KEEPER_REQUEST_TIMEOUT", 30)
    settings.retry_backoff_initial_seconds = _env_int("KEEPER_RETRY_BACKOFF_INITIAL", 15)
    settings.retry_backoff_max_seconds = _env_int("KEEPER_RETRY_BACKOFF_MAX", 600)

    settings.runtime_impl = (env.get("KEEPER_RUNTIME") or "mock").strip().lower()
    if settings.runtime_impl not in {"mock", "onnx"}:
        raise ConfigError(f"KEEPER_RUNTIME must be 'mock' or 'onnx', got {settings.runtime_impl!r}")

    # Opt-out only, and loudly: the device token travels in this request.
    verify = (env.get("KEEPER_VERIFY_TLS") or "true").strip().lower()
    settings.verify_tls = verify not in {"0", "false", "no", "off"}

    settings.log_level = (env.get("KEEPER_LOG_LEVEL") or "INFO").upper()
    return settings
