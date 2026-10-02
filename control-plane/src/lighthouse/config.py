"""Control-plane configuration.

Everything comes from the environment (spec SS17: "Do not hard-code
credentials"), because in CAI this process is an Application configured through
environment variables and has no config file to read.

The one rule worth stating out loud: `LIGHTHOUSE_ENV=cai` makes missing
credentials fatal at startup. A control plane that can revoke models, reachable
from the public internet with authentication silently disabled, is the worst
outcome available here -- considerably worse than failing to boot.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


class ConfigError(RuntimeError):
    """Configuration is unusable. Raised at startup, never later."""


@dataclass(slots=True)
class Settings:
    """Resolved runtime configuration."""

    env: str = "local"

    # Where SQLite and the artifact cache live. In CAI this must be on the
    # project filesystem (/home/cdsw/...) to survive an Application restart;
    # a container-local path would silently lose every enrollment.
    data_dir: Path = field(default_factory=lambda: Path("./.lighthouse"))

    registry_impl: str = "fake"

    # Connectivity thresholds, derived from heartbeat age at read time (spec SS14).
    heartbeat_interval_seconds: int = 10
    online_threshold_seconds: int = 30
    stale_threshold_seconds: int = 60

    # Artifact cache ceiling. Eviction is LRU and never touches a version that a
    # live desired deployment still references.
    artifact_cache_max_bytes: int = 8 * 1024**3
    artifact_chunk_size: int = 1024 * 1024

    # "proxy" streams bytes through this process (the path that must work);
    # "presigned" is a config-gated optimization that is allowed to fail.
    artifact_transport: str = "proxy"

    admin_token: str | None = None
    admin_token_ephemeral: bool = False

    # Dev-only: corrupt one byte when serving artifacts, to prove the device
    # refuses to activate on a checksum mismatch. Refused outside local env.
    dev_corrupt_artifacts: bool = False

    cors_allow_origins: list[str] = field(default_factory=list)
    log_level: str = "INFO"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "lighthouse.db"

    @property
    def artifact_cache_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def is_cai(self) -> bool:
        return self.env == "cai"


def load_settings(environ: dict[str, str] | None = None) -> Settings:
    """Build Settings from the environment, failing loudly on bad input."""
    env_map = dict(os.environ if environ is None else environ)
    prev = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(env_map)
        return _load()
    finally:
        os.environ.clear()
        os.environ.update(prev)


def _load() -> Settings:
    env = os.environ.get("LIGHTHOUSE_ENV", "local").strip().lower()
    if env not in {"local", "cai"}:
        raise ConfigError(f"LIGHTHOUSE_ENV must be 'local' or 'cai', got {env!r}")

    settings = Settings(env=env)

    data_dir = os.environ.get("LIGHTHOUSE_DATA_DIR")
    if data_dir:
        settings.data_dir = Path(data_dir).expanduser()
    elif env == "cai":
        # CAI projects are mounted at /home/cdsw and persist across restarts.
        settings.data_dir = Path("/home/cdsw/.lighthouse")

    settings.registry_impl = os.environ.get("LIGHTHOUSE_REGISTRY", "cai" if env == "cai" else "fake")
    if settings.registry_impl not in {"fake", "cai"}:
        raise ConfigError(f"LIGHTHOUSE_REGISTRY must be 'fake' or 'cai', got {settings.registry_impl!r}")

    settings.heartbeat_interval_seconds = _env_int("LIGHTHOUSE_HEARTBEAT_INTERVAL", 10)
    settings.online_threshold_seconds = _env_int("LIGHTHOUSE_ONLINE_THRESHOLD", 30)
    settings.stale_threshold_seconds = _env_int("LIGHTHOUSE_STALE_THRESHOLD", 60)
    if settings.stale_threshold_seconds <= settings.online_threshold_seconds:
        raise ConfigError(
            "LIGHTHOUSE_STALE_THRESHOLD must exceed LIGHTHOUSE_ONLINE_THRESHOLD "
            f"({settings.stale_threshold_seconds} <= {settings.online_threshold_seconds})"
        )

    settings.artifact_cache_max_bytes = _env_int(
        "LIGHTHOUSE_ARTIFACT_CACHE_MAX_BYTES", 8 * 1024**3
    )
    settings.artifact_chunk_size = _env_int("LIGHTHOUSE_ARTIFACT_CHUNK_SIZE", 1024 * 1024)
    settings.artifact_transport = os.environ.get("LIGHTHOUSE_ARTIFACT_TRANSPORT", "proxy")
    if settings.artifact_transport not in {"proxy", "presigned"}:
        raise ConfigError(
            "LIGHTHOUSE_ARTIFACT_TRANSPORT must be 'proxy' or 'presigned', "
            f"got {settings.artifact_transport!r}"
        )

    settings.log_level = os.environ.get("LIGHTHOUSE_LOG_LEVEL", "INFO").upper()

    origins = os.environ.get("LIGHTHOUSE_CORS_ORIGINS", "")
    settings.cors_allow_origins = [o.strip() for o in origins.split(",") if o.strip()]

    # -- the fail-closed rule --------------------------------------------
    admin_token = os.environ.get("LIGHTHOUSE_ADMIN_TOKEN", "").strip()
    if admin_token:
        settings.admin_token = admin_token
    elif env == "cai":
        raise ConfigError(
            "LIGHTHOUSE_ADMIN_TOKEN is required when LIGHTHOUSE_ENV=cai.\n"
            "A CAI Application with unauthenticated access enabled has no platform "
            "authentication in front of it, so this token is the only thing standing "
            "between the public internet and the deploy/stop/revoke surface. Refusing "
            "to start without it."
        )
    else:
        # Local dev: mint one and print it, so the harness is usable but still
        # never silently open.
        settings.admin_token = f"lha_{secrets.token_urlsafe(24)}"
        settings.admin_token_ephemeral = True

    settings.dev_corrupt_artifacts = _env_bool("LIGHTHOUSE_DEV_CORRUPT_ARTIFACTS", False)
    if settings.dev_corrupt_artifacts and env != "local":
        raise ConfigError(
            "LIGHTHOUSE_DEV_CORRUPT_ARTIFACTS is a local-only test hook and must not "
            "be set outside LIGHTHOUSE_ENV=local"
        )

    return settings
