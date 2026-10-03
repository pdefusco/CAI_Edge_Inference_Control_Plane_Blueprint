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

import logging
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


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

    # -- the CAI registry adapter, read always but only validated when it is the
    # selected implementation ---------------------------------------------
    #
    # `registry_domain` has no default and never will. This repo is public and a
    # registry hostname is a tenant identifier, so the operator either names the
    # host or names a CDP environment and lets the cdp CLI discover it.
    registry_domain: str | None = None
    registry_environment: str | None = None
    registry_api_prefix: str = "/api/v2"

    # Where the UMS workload JWT comes from. "cli" shells out to `cdp`, which is
    # what a CAI Session has; "env" and "file" exist for anywhere that does not,
    # and hand the refresh problem to whatever supplies the value.
    registry_token_source: str = "cli"
    registry_token_env: str | None = None
    registry_token_file: Path | None = None
    registry_workload_name: str = "DE"

    # Metadata calls are quick; an artifact download is not, and shares the
    # agent's reasoning at keeper/config.py for the same split.
    registry_request_timeout: int = 30
    registry_stream_timeout: int = 900

    registry_verify_tls: bool = True
    registry_ca_bundle: Path | None = None

    # Model metadata is cached this long inside the adapter. Deployment-time
    # resolution deliberately ignores it -- see ModelRegistry.get_version.
    registry_cache_ttl_seconds: int = 30

    # Connectivity thresholds, derived from heartbeat age at read time (spec SS14).
    heartbeat_interval_seconds: int = 10
    online_threshold_seconds: int = 30
    stale_threshold_seconds: int = 60

    # Artifact cache ceiling. Eviction is LRU and never touches a version that a
    # live desired deployment still references.
    artifact_cache_max_bytes: int = 8 * 1024**3
    artifact_chunk_size: int = 1024 * 1024

    # "proxy" streams bytes through this process. It is the only implemented
    # transport and the only one that can work for a device with no object-store
    # identity, which is the whole point of the byte proxy. "presigned" is
    # reserved: the CAI registry API offers no signed URL to forward, so nothing
    # reads this field yet. Accepted so a deployment that sets it does not fail
    # to boot, but it changes no behavior.
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


def _load_registry(settings: Settings) -> None:
    """Read the CAI adapter's settings, and validate them if it is the one wired in.

    Validation is gated on `registry_impl` rather than on `env` so that a local
    run against the real registry (the normal way to debug it from a Session) gets
    the same checks as production, and so that `LIGHTHOUSE_REGISTRY=fake` inside
    CAI stays usable without inventing a domain.
    """
    settings.registry_domain = (os.environ.get("LIGHTHOUSE_REGISTRY_DOMAIN") or "").strip() or None
    settings.registry_environment = (
        os.environ.get("LIGHTHOUSE_REGISTRY_ENVIRONMENT") or ""
    ).strip() or None
    settings.registry_api_prefix = "/" + os.environ.get(
        "LIGHTHOUSE_REGISTRY_API_PREFIX", "/api/v2"
    ).strip().strip("/")

    settings.registry_token_source = (
        os.environ.get("LIGHTHOUSE_REGISTRY_TOKEN_SOURCE", "cli").strip().lower()
    )
    settings.registry_token_env = (
        os.environ.get("LIGHTHOUSE_REGISTRY_TOKEN_ENV") or ""
    ).strip() or None
    token_file = (os.environ.get("LIGHTHOUSE_REGISTRY_TOKEN_FILE") or "").strip()
    settings.registry_token_file = Path(token_file).expanduser() if token_file else None
    settings.registry_workload_name = (
        os.environ.get("LIGHTHOUSE_REGISTRY_WORKLOAD_NAME", "DE").strip().upper()
    )

    settings.registry_request_timeout = _env_int("LIGHTHOUSE_REGISTRY_REQUEST_TIMEOUT", 30)
    settings.registry_stream_timeout = _env_int("LIGHTHOUSE_REGISTRY_STREAM_TIMEOUT", 900)
    settings.registry_verify_tls = _env_bool("LIGHTHOUSE_REGISTRY_VERIFY_TLS", True)
    ca_bundle = (os.environ.get("LIGHTHOUSE_REGISTRY_CA_BUNDLE") or "").strip()
    settings.registry_ca_bundle = Path(ca_bundle).expanduser() if ca_bundle else None
    settings.registry_cache_ttl_seconds = _env_int("LIGHTHOUSE_REGISTRY_CACHE_TTL", 30)

    if settings.registry_impl != "cai":
        return

    if settings.registry_domain and settings.registry_environment:
        raise ConfigError(
            "Set LIGHTHOUSE_REGISTRY_DOMAIN or LIGHTHOUSE_REGISTRY_ENVIRONMENT, not both.\n"
            "Naming the host skips discovery; naming the environment discovers the host "
            "via `cdp ml list-model-registries`. Setting both leaves it ambiguous which "
            "registry was intended."
        )
    if not settings.registry_domain and not settings.registry_environment:
        raise ConfigError(
            "The CAI registry adapter needs to know which registry to talk to.\n"
            "Set LIGHTHOUSE_REGISTRY_DOMAIN to its hostname, or set "
            "LIGHTHOUSE_REGISTRY_ENVIRONMENT to a CDP environment name and let "
            "`cdp ml list-model-registries` discover the host. There is deliberately "
            "no default: a registry hostname identifies a tenant."
        )

    if settings.registry_token_source not in {"cli", "env", "file"}:
        raise ConfigError(
            "LIGHTHOUSE_REGISTRY_TOKEN_SOURCE must be 'cli', 'env' or 'file', got "
            f"{settings.registry_token_source!r}"
        )
    if settings.registry_token_source == "env" and not settings.registry_token_env:
        raise ConfigError(
            "LIGHTHOUSE_REGISTRY_TOKEN_SOURCE=env needs LIGHTHOUSE_REGISTRY_TOKEN_ENV "
            "to name the variable holding the workload JWT."
        )
    if settings.registry_token_source == "file" and not settings.registry_token_file:
        raise ConfigError(
            "LIGHTHOUSE_REGISTRY_TOKEN_SOURCE=file needs LIGHTHOUSE_REGISTRY_TOKEN_FILE "
            "to name the file holding the workload JWT."
        )

    # Naming the source is not the same as the source having anything in it, and
    # both providers read lazily -- so without these two checks the control plane
    # boots clean on a credential that cannot work and only admits it on the
    # first request, as a 502 on /models. Checked here for the same reason
    # LIGHTHOUSE_REGISTRY_CA_BUNDLE is checked below: startup is the last moment
    # an operator is still watching.
    if settings.registry_token_source == "env":
        if not (os.environ.get(settings.registry_token_env or "") or "").strip():
            raise ConfigError(
                f"LIGHTHOUSE_REGISTRY_TOKEN_ENV names {settings.registry_token_env!r}, "
                "but that variable is empty or unset. It must hold the workload JWT."
            )
    if settings.registry_token_source == "file":
        token_path = settings.registry_token_file
        if token_path is not None and not token_path.is_file():
            raise ConfigError(
                f"LIGHTHOUSE_REGISTRY_TOKEN_FILE does not exist: {token_path}"
            )

    # The flag names a workload type, not the service being called -- any of the
    # three mints the same general-purpose UMS JWT, and there is no "ML" value.
    # Rejecting anything else here saves a confusing 401 later.
    if settings.registry_workload_name not in {"DE", "DF", "OPDB"}:
        raise ConfigError(
            "LIGHTHOUSE_REGISTRY_WORKLOAD_NAME must be 'DE', 'DF' or 'OPDB', got "
            f"{settings.registry_workload_name!r}. The CDP CLI accepts no other value, "
            "and all three mint the same workload token -- the flag does not name the "
            "service being called."
        )

    if settings.registry_ca_bundle and not settings.registry_ca_bundle.is_file():
        raise ConfigError(
            f"LIGHTHOUSE_REGISTRY_CA_BUNDLE does not exist: {settings.registry_ca_bundle}"
        )
    if not settings.registry_verify_tls:
        # Not fatal -- a private CAI cluster with an internal CA is a real case --
        # but it must not pass unremarked.
        logger.warning(
            "LIGHTHOUSE_REGISTRY_VERIFY_TLS is off: registry TLS certificates are not "
            "being checked. Prefer LIGHTHOUSE_REGISTRY_CA_BUNDLE."
        )


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

    _load_registry(settings)

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
