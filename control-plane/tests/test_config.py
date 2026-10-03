"""Startup configuration, and the rules that refuse to boot (spec SS17).

Every other test module in this suite builds `Settings` directly, which means the
*translation* from environment variables into `Settings` -- and every validation
rule attached to it -- has never been exercised. That is the wrong thing to leave
untested here, because these rules are the only ones that run before there is a
process to observe. A mistake in them is not a failing request; it is a control
plane that boots in a state nobody intended.

Two classes of rule are worth the most attention:

  * **Fail-closed.** `LIGHTHOUSE_ENV=cai` makes a missing admin token fatal.
    Reaching a CAI Application from outside means enabling *unauthenticated*
    platform access, so that token is the only thing in front of the
    deploy/stop/revoke surface. Booting without it is strictly worse than not
    booting.
  * **No default for the registry host.** This repo is public and a registry
    hostname identifies a tenant, so the operator either names the host or names
    an environment to discover it from. A test pins that there is no third way.

Nothing here patches anything: `load_settings()` accepts an explicit environment
mapping, so each test hands it a dict and reads the result. The hostnames below are
deliberately in the `.invalid` TLD, which RFC 2606 reserves precisely so that a
public repository can carry an example that can never resolve to a real tenant.
"""

from __future__ import annotations

import pytest

from lighthouse.config import ConfigError, Settings, load_settings

ADMIN = "lha_configtest0123456789abcdef"

# A hostname that cannot resolve, by RFC 2606. Never a real registry.
FAKE_DOMAIN = "registry.example.invalid"


def _cai_env(**overrides: str) -> dict[str, str]:
    """A CAI environment that passes validation, overridable per test.

    Pass an empty string for a key to remove it, which is how a test asserts on
    something being *absent* rather than wrong.
    """
    env = {
        "LIGHTHOUSE_ENV": "cai",
        "LIGHTHOUSE_ADMIN_TOKEN": ADMIN,
        "LIGHTHOUSE_REGISTRY_DOMAIN": FAKE_DOMAIN,
    }
    env.update(overrides)
    return {key: value for key, value in env.items() if value}


# --------------------------------------------------------------------------
# The fail-closed rule
# --------------------------------------------------------------------------


def test_cai_without_an_admin_token_refuses_to_start():
    """The most important line in config.py.

    A CAI Application with unauthenticated platform access and no admin token is
    an open revoke button on the public internet. Refusing to start is the only
    safe behavior, and the message has to say so.
    """
    with pytest.raises(ConfigError, match="LIGHTHOUSE_ADMIN_TOKEN"):
        load_settings({"LIGHTHOUSE_ENV": "cai", "LIGHTHOUSE_REGISTRY": "fake"})


def test_local_mints_an_ephemeral_admin_token_rather_than_running_open():
    """Local dev stays usable without being silently unauthenticated -- the
    harness gets a token it can print, flagged so the operator knows it will not
    survive a restart."""
    settings = load_settings({"LIGHTHOUSE_ENV": "local"})

    assert settings.admin_token is not None
    assert settings.admin_token.startswith("lha_")
    assert settings.admin_token_ephemeral is True


def test_a_supplied_admin_token_is_not_marked_ephemeral():
    settings = load_settings({"LIGHTHOUSE_ENV": "local", "LIGHTHOUSE_ADMIN_TOKEN": ADMIN})

    assert settings.admin_token == ADMIN
    assert settings.admin_token_ephemeral is False


def test_an_unrecognized_env_is_rejected_rather_than_defaulted():
    """Defaulting an unknown env to `local` would quietly disable the fail-closed
    rule for anyone who typo'd `prod`."""
    with pytest.raises(ConfigError, match="LIGHTHOUSE_ENV"):
        load_settings({"LIGHTHOUSE_ENV": "prod"})


def test_the_artifact_corruption_hook_is_refused_outside_local():
    """A dev hook that corrupts served bytes must be impossible to enable in CAI,
    where it would look exactly like artifact tampering."""
    with pytest.raises(ConfigError, match="LIGHTHOUSE_DEV_CORRUPT_ARTIFACTS"):
        load_settings(_cai_env(LIGHTHOUSE_DEV_CORRUPT_ARTIFACTS="1"))


# --------------------------------------------------------------------------
# Choosing a registry implementation
# --------------------------------------------------------------------------


def test_cai_selects_the_real_registry_by_default():
    """Running in CAI and talking to the fake registry would be a silent demo,
    so `cai` is the default there -- the fake must be asked for explicitly."""
    assert load_settings(_cai_env()).registry_impl == "cai"


def test_local_selects_the_fake_registry_by_default():
    assert load_settings({"LIGHTHOUSE_ENV": "local"}).registry_impl == "fake"


def test_an_unknown_registry_implementation_is_rejected():
    with pytest.raises(ConfigError, match="LIGHTHOUSE_REGISTRY"):
        load_settings({"LIGHTHOUSE_ENV": "local", "LIGHTHOUSE_REGISTRY": "mlflow"})


def test_the_fake_registry_stays_usable_inside_cai_without_inventing_a_domain():
    """Validation is gated on the selected implementation, not on the env, so
    `LIGHTHOUSE_REGISTRY=fake` in CAI is a legitimate configuration that must not
    demand a registry hostname it will never call."""
    settings = load_settings(
        {"LIGHTHOUSE_ENV": "cai", "LIGHTHOUSE_ADMIN_TOKEN": ADMIN, "LIGHTHOUSE_REGISTRY": "fake"}
    )

    assert settings.registry_impl == "fake"
    assert settings.registry_domain is None


# --------------------------------------------------------------------------
# The registry host has no default, deliberately
# --------------------------------------------------------------------------


def test_naming_neither_a_domain_nor_an_environment_is_fatal():
    """There is no default registry hostname and never will be: this repo is
    public, and a hostname is a tenant identifier. The operator names the host or
    names an environment to discover it from."""
    with pytest.raises(ConfigError) as excinfo:
        load_settings(_cai_env(LIGHTHOUSE_REGISTRY_DOMAIN=""))

    message = str(excinfo.value)
    assert "LIGHTHOUSE_REGISTRY_DOMAIN" in message
    assert "LIGHTHOUSE_REGISTRY_ENVIRONMENT" in message


def test_naming_both_a_domain_and_an_environment_is_ambiguous_and_fatal():
    """Naming the host skips discovery; naming the environment performs it.
    Setting both leaves it undefined which registry was meant, and guessing would
    be a guess about which tenant to call."""
    with pytest.raises(ConfigError, match="not both"):
        load_settings(_cai_env(LIGHTHOUSE_REGISTRY_ENVIRONMENT="test-env"))


def test_naming_only_an_environment_defers_the_host_to_discovery():
    settings = load_settings(
        _cai_env(LIGHTHOUSE_REGISTRY_DOMAIN="", LIGHTHOUSE_REGISTRY_ENVIRONMENT="test-env")
    )

    assert settings.registry_environment == "test-env"
    assert settings.registry_domain is None


def test_the_api_prefix_is_normalized_to_exactly_one_leading_slash():
    """The prefix is concatenated onto a base URL, so a stray or missing slash
    produces a 404 against a path that looks right in the config."""
    assert load_settings(_cai_env()).registry_api_prefix == "/api/v2"
    assert (
        load_settings(_cai_env(LIGHTHOUSE_REGISTRY_API_PREFIX="api/v1")).registry_api_prefix
        == "/api/v1"
    )
    assert (
        load_settings(_cai_env(LIGHTHOUSE_REGISTRY_API_PREFIX="/api/v3/")).registry_api_prefix
        == "/api/v3"
    )


# --------------------------------------------------------------------------
# Where the workload token comes from
#
# Both the env and file providers read lazily, at first use. So naming a source
# that holds nothing is not caught by construction -- without validation here the
# control plane boots clean and only admits the problem on its first registry
# call, as a 502 on /models. Startup is the last moment an operator is watching.
# --------------------------------------------------------------------------


def test_an_unknown_token_source_is_rejected():
    with pytest.raises(ConfigError, match="LIGHTHOUSE_REGISTRY_TOKEN_SOURCE"):
        load_settings(_cai_env(LIGHTHOUSE_REGISTRY_TOKEN_SOURCE="vault"))


def test_token_source_env_must_name_a_variable():
    with pytest.raises(ConfigError, match="LIGHTHOUSE_REGISTRY_TOKEN_ENV"):
        load_settings(_cai_env(LIGHTHOUSE_REGISTRY_TOKEN_SOURCE="env"))


def test_token_source_env_must_name_a_variable_that_is_actually_populated():
    """Naming a source is not the same as the source holding a token. Caught at
    startup rather than on the first registry call."""
    with pytest.raises(ConfigError, match="empty or unset"):
        load_settings(
            _cai_env(
                LIGHTHOUSE_REGISTRY_TOKEN_SOURCE="env",
                LIGHTHOUSE_REGISTRY_TOKEN_ENV="WORKLOAD_JWT",
            )
        )


def test_a_whitespace_only_token_variable_counts_as_unset():
    """A variable set to the empty string by a careless template is the common
    shape of this mistake, and it must not read as a credential."""
    with pytest.raises(ConfigError, match="empty or unset"):
        load_settings(
            _cai_env(
                LIGHTHOUSE_REGISTRY_TOKEN_SOURCE="env",
                LIGHTHOUSE_REGISTRY_TOKEN_ENV="WORKLOAD_JWT",
                WORKLOAD_JWT="   ",
            )
        )


def test_a_populated_token_variable_satisfies_validation():
    """The negative control for the two rules above: a correctly configured
    token source must still load, or the checks have been over-tightened into
    blocking a valid deployment."""
    settings = load_settings(
        _cai_env(
            LIGHTHOUSE_REGISTRY_TOKEN_SOURCE="env",
            LIGHTHOUSE_REGISTRY_TOKEN_ENV="WORKLOAD_JWT",
            WORKLOAD_JWT="eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ0ZXN0In0.signature",
        )
    )

    assert settings.registry_token_source == "env"
    assert settings.registry_token_env == "WORKLOAD_JWT"


def test_token_source_file_must_name_a_file():
    with pytest.raises(ConfigError, match="LIGHTHOUSE_REGISTRY_TOKEN_FILE"):
        load_settings(_cai_env(LIGHTHOUSE_REGISTRY_TOKEN_SOURCE="file"))


def test_token_source_file_must_name_a_file_that_exists():
    with pytest.raises(ConfigError, match="does not exist"):
        load_settings(
            _cai_env(
                LIGHTHOUSE_REGISTRY_TOKEN_SOURCE="file",
                LIGHTHOUSE_REGISTRY_TOKEN_FILE="/nonexistent/workload.jwt",
            )
        )


def test_an_existing_token_file_satisfies_validation(tmp_path):
    token_file = tmp_path / "workload.jwt"
    token_file.write_text("eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ0ZXN0In0.signature")

    settings = load_settings(
        _cai_env(
            LIGHTHOUSE_REGISTRY_TOKEN_SOURCE="file",
            LIGHTHOUSE_REGISTRY_TOKEN_FILE=str(token_file),
        )
    )

    assert settings.registry_token_file == token_file


def test_the_cli_token_source_needs_no_further_configuration():
    """`cli` is the default because a CAI Session already has `cdp`; it names no
    variable and no file, so neither check may fire for it."""
    settings = load_settings(_cai_env())

    assert settings.registry_token_source == "cli"


# --------------------------------------------------------------------------
# The workload name names a workload type, not the service being called
# --------------------------------------------------------------------------


@pytest.mark.parametrize("workload", ["DE", "DF", "OPDB"])
def test_every_workload_name_the_cdp_cli_accepts_is_accepted_here(workload):
    settings = load_settings(_cai_env(LIGHTHOUSE_REGISTRY_WORKLOAD_NAME=workload))

    assert settings.registry_workload_name == workload


@pytest.mark.parametrize("workload", ["ML", "CAI", "AI"])
def test_a_workload_name_the_cdp_cli_would_reject_is_refused_at_startup(workload):
    """`ML` is the tempting wrong answer -- there is no such value, and all three
    real ones mint the same general-purpose UMS token. Rejecting here saves
    debugging a confusing 401 from the registry later."""
    with pytest.raises(ConfigError, match="LIGHTHOUSE_REGISTRY_WORKLOAD_NAME"):
        load_settings(_cai_env(LIGHTHOUSE_REGISTRY_WORKLOAD_NAME=workload))


def test_the_workload_name_is_case_normalized():
    """An operator writing `de` meant `DE`, and failing on case would be a
    pointless startup failure."""
    assert load_settings(_cai_env(LIGHTHOUSE_REGISTRY_WORKLOAD_NAME="de")).registry_workload_name == "DE"


# --------------------------------------------------------------------------
# TLS, and the rest of the numeric settings
# --------------------------------------------------------------------------


def test_a_ca_bundle_that_does_not_exist_is_fatal():
    """Discovered at startup rather than as a TLS error on the first call, where
    it reads as the registry being unreachable."""
    with pytest.raises(ConfigError, match="LIGHTHOUSE_REGISTRY_CA_BUNDLE"):
        load_settings(_cai_env(LIGHTHOUSE_REGISTRY_CA_BUNDLE="/nonexistent/ca.pem"))


def test_disabling_tls_verification_is_allowed_but_warns(caplog):
    """A private CAI cluster with an internal CA is a real case, so this is not
    fatal -- but it must never pass unremarked."""
    with caplog.at_level("WARNING"):
        settings = load_settings(_cai_env(LIGHTHOUSE_REGISTRY_VERIFY_TLS="false"))

    assert settings.registry_verify_tls is False
    assert any("VERIFY_TLS" in record.getMessage() for record in caplog.records)


def test_a_non_integer_setting_names_the_variable_that_was_wrong():
    """The operator needs to know *which* of a dozen numeric settings was bad."""
    with pytest.raises(ConfigError, match="LIGHTHOUSE_REGISTRY_REQUEST_TIMEOUT"):
        load_settings(_cai_env(LIGHTHOUSE_REGISTRY_REQUEST_TIMEOUT="soon"))


def test_the_stale_threshold_must_exceed_the_online_threshold():
    """Inverted thresholds would make a device simultaneously online and stale,
    and the connectivity view is derived from them at read time."""
    with pytest.raises(ConfigError, match="LIGHTHOUSE_STALE_THRESHOLD"):
        load_settings(
            {
                "LIGHTHOUSE_ENV": "local",
                "LIGHTHOUSE_ONLINE_THRESHOLD": "60",
                "LIGHTHOUSE_STALE_THRESHOLD": "30",
            }
        )


def test_an_unknown_artifact_transport_is_rejected():
    with pytest.raises(ConfigError, match="LIGHTHOUSE_ARTIFACT_TRANSPORT"):
        load_settings({"LIGHTHOUSE_ENV": "local", "LIGHTHOUSE_ARTIFACT_TRANSPORT": "ftp"})


# --------------------------------------------------------------------------
# Where state lives
# --------------------------------------------------------------------------


def test_cai_defaults_its_data_dir_onto_the_project_filesystem():
    """CAI projects are mounted at /home/cdsw and persist across restarts. A
    container-local default would silently lose every enrollment on redeploy."""
    settings = load_settings(_cai_env())

    assert settings.data_dir.as_posix() == "/home/cdsw/.lighthouse"
    assert settings.db_path.as_posix() == "/home/cdsw/.lighthouse/lighthouse.db"


def test_an_explicit_data_dir_wins_over_the_cai_default(tmp_path):
    settings = load_settings(_cai_env(LIGHTHOUSE_DATA_DIR=str(tmp_path)))

    assert settings.data_dir == tmp_path


def test_loading_settings_does_not_leak_into_the_real_environment():
    """`load_settings` swaps `os.environ` to evaluate the mapping it was given.
    If it failed to restore it, one test would silently reconfigure every test
    that ran after it -- the kind of failure that presents as flakiness."""
    import os

    before = dict(os.environ)
    load_settings(_cai_env(LIGHTHOUSE_REGISTRY_WORKLOAD_NAME="OPDB"))

    assert dict(os.environ) == before


def test_cors_origins_are_split_and_stripped():
    settings = load_settings(
        {"LIGHTHOUSE_ENV": "local", "LIGHTHOUSE_CORS_ORIGINS": "http://a.invalid, http://b.invalid"}
    )

    assert settings.cors_allow_origins == ["http://a.invalid", "http://b.invalid"]


def test_the_defaults_match_the_dataclass():
    """A sanity check that `load_settings` with a bare env agrees with the
    `Settings` defaults, so the two definitions cannot drift apart unnoticed."""
    loaded = load_settings({"LIGHTHOUSE_ENV": "local"})
    defaults = Settings()

    assert loaded.registry_api_prefix == defaults.registry_api_prefix
    assert loaded.registry_request_timeout == defaults.registry_request_timeout
    assert loaded.registry_stream_timeout == defaults.registry_stream_timeout
    assert loaded.registry_cache_ttl_seconds == defaults.registry_cache_ttl_seconds
    assert loaded.artifact_chunk_size == defaults.artifact_chunk_size
