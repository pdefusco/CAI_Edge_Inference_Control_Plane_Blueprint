"""Startup configuration -- the only code in the agent that runs before logging.

There was no test file for `config.py` until this one, which is why a stale
`os.environ` read inside `_env_int` survived: `load_settings(environ=...)` has an
injectable mapping precisely so configuration can be tested without touching the
process, and for the five integer settings it silently did not work. Nothing here
needs `monkeypatch` except the tests that exist to prove the ambient environment is
*ignored*.

Why this file matters more than its size suggests: every failure below is a startup
failure on a headless device. A misconfiguration that raises `ConfigError` becomes
`main.py`'s exit 2 and a one-line journal entry an operator can read; anything that
raises something else escapes that handler and becomes a traceback, or worse, a
default that quietly works differently than the operator asked for.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from keeper.config import AgentSettings, ConfigError, load_settings
from keeper.runtime import MockRuntime, build_runtime
from keeper.runtime.base import ModelLoadError

MINIMAL = {
    "KEEPER_DEVICE_ID": "jetson-01",
    "KEEPER_CONTROL_PLANE_URL": "https://lighthouse.invalid",
    "KEEPER_TOKEN": "lhd_0123456789abcdef.secret",
}


def env(**overrides: str) -> dict[str, str]:
    """A valid environment plus whatever this test is about.

    Built fresh per call so no test can mutate another's; `None` values are a
    deliberate way to *remove* a required key.
    """
    merged = {**MINIMAL, **overrides}
    return {k: v for k, v in merged.items() if v is not None}  # type: ignore[comparison-overlap]


class TestRequiredSettings:
    def test_the_minimal_environment_is_enough(self) -> None:
        settings = load_settings(env())
        assert settings.device_id == "jetson-01"
        assert settings.control_plane_url == "https://lighthouse.invalid"
        assert settings.token == "lhd_0123456789abcdef.secret"

    @pytest.mark.parametrize("missing", ["KEEPER_DEVICE_ID", "KEEPER_CONTROL_PLANE_URL"])
    def test_a_required_setting_names_itself_when_absent(self, missing: str) -> None:
        """The error text is the whole user interface here -- it is what an
        operator sees in `journalctl` after a failed `systemctl start`."""
        broken = env()
        del broken[missing]
        with pytest.raises(ConfigError, match=missing):
            load_settings(broken)

    @pytest.mark.parametrize("blank", ["", "   ", "\n"])
    def test_whitespace_is_not_a_value(self, blank: str) -> None:
        """An `EnvironmentFile` line like `KEEPER_DEVICE_ID=` sets the variable to
        the empty string rather than leaving it unset, so "present" is not the same
        question as "configured"."""
        with pytest.raises(ConfigError, match="KEEPER_DEVICE_ID"):
            load_settings(env(KEEPER_DEVICE_ID=blank))

    def test_surrounding_whitespace_is_stripped(self) -> None:
        """A token pasted into a file or an env file usually arrives with a
        newline. Sending that in an `Authorization` header is a 401 nobody can
        debug from the device side."""
        settings = load_settings(
            env(KEEPER_DEVICE_ID="  jetson-01  ", KEEPER_TOKEN="  lhd_a.b\n")
        )
        assert settings.device_id == "jetson-01"
        assert settings.token == "lhd_a.b"

    @pytest.mark.parametrize(
        "url",
        ["lighthouse.invalid", "ftp://lighthouse.invalid", "//lighthouse.invalid", "localhost:8000"],
    )
    def test_a_url_without_an_http_scheme_is_refused(self, url: str) -> None:
        """`config.url()` only joins strings; a schemeless base would fail much
        later, inside httpx, on the first poll of a device in the field."""
        with pytest.raises(ConfigError, match="http"):
            load_settings(env(KEEPER_CONTROL_PLANE_URL=url))

    def test_plain_http_is_allowed(self) -> None:
        """Deliberate: the dev harness and an on-LAN fallback both use it. The
        loud part is `KEEPER_VERIFY_TLS`, not the scheme."""
        assert load_settings(env(KEEPER_CONTROL_PLANE_URL="http://localhost:8000"))


class TestToken:
    """The token is a permanent deployment credential, so where it comes from is a
    security decision, not a convenience. A file is recommended over an env var
    because `/proc/<pid>/environ` and `systemctl show` both expose the latter."""

    def test_it_can_come_from_a_file(self, tmp_path: Path) -> None:
        path = tmp_path / "token"
        path.write_text("lhd_fromfile.secret\n")
        broken = env()
        del broken["KEEPER_TOKEN"]
        settings = load_settings({**broken, "KEEPER_TOKEN_FILE": str(path)})
        assert settings.token == "lhd_fromfile.secret"

    def test_the_env_var_wins_when_both_are_set(self, tmp_path: Path) -> None:
        """Documenting the precedence rather than asserting it is right: the file
        is never read when `KEEPER_TOKEN` is set, so a stale env var silently
        shadows a freshly rotated token file. Worth knowing before debugging a
        rotation that "did not take"."""
        path = tmp_path / "token"
        path.write_text("lhd_fromfile.secret")
        settings = load_settings(env(KEEPER_TOKEN_FILE=str(path)))
        assert settings.token == "lhd_0123456789abcdef.secret"

    def test_a_missing_token_file_is_a_config_error_not_an_oserror(
        self, tmp_path: Path
    ) -> None:
        """So it lands in `main.py`'s exit-2 handler instead of a traceback."""
        broken = env()
        del broken["KEEPER_TOKEN"]
        with pytest.raises(ConfigError, match="KEEPER_TOKEN_FILE"):
            load_settings({**broken, "KEEPER_TOKEN_FILE": str(tmp_path / "absent")})

    def test_neither_source_is_an_error_that_names_both(self) -> None:
        broken = env()
        del broken["KEEPER_TOKEN"]
        with pytest.raises(ConfigError) as caught:
            load_settings(broken)
        assert "KEEPER_TOKEN" in str(caught.value)
        assert "KEEPER_TOKEN_FILE" in str(caught.value)

    def test_an_empty_token_file_is_refused(self, tmp_path: Path) -> None:
        """A zero-byte file is the shape a failed out-of-band token transport
        leaves behind -- `scp` of a file that was never written, or a truncated
        write. It must not read as "no token configured but carry on"."""
        path = tmp_path / "token"
        path.write_text("\n")
        broken = env()
        del broken["KEEPER_TOKEN"]
        with pytest.raises(ConfigError, match="required"):
            load_settings({**broken, "KEEPER_TOKEN_FILE": str(path)})

    def test_the_token_is_not_in_the_error_text_when_the_url_is_bad(self) -> None:
        """Errors from here are printed to stderr and land in the journal, which
        `main.py`'s redacting filter does not cover -- it filters log records, and
        this print happens before logging is configured at all."""
        with pytest.raises(ConfigError) as caught:
            load_settings(env(KEEPER_CONTROL_PLANE_URL="nope"))
        assert "lhd_" not in str(caught.value)


class TestRuntimeSelection:
    """`KEEPER_RUNTIME` is the switch between a device that really infers and one
    that pretends. Mistyping it must not silently fall back to the pretender."""

    @pytest.mark.parametrize("raw", ["onnx", "ONNX", " onnx ", "Onnx\n"])
    def test_it_is_case_and_whitespace_insensitive(self, raw: str) -> None:
        assert load_settings(env(KEEPER_RUNTIME=raw)).runtime_impl == "onnx"

    def test_it_defaults_to_mock(self) -> None:
        """So the dev harness works with no configuration, and so a device that
        was *meant* to be on onnx reports `runtime=mock` in its heartbeat rather
        than failing to start."""
        assert load_settings(env()).runtime_impl == "mock"

    @pytest.mark.parametrize("raw", ["onnxruntime", "tensorrt", "none", "true", "onnx mock"])
    def test_an_unknown_runtime_is_refused_at_startup(self, raw: str) -> None:
        """Not at first inference. `tensorrt` is the interesting one: it is spec
        Phase 8 and a plausible thing for an operator to try early."""
        with pytest.raises(ConfigError, match="KEEPER_RUNTIME"):
            load_settings(env(KEEPER_RUNTIME=raw))

    def test_an_empty_value_takes_the_default(self) -> None:
        """`KEEPER_RUNTIME=` in an env file, again."""
        assert load_settings(env(KEEPER_RUNTIME="")).runtime_impl == "mock"

    def test_build_runtime_dispatches_mock(self) -> None:
        assert isinstance(build_runtime("mock"), MockRuntime)

    def test_build_runtime_reaches_the_onnx_branch(self) -> None:
        """Asserted without requiring the wheel, because this test runs in the
        default `make test` on a laptop with no ML stack.

        Both outcomes prove dispatch worked: with onnxruntime installed the
        constructor returns, and without it the lazy import raises
        `ModelLoadError` from `OnnxRuntime.__init__`. What would *not* be accepted
        is `ValueError`, which is what the unknown-implementation fall-through
        raises -- so a typo'd branch name still fails here.
        """
        try:
            runtime = build_runtime("onnx")
        except ModelLoadError as exc:
            assert "onnxruntime" in str(exc)
        else:
            assert runtime.name == "onnxruntime"

    def test_build_runtime_refuses_an_unknown_implementation(self) -> None:
        """Belt and braces: `load_settings` should have caught it first, so
        reaching this is a bug rather than a misconfiguration -- hence
        `ValueError` and not `ConfigError`."""
        with pytest.raises(ValueError, match="unknown runtime"):
            build_runtime("tensorrt")


class TestIntegerSettings:
    """The five timing settings, and the bug this file was written to catch."""

    def test_the_injected_environ_configures_them(self) -> None:
        settings = load_settings(
            env(
                KEEPER_POLL_INTERVAL="3",
                KEEPER_DOWNLOAD_TIMEOUT="60",
                KEEPER_REQUEST_TIMEOUT="5",
                KEEPER_RETRY_BACKOFF_INITIAL="2",
                KEEPER_RETRY_BACKOFF_MAX="20",
            )
        )
        assert settings.poll_interval_seconds == 3
        assert settings.download_timeout_seconds == 60
        assert settings.request_timeout_seconds == 5
        assert settings.retry_backoff_initial_seconds == 2
        assert settings.retry_backoff_max_seconds == 20

    def test_the_process_environment_is_ignored_when_one_is_injected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The regression. `_env_int` read `os.environ` directly, so an ambient
        `KEEPER_POLL_INTERVAL` overrode an injected one -- which makes every test
        in this file depend on the developer's shell, and makes the injectable
        mapping a half-truth."""
        monkeypatch.setenv("KEEPER_POLL_INTERVAL", "999")
        monkeypatch.setenv("KEEPER_LOG_LEVEL", "CRITICAL")
        settings = load_settings(env(KEEPER_POLL_INTERVAL="3"))
        assert settings.poll_interval_seconds == 3
        assert settings.log_level == "INFO"

    @pytest.mark.parametrize("raw", ["ten", "3.5", "10s", "-"])
    def test_a_non_integer_names_the_variable_and_the_value(self, raw: str) -> None:
        with pytest.raises(ConfigError) as caught:
            load_settings(env(KEEPER_POLL_INTERVAL=raw))
        assert "KEEPER_POLL_INTERVAL" in str(caught.value)
        assert raw in str(caught.value)

    def test_an_empty_value_takes_the_default(self) -> None:
        assert load_settings(env(KEEPER_POLL_INTERVAL="")).poll_interval_seconds == 10

    def test_a_negative_value_is_accepted_today(self) -> None:
        """Pinning the current behaviour, not endorsing it. A negative poll
        interval reaches `Event.wait(-1)`, which returns immediately and spins the
        loop as fast as the control plane will answer. Not worth a validation rule
        on its own, but worth a test that will notice if one is added.
        """
        assert load_settings(env(KEEPER_POLL_INTERVAL="-1")).poll_interval_seconds == -1


class TestDerivedPaths:
    def test_the_data_dir_drives_everything_under_it(self, tmp_path: Path) -> None:
        settings = load_settings(env(KEEPER_DATA_DIR=str(tmp_path)))
        assert settings.artifact_dir == tmp_path / "artifacts"
        assert settings.model_dir == tmp_path / "models"
        assert settings.state_path == tmp_path / "state.json"

    def test_a_tilde_is_expanded(self) -> None:
        """systemd units do not expand `~`, and neither does `Path`. An operator
        testing by hand before writing the unit file will use one."""
        settings = load_settings(env(KEEPER_DATA_DIR="~/keeper-data"))
        assert "~" not in str(settings.data_dir)
        assert settings.data_dir.is_absolute()

    def test_the_default_is_the_systemd_state_directory(self) -> None:
        """Matches `StateDirectory=keeper` in the unit file. If one moves, so must
        the other, or the agent loses its state on every deploy."""
        assert load_settings(env()).data_dir == Path("/var/lib/keeper")


class TestTlsAndLogging:
    @pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off", " Off ", "FALSE\n"])
    def test_tls_verification_can_be_turned_off_explicitly(self, raw: str) -> None:
        """Case and surrounding whitespace do not change the meaning. The last two
        cases are the ones that matter in practice: an env file or a hand-edited
        unit leaves a trailing space or newline, and a flag that meant the
        *opposite* thing because of invisible whitespace would be a nasty way to
        find out the device token had been exposed."""
        assert load_settings(env(KEEPER_VERIFY_TLS=raw)).verify_tls is False

    @pytest.mark.parametrize("raw", ["", "true", "1", "yes", "maybe", "disabled", "nope"])
    def test_anything_else_leaves_it_on(self, raw: str) -> None:
        """Fail secure. `"disabled"` is the trap -- it reads like an opt-out and is
        not one, which is the correct direction for a flag whose only effect is to
        expose the device token to anyone on the path. An operator who meant it has
        to use one of the six words above, and `main.py:95-99` then logs a warning
        every start."""
        assert load_settings(env(KEEPER_VERIFY_TLS=raw)).verify_tls is True

    def test_the_log_level_is_uppercased(self) -> None:
        """`configure_logging` does `getattr(logging, level)`, so `"debug"` would
        silently become `INFO` via its default."""
        assert load_settings(env(KEEPER_LOG_LEVEL="debug")).log_level == "DEBUG"

    def test_an_unknown_log_level_is_not_a_startup_failure(self) -> None:
        """Deliberate asymmetry with `KEEPER_RUNTIME`: a bad log level costs
        verbosity, a bad runtime costs correctness. `configure_logging` falls back
        to INFO."""
        assert load_settings(env(KEEPER_LOG_LEVEL="chatty")).log_level == "CHATTY"


class TestUrlJoin:
    """`AgentSettings.url` is the only place the base URL appears, which is why the
    control plane can return relative artifact URIs that work through the CAI
    domain, a tunnel, or localhost without the server knowing which."""

    @pytest.mark.parametrize(
        ("base", "path"),
        [
            ("https://host.invalid", "/api/v1/health"),
            ("https://host.invalid/", "/api/v1/health"),
            ("https://host.invalid", "api/v1/health"),
            ("https://host.invalid/", "api/v1/health"),
        ],
    )
    def test_exactly_one_slash_survives_the_join(self, base: str, path: str) -> None:
        settings = AgentSettings(device_id="d", control_plane_url=base, token="t")
        assert settings.url(path) == "https://host.invalid/api/v1/health"

    def test_a_base_path_is_preserved(self) -> None:
        """A CAI Application can be served under a prefix, and an artifact URI
        joined onto a truncated base would 404 on the device only."""
        settings = AgentSettings(
            device_id="d", control_plane_url="https://host.invalid/lighthouse", token="t"
        )
        assert settings.url("/api/v1/health") == "https://host.invalid/lighthouse/api/v1/health"

    def test_a_query_string_in_the_path_is_untouched(self) -> None:
        """Artifact URIs carry `?model=&version=`, so this is the real shape."""
        settings = AgentSettings(
            device_id="d", control_plane_url="https://host.invalid", token="t"
        )
        joined = settings.url("/api/v1/devices/d/artifact?model=fashion-cnn&version=1")
        assert joined.endswith("/artifact?model=fashion-cnn&version=1")
