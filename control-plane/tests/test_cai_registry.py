"""Tests for the real CAI registry adapter (`lighthouse.registry.cai`).

New pattern for this repo: `httpx.MockTransport`.

`tests/conftest.py` states the house rule in prose: "Nothing here mocks an
HTTP layer, so a route that only works because a test patched something
around it cannot pass." That rule is about patching something *around* the
code under test -- swapping out a method, an instance attribute, a seam the
production code never actually exercises -- so that a bug in the real call
path goes unnoticed because the test never took that path either.

`httpx.MockTransport` is different in kind, not just in degree, from that.
`CAIRegistryClient` still builds a genuine `httpx.Client`; nothing about
request construction, header assembly, status-code branching, redirect
following, or response-body iteration is bypassed or stubbed. The only thing
swapped out is the socket -- the one part of the stack this repo has no
business dialing out through in a test run anyway, and whose unavailability
(no live registry, no network, no tenant hostname committed to this public
repo) is exactly why CAI needs an adapter instead of a direct dependency in
the first place. The handler passed to `MockTransport` plays the server: it
receives the real `httpx.Request` the client built -- real method, real URL,
real headers, real query params -- and hands back a real `httpx.Response`.
`CAIRegistryClient._send`, `.send_stream`, and `.unauthenticated_get` all run
their actual code, over actual httpx objects. The thing under test in this
file *is* the HTTP layer: retry-after-401, redirect-without-Authorization,
timeout-to-`RegistryUnavailable`, and multipart body extraction all live
inside those methods and `_open_artifact_stream`. The seam this repo already
prefers elsewhere -- subclassing and overriding a method -- would have to
override exactly those methods to get coverage here, which would delete the
behavior being verified rather than exercise it. MockTransport is the one
way to drive this module's real HTTP handling without a socket, which is why
it is introduced here and nowhere else in the suite.

Every collaborator is wired in through a constructor the adapter already
exposes -- `CAIRegistryClient(settings, domain, token_provider, client=...)`,
`CAIModelRegistry(settings, client, clock=...)`, `CdpCliTokenProvider(...,
runner=...)`, and `discover_domain(environment, runner)` -- never by patching
an attribute onto an instance after construction. No `unittest.mock`, no
`monkeypatch` on adapter internals, and `boto3` is never imported (it is not
installed; this module does not need it either, see its own docstring).
"""

from __future__ import annotations

import gzip
import json
import logging
import subprocess
import time
from pathlib import Path

import httpx
import pytest
from lighthouse_contracts import ArtifactFormat, Packaging

from lighthouse.config import ConfigError, Settings
from lighthouse.registry.base import (
    ModelNotFound,
    RegistryAuthError,
    RegistryModelVersion,
    RegistryUnavailable,
    UnsupportedFlavor,
    VersionFailed,
    VersionNotReady,
)
from lighthouse.registry.cai import (
    CAIModelRegistry,
    CAIRegistryClient,
    CdpCliTokenProvider,
    _format_and_packaging,
    discover_domain,
)

PREFIX = "/api/v2"
DOMAIN = "registry.example.invalid"

# --- test doubles, wired in through real constructors -----------------------


class _FakeTokenProvider:
    """A `TokenProvider` test double: hands back a scripted token (or token
    sequence) and counts how many times it was told to discard it.

    This is the collaborator `CAIRegistryClient`'s constructor already
    accepts -- not a patch, a different real implementation of the same
    `TokenProvider` protocol `CdpCliTokenProvider` implements for production.
    """

    def __init__(self, tokens: list[str] | None = None) -> None:
        self._tokens = tokens if tokens is not None else ["fake-bearer-token"]
        self._calls = 0
        self.invalidate_count = 0

    def token(self) -> str:
        index = min(self._calls, len(self._tokens) - 1)
        self._calls += 1
        return self._tokens[index]

    def invalidate(self) -> None:
        self.invalidate_count += 1


def _settings(**overrides: object) -> Settings:
    """A `Settings` with just enough of the CAI adapter's own fields set to
    build a client -- independent of `config.py`'s cli/env/file validation,
    which this suite never calls (the token provider is injected directly)."""
    kwargs: dict[str, object] = {"registry_domain": DOMAIN}
    kwargs.update(overrides)
    return Settings(**kwargs)  # type: ignore[arg-type]


def _client_for(
    settings: Settings,
    handler,
    *,
    token_provider: _FakeTokenProvider | None = None,
) -> tuple[CAIRegistryClient, _FakeTokenProvider]:
    token_provider = token_provider or _FakeTokenProvider()
    http_client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    client = CAIRegistryClient(settings, DOMAIN, token_provider, client=http_client)
    return client, token_provider


def _registry_for(
    settings: Settings,
    handler,
    *,
    token_provider: _FakeTokenProvider | None = None,
    clock=None,
) -> tuple[CAIModelRegistry, _FakeTokenProvider]:
    client, token_provider = _client_for(settings, handler, token_provider=token_provider)
    registry = CAIModelRegistry(settings, client, clock=clock or time.monotonic)
    return registry, token_provider


def _route(routes: dict):
    """Build a `MockTransport` handler from an exact `(method, path) ->
    response-or-callable` table. A callable entry receives the `httpx.Request`
    (so a route can inspect query params, headers, or close over a counter)
    and must return an `httpx.Response`."""

    def handle(request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        target = routes.get(key)
        if target is None:
            raise AssertionError(f"unexpected request: {request.method} {request.url}")
        return target(request) if callable(target) else target

    return handle


# --- wire-shape builders, matching the registry's own swagger.json ----------


def _model_entry(model_id: str, name: str) -> dict:
    """A `ModelWithoutVersion` listing entry. `id`, not `model_id`."""
    return {
        "id": model_id,
        "name": name,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "creator": {"user_name": "test-user"},
        "description": "",
        "tags": [],
        "visibility": "PRIVATE",
    }


def _version_entry(
    version: int,
    *,
    status: str = "READY",
    created_at: str = "2024-01-01T00:00:00Z",
    model_id: str = "model-1",
    model_name: str = "fashion-cnn",
    metadata: dict | None = None,
    tags: list | None = None,
    artifact_uri: str = "s3a://bucket/fashion-cnn/1",
    error_message: str | None = None,
) -> dict:
    """A `ModelVersion`. `version` is an integer on the wire, on purpose."""
    return {
        "artifact_uri": artifact_uri,
        "created_at": created_at,
        "error_message": error_message,
        "metadata": metadata if metadata is not None else {"model_repo_type": "mlflow", "mlflowMetadata": {}},
        "model_id": model_id,
        "model_name": model_name,
        "notes": None,
        "status": status,
        "tags": tags or [],
        "updated_at": created_at,
        "user": {"user_name": "test-user"},
        "version": version,
    }


# --- listing and pagination -------------------------------------------------


def test_an_empty_registry_lists_no_models():
    """The live registry answers an empty catalog with `{"models": null}`, not
    `{"models": []}` -- a `.get("models") or []` elsewhere would already save
    this, but `list_models()` is the only place that promise is actually kept."""
    handler = _route({("GET", f"{PREFIX}/models"): httpx.Response(200, json={"models": None})})
    registry, _ = _registry_for(_settings(), handler)

    assert registry.list_models() == []


def test_a_paginated_listing_is_followed_to_the_last_page():
    """`_fetch_all_models` must keep paging on `next_page_token` until the
    registry stops supplying one, or a tenant with more than one page of
    models would silently lose everything past the first."""

    def models_route(request: httpx.Request) -> httpx.Response:
        token = request.url.params.get("page_token")
        if token is None:
            return httpx.Response(
                200,
                json={"models": [_model_entry("m1", "fashion-cnn")], "next_page_token": "page-2"},
            )
        assert token == "page-2"
        return httpx.Response(200, json={"models": [_model_entry("m2", "pose-net")], "next_page_token": ""})

    handler = _route({("GET", f"{PREFIX}/models"): models_route})
    registry, _ = _registry_for(_settings(), handler)

    assert registry.list_models() == ["fashion-cnn", "pose-net"]


def test_a_page_token_that_never_advances_does_not_spin_forever():
    """The request parameter that sends `next_page_token` back is unverified
    against the real registry. A handler that answers the same token on every
    page -- as a server would if it never recognized the parameter at all --
    must make paging stop at "first page only", not loop until the 50-page cap
    (or longer, if that cap is ever raised)."""
    call_count = 0

    def models_route(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(
            200,
            json={"models": [_model_entry("m1", "fashion-cnn")], "next_page_token": "stuck"},
        )

    handler = _route({("GET", f"{PREFIX}/models"): models_route})
    registry, _ = _registry_for(_settings(), handler)

    assert registry.list_models() == ["fashion-cnn"]
    assert call_count == 2  # first page, then the repeat that proves the token never advanced


# --- version ordering, identity, and lineage ---------------------------------


def test_versions_come_back_oldest_first_even_when_the_registry_reorders_them():
    """`ModelRegistry.list_versions` contracts oldest-first, and nothing above
    this adapter sorts -- so if the registry's own `model_versions` array
    comes back in any other order, this is the only place that gets fixed."""
    model_versions = [
        _version_entry(3, created_at="2024-03-01T00:00:00Z"),
        _version_entry(1, created_at="2024-01-01T00:00:00Z"),
        _version_entry(2, created_at="2024-02-01T00:00:00Z"),
    ]
    handler = _route(
        {
            ("GET", f"{PREFIX}/models"): httpx.Response(200, json={"models": [_model_entry("model-1", "fashion-cnn")]}),
            ("GET", f"{PREFIX}/models/model-1"): httpx.Response(200, json={"model_versions": model_versions}),
        }
    )
    registry, _ = _registry_for(_settings(), handler)

    versions = registry.list_versions("fashion-cnn")

    assert [v.version for v in versions] == ["1", "2", "3"]


def test_an_integer_version_becomes_a_string_at_the_boundary():
    """`RegistryModelVersion.version` is a `str` everywhere above this module;
    the wire's `version` field is a JSON integer. The boundary between the two
    is exactly `_parse_version`, so this pins that it actually converts rather
    than passing an `int` through and failing far away, in string-keyed code."""
    handler = _route(
        {
            ("GET", f"{PREFIX}/models"): httpx.Response(200, json={"models": [_model_entry("model-1", "fashion-cnn")]}),
            ("GET", f"{PREFIX}/models/model-1/versions/7"): httpx.Response(200, json=_version_entry(7)),
        }
    )
    registry, _ = _registry_for(_settings(), handler)

    mv = registry.get_version("fashion-cnn", "7")

    assert mv.version == "7"
    assert isinstance(mv.version, str)


def test_the_lineage_key_changes_when_a_version_number_is_reused():
    """The most important test in this file.

    CAI has no stable per-version content id -- a version is just an integer,
    and `DELETE /models/{id}/versions/{version}` means that integer can be
    deleted and reissued against entirely different bytes later. If the same
    `(model_id, version)` always produced the same `cache_key`,
    `services/deployment_service.py`'s `mv.cache_key != row.cache_key` drift
    check would PASS across a reissue -- the control plane would keep serving
    a device the stale, previously cached artifact under a label that looks
    unchanged. Composing the key with `created_at` (never cached, see
    `get_version`'s own docstring) is what makes a reissue read as a new
    lineage. This also doubles as the "get_version is never cached" check:
    two calls for the same (model, version) make two real requests.
    """
    call_count = 0

    def version_route(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        created_at = "2024-01-01T00:00:00Z" if call_count == 1 else "2024-06-01T00:00:00Z"
        return httpx.Response(200, json=_version_entry(7, created_at=created_at))

    handler = _route(
        {
            ("GET", f"{PREFIX}/models"): httpx.Response(200, json={"models": [_model_entry("model-1", "fashion-cnn")]}),
            ("GET", f"{PREFIX}/models/model-1/versions/7"): version_route,
        }
    )
    registry, _ = _registry_for(_settings(), handler)

    first = registry.get_version("fashion-cnn", "7")
    second = registry.get_version("fashion-cnn", "7")

    assert first.version == second.version == "7"
    assert first.model_id == second.model_id
    assert first.cache_key != second.cache_key
    assert call_count == 2


def test_a_model_registered_after_the_cache_was_filled_is_still_found():
    """The name->id map is cached for `registry_cache_ttl_seconds`. A model
    registered seconds after that cache filled must not have to wait out the
    TTL -- `_resolve_model_id` refreshes once on a miss before raising
    `ModelNotFound`, and this is the test that would catch a regression
    turning that into an unconditional raise."""
    registered: list[dict] = [_model_entry("model-1", "fashion-cnn")]

    def models_route(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": list(registered)})

    handler = _route(
        {
            ("GET", f"{PREFIX}/models"): models_route,
            ("GET", f"{PREFIX}/models/model-2/versions/1"): httpx.Response(200, json=_version_entry(1, model_id="model-2", model_name="pose-net")),
        }
    )
    registry, _ = _registry_for(_settings(), handler)

    registry.list_models()  # fills the name->id cache with just fashion-cnn
    registered.append(_model_entry("model-2", "pose-net"))  # registered just now

    mv = registry.get_version("pose-net", "1")

    assert mv.name == "pose-net"
    assert mv.model_id == "model-2"


# --- status mapping -----------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected_exception"),
    [
        ("REGISTERING", VersionNotReady),
        ("UPLOADING", VersionNotReady),
        ("UNKNOWN", VersionNotReady),
        ("UPLOAD_FAILED", VersionFailed),
        ("DELETE_FAILED", VersionFailed),
        ("DELETED", ModelNotFound),
        ("DELETING", ModelNotFound),
        ("READY", None),
    ],
)
def test_each_registry_status_maps_to_the_error_an_operator_can_act_on(status, expected_exception):
    """All eight `Status` values, swept in one place, because the remedy an
    operator needs is different for each bucket: retry later (not-ready),
    register a new version (failed), or stop looking (gone). Collapsing any
    two of these into the same exception would send an operator to the wrong
    dashboard."""
    handler = _route(
        {
            ("GET", f"{PREFIX}/models"): httpx.Response(200, json={"models": [_model_entry("model-1", "fashion-cnn")]}),
            ("GET", f"{PREFIX}/models/model-1/versions/1"): httpx.Response(200, json=_version_entry(1, status=status, error_message="boom")),
        }
    )
    registry, _ = _registry_for(_settings(), handler)

    if expected_exception is None:
        mv = registry.get_version("fashion-cnn", "1")
        assert mv.status == "READY"
    else:
        with pytest.raises(expected_exception):
            registry.get_version("fashion-cnn", "1")


def test_a_broken_version_still_appears_in_the_listing():
    """`list_versions` raises nothing for a failed version -- it passes the
    raw status through so the dashboard can show the broken version to an
    operator. Only `get_version`, the deploy-time gate, refuses it."""
    handler = _route(
        {
            ("GET", f"{PREFIX}/models"): httpx.Response(200, json={"models": [_model_entry("model-1", "fashion-cnn")]}),
            ("GET", f"{PREFIX}/models/model-1"): httpx.Response(
                200, json={"model_versions": [_version_entry(1, status="UPLOAD_FAILED")]}
            ),
        }
    )
    registry, _ = _registry_for(_settings(), handler)

    versions = registry.list_versions("fashion-cnn")

    assert len(versions) == 1
    assert versions[0].status == "UPLOAD_FAILED"


def test_an_hf_sourced_version_is_refused_before_anything_is_downloaded():
    """A Hugging Face-sourced version carries no ONNX flavor information at
    all. `get_version` must refuse it with `UnsupportedFlavor` at the deploy
    gate -- before any artifact bytes are requested -- rather than let a
    device discover the mismatch after downloading something it cannot run."""
    handler = _route(
        {
            ("GET", f"{PREFIX}/models"): httpx.Response(200, json={"models": [_model_entry("model-1", "llm-chat")]}),
            ("GET", f"{PREFIX}/models/model-1/versions/1"): httpx.Response(
                200,
                json=_version_entry(1, model_name="llm-chat", metadata={"huggingface_metadata": {"repo": "org/model"}}),
            ),
        }
    )
    registry, _ = _registry_for(_settings(), handler)

    with pytest.raises(UnsupportedFlavor):
        registry.get_version("llm-chat", "1")


def test_tags_arrive_as_pairs_and_become_a_mapping():
    """`Tag` is an array of `{"key", "value"}` objects on the wire, not a map
    -- `_tags_from_wire` is the only place that conversion happens, so a
    version's tags must come back as a plain `dict` to every caller above."""
    handler = _route(
        {
            ("GET", f"{PREFIX}/models"): httpx.Response(200, json={"models": [_model_entry("model-1", "fashion-cnn")]}),
            ("GET", f"{PREFIX}/models/model-1/versions/1"): httpx.Response(
                200,
                json=_version_entry(1, tags=[{"key": "team", "value": "vision"}, {"key": "stage", "value": "prod"}]),
            ),
        }
    )
    registry, _ = _registry_for(_settings(), handler)

    mv = registry.get_version("fashion-cnn", "1")

    assert mv.tags == {"team": "vision", "stage": "prod"}


# --- what `metadata` has to say for a version to be deployable ----------------
#
# `_format_and_packaging` is the whole of the deployability decision before any
# bytes exist, and it is the one part of this adapter that was written against a
# registry nobody had put a model into yet. These tests pin the parts that are
# knowable without the tenant, and leave one explicitly-marked slot for the part
# that is not.


def _version_with(metadata: dict):
    """A registry serving one version whose metadata is exactly `metadata`."""
    return _route(
        {
            ("GET", f"{PREFIX}/models"): httpx.Response(
                200, json={"models": [_model_entry("model-1", "fashion-cnn")]}
            ),
            ("GET", f"{PREFIX}/models/model-1/versions/1"): httpx.Response(
                200, json=_version_entry(1, metadata=metadata)
            ),
        }
    )


@pytest.mark.parametrize(
    "repo_type",
    ["mlflow", "MLFLOW", "MLflow", "MLFLOW_MODEL", "MODEL_REPO_TYPE_MLFLOW", " mlflow "],
)
def test_an_enum_shaped_repo_type_still_reads_as_mlflow(repo_type):
    """The failure this is here to prevent is silent and total.

    `model_repo_type` was only ever observed as the string `"mlflow"`. An API
    that returns a bare lowercase value in one place very often returns
    `"MLFLOW_MODEL"` in another, and an exact-equality test answers UNKNOWN for
    that -- which `get_version` turns into `UnsupportedFlavor` for *every*
    model in the registry. Nothing about that failure points at this line.

    No `mlflowMetadata` key here on purpose: with one, the second half of the
    condition would carry the test and the repo_type branch would go
    unexercised.
    """
    registry, _ = _registry_for(_settings(), _version_with({"model_repo_type": repo_type}))

    mv = registry.get_version("fashion-cnn", "1")

    assert mv.format is ArtifactFormat.ONNX
    assert mv.packaging is Packaging.MLFLOW_TAR_GZ


@pytest.mark.parametrize("repo_type", ["hf", "HUGGINGFACE_MODEL", "ngc", "NGC_MODEL"])
def test_an_enum_shaped_hf_or_ngc_repo_type_is_still_refused(repo_type):
    """Documents rather than fixes: these already reached UNKNOWN by falling
    through. Pinned because the token matching above now claims them
    deliberately, and a later edit that broadened the MLflow branch too far
    would otherwise flip them to deployable with nothing to catch it."""
    registry, _ = _registry_for(_settings(), _version_with({"model_repo_type": repo_type}))

    with pytest.raises(UnsupportedFlavor):
        registry.get_version("fashion-cnn", "1")


def test_a_repo_type_that_merely_contains_hf_is_not_read_as_huggingface():
    """Why tokens and not a substring test. `"hf"` inside a longer word is a
    false positive, and this one would refuse an MLflow model."""
    registry, _ = _registry_for(
        _settings(), _version_with({"model_repo_type": "mlflow-shfmt-pipeline"})
    )

    assert registry.get_version("fashion-cnn", "1").format is ArtifactFormat.ONNX


def test_empty_metadata_is_refused_and_this_is_the_line_m3_may_have_to_change():
    """Pinning today's behaviour so the change is deliberate when it comes.

    If the tenant reports `{}` for a normally-registered MLflow model, this
    assertion is what has to be inverted -- and inverting it makes every
    version in the registry deployable on no evidence at all, which is why it
    waits on the observed payload rather than on an argument.
    """
    registry, _ = _registry_for(_settings(), _version_with({}))

    with pytest.raises(UnsupportedFlavor):
        registry.get_version("fashion-cnn", "1")


# The evidence slot. Deliberately read from `.dev/`, which `.gitignore` has
# ignored since M2: a real version payload is tenant data and this repo is
# public, so the one file that would settle this must live somewhere it cannot
# be committed from. Drop the `metadata` object printed by
# `scripts/register_model.py` into it and this stops skipping.
_OBSERVED = (
    Path(__file__).resolve().parents[2] / ".dev" / "m3" / "observed_version_metadata.json"
)


@pytest.mark.skipif(
    not _OBSERVED.is_file(),
    reason=f"no observed registry metadata yet -- write one to {_OBSERVED} "
    "(the `metadata` object from scripts/register_model.py's findings block)",
)
def test_the_observed_metadata_of_a_real_mlflow_version_yields_onnx():
    """The assertion M3 exists to make, against the real payload.

    Runs the production function on bytes that came off the tenant. A failure
    here is not a flaky test -- it means a model registered the normal way is
    undeployable, and the message names which branch of
    `_format_and_packaging` has to move.
    """
    observed = json.loads(_OBSERVED.read_text())
    metadata = observed.get("metadata", observed)
    assert isinstance(metadata, dict), f"{_OBSERVED} should hold the metadata object"

    fmt, packaging = _format_and_packaging(metadata)

    assert fmt is ArtifactFormat.ONNX, (
        f"a real MLflow version reported metadata keys {sorted(metadata)} with "
        f"model_repo_type={metadata.get('model_repo_type')!r}, which this adapter "
        f"reads as {fmt.value}. Every model in the registry is undeployable until "
        "the matching branch in _format_and_packaging covers this shape."
    )
    assert packaging is Packaging.MLFLOW_TAR_GZ


# --- auth and transport failure handling --------------------------------------


def test_a_rejected_token_is_minted_once_more_and_then_given_up_on():
    """A 401 must trigger exactly one retry after `invalidate()` -- not zero
    (a token that just expired would fail every call forever) and not an
    unbounded loop (a registry that always 401s, e.g. a revoked workload,
    must fail fast with `RegistryAuthError` instead of hammering the auth
    chain). Asserting the invalidate and request counts is what tells the two
    apart from "it happened to raise the right exception"."""
    call_count = 0

    def models_route(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(401, json={"message": "unauthorized"})

    handler = _route({("GET", f"{PREFIX}/models"): models_route})
    registry, token_provider = _registry_for(_settings(), handler)

    with pytest.raises(RegistryAuthError):
        registry.list_models()

    assert call_count == 2
    assert token_provider.invalidate_count == 1


def test_a_timeout_is_a_registry_unavailable_not_an_httpx_error():
    """Nothing above `CAIRegistryClient._send` may see an `httpx` exception
    type -- that is the entire point of the adapter boundary this module's
    own docstring describes. A raw `httpx.TimeoutException` from the transport
    must come out the other side as `RegistryUnavailable`."""

    def models_route(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("simulated read timeout", request=request)

    handler = _route({("GET", f"{PREFIX}/models"): models_route})
    registry, _ = _registry_for(_settings(), handler)

    with pytest.raises(RegistryUnavailable):
        registry.list_models()


def test_the_workload_token_never_reaches_the_logs(caplog):
    """The bearer token is a credential capable of calling the registry as
    this workload. `_send`'s own docstring claims the request path is safe to
    log but the token never is -- this is the test that actually holds that
    claim to account, across both a listing warning path and the artifact
    probe's info log."""
    secret_token = "sentinel-super-secret-jwt-value-should-never-be-logged"
    duplicate_models = {
        "models": [_model_entry("model-1", "fashion-cnn"), _model_entry("model-2", "fashion-cnn")]
    }
    handler = _route({("GET", f"{PREFIX}/models"): httpx.Response(200, json=duplicate_models)})
    registry, _ = _registry_for(
        _settings(), handler, token_provider=_FakeTokenProvider([secret_token])
    )

    with caplog.at_level(logging.DEBUG):
        registry.list_models()  # triggers the duplicate-name warning path

        mv = RegistryModelVersion(
            name="fashion-cnn",
            version="1",
            model_id="model-1",
            version_uuid="1-1704067200",
            artifact_uri="s3a://bucket/fashion-cnn/1",
        )
        artifact_handler = _route(
            {
                ("GET", f"{PREFIX}/models/model-1/versions/1/artifact"): httpx.Response(
                    200, content=b"bytes", headers={"content-type": "application/octet-stream"}
                )
            }
        )
        artifact_client, _ = _client_for(
            _settings(), artifact_handler, token_provider=_FakeTokenProvider([secret_token])
        )
        artifact_registry = CAIModelRegistry(_settings(), artifact_client)
        with artifact_registry.open_artifact(mv) as stream:
            list(stream.chunks())

    assert secret_token not in caplog.text


def test_the_bearer_token_is_not_forwarded_when_the_artifact_redirects():
    """The workload JWT authenticates calls to the registry's own API; a
    redirect off the `/artifact` route points at an object store that must
    never see it. The first request (to the registry) must carry
    Authorization; the second (to the redirect target) must not."""
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        if request.url.path == f"{PREFIX}/models/model-1/versions/3/artifact":
            return httpx.Response(302, headers={"location": "https://objectstore.example.invalid/bucket/key"})
        if request.url.host == "objectstore.example.invalid":
            return httpx.Response(200, content=b"redirected-artifact-bytes", headers={"content-type": "application/octet-stream"})
        raise AssertionError(f"unexpected request: {request.url}")

    client, _ = _client_for(_settings(), handler)
    registry = CAIModelRegistry(_settings(), client)
    mv = RegistryModelVersion(
        name="fashion-cnn",
        version="3",
        model_id="model-1",
        version_uuid="3-1704067200",
        artifact_uri="s3a://bucket/fashion-cnn/3",
    )

    with registry.open_artifact(mv) as stream:
        data = b"".join(stream.chunks())

    assert data == b"redirected-artifact-bytes"
    assert len(requests_seen) == 2
    assert "authorization" in requests_seen[0].headers
    assert "authorization" not in requests_seen[1].headers


def test_an_artifact_served_as_multipart_yields_only_the_part_body():
    """The registry's spec declares `produces: multipart/form-data` on the
    artifact route, which is likely a mislabel of a route that otherwise
    serves raw bytes -- so the adapter branches on content-type and extracts
    just the first part's body. The boundary markers and part headers must
    not leak into the bytes the artifact service goes on to hash."""
    boundary = "LighthouseTestBoundary123"
    part_body = b"THE-ONLY-BYTES-THAT-SHOULD-SURVIVE-PARSING"
    raw = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="model.tar.gz"\r\n'
        "Content-Type: application/octet-stream\r\n"
        "\r\n"
    ).encode("ascii") + part_body + f"\r\n--{boundary}--\r\n".encode("ascii")

    handler = _route(
        {
            ("GET", f"{PREFIX}/models/model-1/versions/5/artifact"): httpx.Response(
                200, content=raw, headers={"content-type": f"multipart/form-data; boundary={boundary}"}
            )
        }
    )
    client, _ = _client_for(_settings(), handler)
    registry = CAIModelRegistry(_settings(), client)
    mv = RegistryModelVersion(
        name="fashion-cnn",
        version="5",
        model_id="model-1",
        version_uuid="5-1704067200",
        artifact_uri="s3a://bucket/fashion-cnn/5",
    )

    with registry.open_artifact(mv) as stream:
        data = b"".join(stream.chunks())

    assert data == part_body


def test_an_octet_stream_artifact_is_streamed_straight_through():
    """The common case: a 200 with `content-type: application/octet-stream`
    is streamed with no interpretation at all. This is the baseline the
    multipart and redirect branches are deliberate exceptions to."""
    raw_bytes = b"straight-through-artifact-bytes" * 10
    handler = _route(
        {
            ("GET", f"{PREFIX}/models/model-1/versions/2/artifact"): httpx.Response(
                200, content=raw_bytes, headers={"content-type": "application/octet-stream"}
            )
        }
    )
    client, _ = _client_for(_settings(), handler)
    registry = CAIModelRegistry(_settings(), client)
    mv = RegistryModelVersion(
        name="fashion-cnn",
        version="2",
        model_id="model-1",
        version_uuid="2-1704067200",
        artifact_uri="s3a://bucket/fashion-cnn/2",
    )

    with registry.open_artifact(mv) as stream:
        data = b"".join(stream.chunks())

    assert data == raw_bytes


def test_the_one_artifact_probe_log_names_the_body_encoding(caplog):
    """The artifact probe logs once per process, and that single line is what
    anybody will read after the first real registration. It has to carry
    `content-encoding`, because the failure it distinguishes is otherwise
    close to undiagnosable.

    httpx decodes a `Content-Encoding: gzip` body transparently. If the
    registry serves the tarball that way, the bytes reaching the cache are a
    *bare* tar while `Packaging` still says MLFLOW_TAR_GZ -- and nothing
    upstream notices, because the control plane hashes the decoded bytes and
    the device verifies those same bytes, so every checksum in the system
    agrees. The only thing that fails is `tarfile.open(..., "r:gz")`: once
    server-side in `_read_entrypoint`, and again on the Jetson in `_activate`,
    after a full download over a home uplink.

    Asserted on the emitted record rather than on the headers dict, since the
    whole value of the line is that it is *in the log* for an operator to
    paste back.
    """
    inner_tar = b"what-the-device-needs-to-see-as-a-gzip-tar"
    handler = _route(
        {
            ("GET", f"{PREFIX}/models/model-1/versions/2/artifact"): httpx.Response(
                200,
                content=gzip.compress(inner_tar),
                headers={
                    "content-type": "application/x-tar",
                    "content-encoding": "gzip",
                },
            )
        }
    )
    client, _ = _client_for(_settings(), handler)
    registry = CAIModelRegistry(_settings(), client)
    mv = RegistryModelVersion(
        name="fashion-cnn",
        version="2",
        model_id="model-1",
        version_uuid="2-1704067200",
        artifact_uri="s3a://bucket/fashion-cnn/2",
    )

    with caplog.at_level(logging.INFO):
        with registry.open_artifact(mv) as stream:
            data = b"".join(stream.chunks())

    probe_lines = [r.getMessage() for r in caplog.records if "artifact response" in r.getMessage()]
    assert probe_lines, "the one-shot artifact probe line was not logged at INFO"
    assert "content-encoding='gzip'" in probe_lines[0]

    # The defect itself, asserted rather than described: what came out of the
    # stream is the *decoded* tar, not the gzip the registry sent. Writing this
    # to the cache leaves a file that `tarfile.open(..., "r:gz")` cannot read
    # while every digest still agrees -- which is why the header above has to
    # be in the log.
    assert data == inner_tar
    assert data[:2] != b"\x1f\x8b"


# --- auth-chain subprocess boundary --------------------------------------------

def test_a_failed_cdp_call_surfaces_its_stderr_and_never_its_stdout():
    """`CdpCliTokenProvider._fetch` deliberately trusts only stderr on a
    non-zero exit: some CLI versions echo a freshly minted JWT to stdout even
    on a failure path, and that string must never end up inside a
    `RegistryAuthError` message that could be logged or displayed."""
    stdout_token_lookalike = "eyJFAKE.PAYLOAD.SHOULD-NEVER-APPEAR-IN-THE-RAISED-MESSAGE"
    stderr_reason = "permission denied: credential is not entitled to workload DE"

    def fake_runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args, returncode=1, stdout=f'{{"token": "{stdout_token_lookalike}"}}', stderr=stderr_reason
        )

    provider = CdpCliTokenProvider(runner=fake_runner)

    with pytest.raises(RegistryAuthError) as excinfo:
        provider.token()

    message = str(excinfo.value)
    assert stderr_reason in message
    assert stdout_token_lookalike not in message


def test_a_tenant_wide_registry_listing_is_never_guessed_from():
    """`cdp ml list-model-registries` is tenant-wide -- it answers with every
    registry the credential can see, which in practice is mostly other
    people's. `discover_domain` must require an exact `environmentName` match
    and raise rather than ever fall back to "pick the first one"."""

    def fake_runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        payload = {
            "modelRegistries": [
                {"environmentName": "someone-elses-env", "domain": "other.example.invalid", "status": "created:finished"}
            ]
        }
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=json.dumps(payload), stderr="")

    with pytest.raises(ConfigError):
        discover_domain("test-env", fake_runner)
