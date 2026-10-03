"""The operator's view of the registry, and why a version cannot be deployed.

`ModelCatalog` is 83 lines and had no test file. It is also the only thing that
decides what an operator *sees* in the deploy picker, which makes two of its
behaviors worth pinning precisely:

  * **Newest-first ordering.** `ModelRegistry.list_versions` contracts oldest-first
    (`registry/base.py:176-181`) and the catalog reverses it. Nothing anywhere else
    sorts versions, so if either side of that pair changed, the picker would offer
    the oldest version at the top and an operator would deploy a stale model while
    reading a list that looked right.
  * **`deployable` and its reason.** This is the deploy gate's only
    *pre-deployment* warning. Getting it wrong means an operator assigns a version
    the device refuses after a full download -- the failure the field exists to
    prevent. M2 made this reachable for real: a Hugging Face or NGC version now
    arrives as `ArtifactFormat.UNKNOWN`, and the only thing that turns that into
    visible, actionable text is `_version_view`.

The registry is a hand-built stub rather than `FakeModelRegistry`, because these
tests need to inject version *states* the fake never produces -- `UPLOAD_FAILED`,
`UNKNOWN` format, a listing that raises for one model and succeeds for another.
It implements only the four methods `ModelCatalog` calls; nothing is patched.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from lighthouse_contracts import ArtifactFormat

from lighthouse.registry import ModelNotFound, RegistryModelVersion, RegistryUnavailable
from lighthouse.services.catalog import ModelCatalog


def _version(
    name: str = "fashion-cnn",
    version: str = "1",
    *,
    status: str = "READY",
    format: ArtifactFormat = ArtifactFormat.ONNX,
) -> RegistryModelVersion:
    return RegistryModelVersion(
        name=name,
        version=version,
        model_id=f"m-{name}",
        version_uuid=f"{version}-stable",
        artifact_uri=f"s3://bucket/{name}/{version}",
        status=status,
        format=format,
        created_at=datetime(2026, 1, int(version) if version.isdigit() else 1, tzinfo=timezone.utc),
    )


class StubRegistry:
    """Only what `ModelCatalog` calls. `versions` maps a name to either a list
    (returned as-is, oldest first) or an exception instance (raised)."""

    name = "stub"

    def __init__(self, versions: dict[str, object] | None = None, *, reachable: bool = True) -> None:
        self._versions = versions or {}
        self._reachable = reachable
        self.list_versions_calls: list[str] = []

    def list_models(self) -> list[str]:
        return sorted(self._versions)

    def list_versions(self, model_name: str) -> list[RegistryModelVersion]:
        self.list_versions_calls.append(model_name)
        result = self._versions.get(model_name)
        if isinstance(result, Exception):
            raise result
        if result is None:
            raise ModelNotFound(model_name)
        return list(result)  # type: ignore[arg-type]

    def ping(self) -> bool:
        if not self._reachable:
            raise RegistryUnavailable("stub is down")
        return True


# --------------------------------------------------------------------------
# Ordering -- the contract that spans two files
# --------------------------------------------------------------------------


def test_versions_are_presented_newest_first():
    """The registry contracts oldest-first and the catalog reverses it. If that
    reversal were dropped, the picker's top entry would be the oldest version and
    nothing else in the system would notice."""
    registry = StubRegistry({"fashion-cnn": [_version(version="1"), _version(version="2"), _version(version="3")]})

    views = ModelCatalog(registry).list_versions("fashion-cnn")

    assert [v.version for v in views] == ["3", "2", "1"]


def test_the_models_listing_orders_versions_the_same_way_as_the_detail_listing():
    """Two code paths build the same view, so they are the two places this could
    drift apart."""
    registry = StubRegistry({"fashion-cnn": [_version(version="1"), _version(version="2")]})
    catalog = ModelCatalog(registry)

    from_listing = catalog.list_models()[0].versions
    from_detail = catalog.list_versions("fashion-cnn")

    assert [v.version for v in from_listing] == [v.version for v in from_detail] == ["2", "1"]


# --------------------------------------------------------------------------
# Why a version cannot be deployed
# --------------------------------------------------------------------------


def test_a_ready_onnx_version_is_deployable_with_no_reason_given():
    """The negative control for every rule below: if this ever fails, the gate has
    been tightened into refusing valid models."""
    registry = StubRegistry({"fashion-cnn": [_version()]})

    view = ModelCatalog(registry).list_versions("fashion-cnn")[0]

    assert view.deployable is True
    assert view.reason is None


@pytest.mark.parametrize("status", ["REGISTERING", "UPLOADING", "UPLOAD_FAILED", "DELETE_FAILED", "UNKNOWN"])
def test_a_version_the_registry_has_not_finished_is_not_deployable(status):
    """Every non-READY status blocks deployment and names itself in the reason.
    Parametrized rather than spot-checked because a status that silently read as
    deployable would put unbuilt bytes into desired state."""
    registry = StubRegistry({"fashion-cnn": [_version(status=status)]})

    view = ModelCatalog(registry).list_versions("fashion-cnn")[0]

    assert view.deployable is False
    assert status in (view.reason or "")


def test_the_raw_registry_status_reaches_the_operator_unedited():
    """The adapter deliberately does not editorialize a broken status out of the
    listing, so the dashboard must show the registry's own word for it -- that
    string is what an operator searches the registry's docs for."""
    registry = StubRegistry({"fashion-cnn": [_version(status="UPLOAD_FAILED")]})

    view = ModelCatalog(registry).list_versions("fashion-cnn")[0]

    assert view.status == "UPLOAD_FAILED"
    assert view.reason == "registry status is UPLOAD_FAILED"


def test_a_version_with_no_edge_runnable_format_is_refused_before_deployment():
    """M2's Hugging Face and NGC path. The CAI adapter maps those to UNKNOWN, and
    this is the only place that becomes text an operator can act on instead of a
    download that fails on the device."""
    registry = StubRegistry({"hf-model": [_version(name="hf-model", format=ArtifactFormat.UNKNOWN)]})

    view = ModelCatalog(registry).list_versions("hf-model")[0]

    assert view.deployable is False
    assert "not runnable at the edge" in (view.reason or "")


def test_an_unready_version_reports_its_status_rather_than_its_format():
    """Both rules fire for a version that is neither READY nor ONNX. Status wins,
    because it is the one that may yet resolve on its own -- telling an operator to
    re-export a model that is merely still uploading sends them to do pointless
    work."""
    registry = StubRegistry(
        {"fashion-cnn": [_version(status="UPLOADING", format=ArtifactFormat.UNKNOWN)]}
    )

    view = ModelCatalog(registry).list_versions("fashion-cnn")[0]

    assert view.reason == "registry status is UPLOADING"


# --------------------------------------------------------------------------
# One bad model must not blank the catalog
# --------------------------------------------------------------------------


def test_a_model_whose_versions_cannot_be_listed_is_still_listed():
    """A registry error on one model returns it with no versions rather than
    failing the page. The whole catalog going blank because of one broken entry
    would be a much worse outage than one model looking empty."""
    registry = StubRegistry(
        {
            "broken": RegistryUnavailable("upstream exploded"),
            "fine": [_version(name="fine")],
        }
    )

    views = ModelCatalog(registry).list_models()

    assert [v.name for v in views] == ["broken", "fine"]
    assert views[0].versions == []
    assert len(views[1].versions) == 1


def test_a_model_with_no_versions_carries_no_model_id():
    """`model_id` is read off the first resolved version, so there is nothing to
    report when there are none. It must be None rather than a guess -- it keys the
    artifact cache."""
    registry = StubRegistry({"empty": []})

    view = ModelCatalog(registry).list_models()[0]

    assert view.model_id is None
    assert view.versions == []


def test_the_model_id_comes_from_the_registry_not_from_the_name():
    registry = StubRegistry({"fashion-cnn": [_version()]})

    view = ModelCatalog(registry).list_models()[0]

    assert view.model_id == "m-fashion-cnn"


def test_an_unknown_model_is_not_the_same_as_a_model_with_no_versions():
    """`list_versions` raises ModelNotFound, which the API maps to 404. Returning
    an empty list would let a mistyped name look like a real model that simply has
    nothing in it, and an operator would wait for a version that will never come."""
    registry = StubRegistry({"fashion-cnn": [_version()]})

    with pytest.raises(ModelNotFound):
        ModelCatalog(registry).list_versions("typo-cnn")


def test_a_model_that_exists_but_has_no_versions_also_reads_as_not_found():
    """Deliberate: the deploy picker has nothing to offer either way, and the 404
    says so without inventing a distinction the operator cannot act on."""
    registry = StubRegistry({"empty": []})

    with pytest.raises(ModelNotFound):
        ModelCatalog(registry).list_versions("empty")


# --------------------------------------------------------------------------
# Avoiding work, and surviving an unreachable registry
# --------------------------------------------------------------------------


def test_listing_without_versions_makes_no_per_model_version_call():
    """`list_models` is 1 + N round trips with versions included. The flag is the
    only way to get the cheap listing, so it must actually skip the calls."""
    registry = StubRegistry({"a": [_version(name="a")], "b": [_version(name="b")]})

    views = ModelCatalog(registry).list_models(include_versions=False)

    assert registry.list_versions_calls == []
    assert [v.name for v in views] == ["a", "b"]
    assert all(v.versions == [] for v in views)


def test_an_unreachable_registry_is_a_false_ping_not_an_exception():
    """`/health` calls this on every probe. A raised error would turn a degraded
    registry into a 500 on the health endpoint, which reads as the control plane
    being down rather than its upstream."""
    registry = StubRegistry(reachable=False)

    assert ModelCatalog(registry).ping() is False


def test_a_reachable_registry_pings_true():
    assert ModelCatalog(StubRegistry()).ping() is True
