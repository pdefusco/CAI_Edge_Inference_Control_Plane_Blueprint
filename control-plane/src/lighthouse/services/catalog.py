"""Read-only view of the model registry for operators (spec SS12).

Thin on purpose. The interesting work -- resolving names, interpreting flavors,
deciding what "ready" means -- belongs to the registry adapter; this layer only
turns registry objects into wire views and annotates *why* a version cannot be
deployed. Surfacing that reason in the picker is what stops an operator from
assigning a version the device would reject after a download.
"""

from __future__ import annotations

import logging

from lighthouse_contracts import ArtifactFormat, ModelVersionView, ModelView

from ..registry import ModelNotFound, ModelRegistry, RegistryError, RegistryModelVersion

log = logging.getLogger(__name__)


class ModelCatalog:
    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def list_models(self, include_versions: bool = True) -> list[ModelView]:
        """Every registered model.

        A model whose versions cannot be listed is still returned, with an empty
        version list: a single bad model should not blank the whole catalog page.
        """
        out: list[ModelView] = []
        for name in self._registry.list_models():
            versions: list[ModelVersionView] = []
            model_id: str | None = None
            if include_versions:
                try:
                    resolved = self._registry.list_versions(name)
                except RegistryError as exc:
                    log.warning("could not list versions of %s: %s", name, exc)
                    resolved = []
                if resolved:
                    model_id = resolved[0].model_id
                versions = [_version_view(mv) for mv in reversed(resolved)]
            out.append(ModelView(name=name, model_id=model_id, versions=versions))
        return out

    def list_versions(self, model_name: str) -> list[ModelVersionView]:
        """Versions of one model, newest first.

        Raises ModelNotFound, which the API maps to 404 -- an unknown model name
        must not look like a model with no versions.
        """
        resolved = self._registry.list_versions(model_name)
        if not resolved:
            raise ModelNotFound(model_name)
        return [_version_view(mv) for mv in reversed(resolved)]

    def ping(self) -> bool:
        try:
            return self._registry.ping()
        except RegistryError:
            return False


def _version_view(mv: RegistryModelVersion) -> ModelVersionView:
    deployable = True
    reason: str | None = None
    if mv.status != "READY":
        deployable = False
        reason = f"registry status is {mv.status}"
    elif mv.format is not ArtifactFormat.ONNX:
        deployable = False
        reason = f"format {mv.format.value} is not runnable at the edge"
    return ModelVersionView(
        name=mv.name,
        version=mv.version,
        status=mv.status,
        format=mv.format,
        created_at=mv.created_at,
        registry_artifact_uri=mv.artifact_uri,
        deployable=deployable,
        reason=reason,
    )
