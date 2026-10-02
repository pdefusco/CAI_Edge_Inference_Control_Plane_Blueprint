"""Registry catalog routes (spec SS12)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from lighthouse_contracts import ModelVersionView, ModelView

from ..registry import RegistryError
from .auth import require_operator
from .deps import AppContext, ctx
from .errors import registry_http_error

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelView])
def list_models(
    versions: bool = True,
    _operator=Depends(require_operator),
    context: AppContext = Depends(ctx),
) -> list[ModelView]:
    """Models available to deploy.

    Operator-only: the list of every model in the registry is not something a
    single edge device should be able to enumerate.
    """
    try:
        return context.catalog.list_models(include_versions=versions)
    except RegistryError as exc:
        raise registry_http_error(exc) from exc


@router.get("/{model_name}/versions", response_model=list[ModelVersionView])
def list_model_versions(
    model_name: str,
    _operator=Depends(require_operator),
    context: AppContext = Depends(ctx),
) -> list[ModelVersionView]:
    """Versions of one model, newest first, each annotated with whether it can be
    deployed and -- when it cannot -- why."""
    try:
        return context.catalog.list_versions(model_name)
    except RegistryError as exc:
        raise registry_http_error(exc) from exc
