"""Device routes (spec SS12).

Note which dependency guards which route. `desired-state` and `heartbeat` take
`require_device`; everything else takes `require_operator`. That split is the
architecture in miniature -- the operator writes desired state and may never write
actual state, the device writes actual state and may never write desired state.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Response, status
from lighthouse_contracts import (
    AuditEventView,
    DeploymentRequest,
    DesiredStateResponse,
    DeviceRegistrationRequest,
    DeviceTokenIssued,
    DeviceView,
    HeartbeatRequest,
    HeartbeatResponse,
)
from pydantic import BaseModel, ConfigDict

from ..registry import RegistryError
from ..services import DeviceNotFound, DevicePrincipal, NothingDeployed
from .auth import require_device, require_operator
from .deps import AppContext, ctx
from .errors import registry_http_error

log = logging.getLogger(__name__)

router = APIRouter(prefix="/devices", tags=["devices"])


class ReasonRequest(BaseModel):
    """Optional justification recorded in the audit trail."""

    model_config = ConfigDict(extra="forbid")

    reason: str | None = None


class DeviceRegistered(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: DeviceView
    credentials: DeviceTokenIssued


class DeploymentAccepted(BaseModel):
    """SS13: the request is accepted and returns immediately; it does not wait for
    the device. `artifact_ready` false means the control plane is still fetching
    the bytes, which is a wait state for the agent, not a failure."""

    model_config = ConfigDict(extra="forbid")

    device_id: str
    generation: int
    desired_state: str
    model_name: str | None = None
    model_version: str | None = None
    artifact_ready: bool = True


# -- operator surface ----------------------------------------------------


@router.get("", response_model=list[DeviceView])
def list_devices(
    _operator=Depends(require_operator),
    context: AppContext = Depends(ctx),
) -> list[DeviceView]:
    return context.devices.list_views()


@router.post("", response_model=DeviceRegistered, status_code=status.HTTP_201_CREATED)
def register_device(
    request: DeviceRegistrationRequest,
    _operator=Depends(require_operator),
    context: AppContext = Depends(ctx),
) -> DeviceRegistered:
    """Enrol a device and return its token **once**.

    Only sha256(secret) is stored, so this response is the single opportunity to
    capture the credential. A lost token is reissued, never recovered.
    """
    from ..services import DeviceAlreadyExists

    try:
        device, credentials = context.devices.register(request)
    except DeviceAlreadyExists as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT, detail=f"device already registered: {exc}"
        ) from exc
    return DeviceRegistered(
        device=context.devices.get_view(device.device_id),
        credentials=credentials,
    )


@router.get("/{device_id}", response_model=DeviceView)
def get_device(
    device_id: str,
    _operator=Depends(require_operator),
    context: AppContext = Depends(ctx),
) -> DeviceView:
    try:
        return context.devices.get_view(device_id)
    except DeviceNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such device") from exc


@router.delete("/{device_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_device(
    device_id: str,
    _operator=Depends(require_operator),
    context: AppContext = Depends(ctx),
) -> Response:
    try:
        context.devices.delete(device_id)
    except DeviceNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such device") from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{device_id}/tokens",
    response_model=DeviceTokenIssued,
    status_code=status.HTTP_201_CREATED,
)
def issue_device_token(
    device_id: str,
    body: ReasonRequest | None = None,
    _operator=Depends(require_operator),
    context: AppContext = Depends(ctx),
) -> DeviceTokenIssued:
    """Mint an additional token so rotation needs no downtime: issue, install on
    the device, then revoke the old one."""
    try:
        return context.devices.issue_token(device_id, label=body.reason if body else None)
    except DeviceNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such device") from exc


@router.delete("/{device_id}/tokens/{token_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_device_token(
    device_id: str,
    token_id: str,
    _operator=Depends(require_operator),
    context: AppContext = Depends(ctx),
) -> Response:
    if not context.devices.revoke_token(token_id):
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail="no such active token"
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.put("/{device_id}/deployment", response_model=DeploymentAccepted)
def put_deployment(
    device_id: str,
    request: DeploymentRequest,
    _operator=Depends(require_operator),
    context: AppContext = Depends(ctx),
) -> DeploymentAccepted:
    """Assign a model version (spec SS13).

    Naming an earlier version is how rollback works; there is no separate
    endpoint. The audit event is classified server-side from deployment history,
    so an operator cannot mislabel one.
    """
    try:
        row = context.deployments.deploy(device_id, request)
    except DeviceNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such device") from exc
    except RegistryError as exc:
        raise registry_http_error(exc) from exc

    ready = True
    if row.cache_key is not None:
        ready = context.artifacts.get_ready(row.cache_key) is not None
    return DeploymentAccepted(
        device_id=device_id,
        generation=row.generation,
        desired_state=row.desired_state.value,
        model_name=row.model_name,
        model_version=row.model_version,
        artifact_ready=ready,
    )


@router.post("/{device_id}/stop", response_model=DeploymentAccepted)
def stop_device(
    device_id: str,
    body: ReasonRequest | None = None,
    _operator=Depends(require_operator),
    context: AppContext = Depends(ctx),
) -> DeploymentAccepted:
    """Stop inference, keep artifacts on disk, allow restart (spec SS5)."""
    return _transition(context, device_id, "stop", body.reason if body else None)


@router.post("/{device_id}/revoke", response_model=DeploymentAccepted)
def revoke_device(
    device_id: str,
    body: ReasonRequest | None = None,
    _operator=Depends(require_operator),
    context: AppContext = Depends(ctx),
) -> DeploymentAccepted:
    """Withdraw authorization: stop inference and delete artifacts (spec SS5)."""
    return _transition(context, device_id, "revoke", body.reason if body else None)


def _transition(
    context: AppContext, device_id: str, action: str, reason: str | None
) -> DeploymentAccepted:
    fn = context.deployments.stop if action == "stop" else context.deployments.revoke
    try:
        row = fn(device_id, reason)
    except DeviceNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such device") from exc
    except NothingDeployed as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return DeploymentAccepted(
        device_id=device_id,
        generation=row.generation,
        desired_state=row.desired_state.value,
        model_name=row.model_name,
        model_version=row.model_version,
    )


@router.get("/{device_id}/events", response_model=list[AuditEventView])
def device_events(
    device_id: str,
    limit: int = 100,
    _operator=Depends(require_operator),
    context: AppContext = Depends(ctx),
) -> list[AuditEventView]:
    if context.store.get_device(device_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such device")
    return context.audit.list_events(device_id=device_id, limit=min(limit, 1000))


# -- device surface ------------------------------------------------------


@router.get("/{device_id}/desired-state", response_model=DesiredStateResponse)
def get_desired_state(
    device_id: str,
    principal: DevicePrincipal = Depends(require_device),
    context: AppContext = Depends(ctx),
) -> DesiredStateResponse:
    """What the control plane wants this device to be doing (spec SS8).

    A cheap read: it never triggers a fresh materialization, only a cooldown-gated
    retry of one that previously failed.
    """
    try:
        return context.deployments.desired_state_for(device_id)
    except DeviceNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such device") from exc


@router.post("/{device_id}/heartbeat", response_model=HeartbeatResponse)
def post_heartbeat(
    device_id: str,
    heartbeat: HeartbeatRequest,
    principal: DevicePrincipal = Depends(require_device),
    context: AppContext = Depends(ctx),
) -> HeartbeatResponse:
    """Report actual state (spec SS7).

    `heartbeat.device_id` is checked against the authenticated identity and
    rejected on mismatch. It never *establishes* identity -- if it did, one
    compromised device could rewrite the whole fleet's actual state and the
    governance view would become fiction.
    """
    if heartbeat.device_id != principal.device_id:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="heartbeat device_id does not match the authenticated device",
        )
    try:
        return context.deployments.record_heartbeat(device_id, heartbeat)
    except DeviceNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such device") from exc
