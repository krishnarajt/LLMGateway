"""
User routes — accessible to any authenticated user.
Covers:
  - Gateway API key management (create, list, revoke)
  - Permission requests (create, list own)
  - List available models (public info)
"""

import hashlib
import secrets

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.db.database import get_db
from app.db.models import (
    User,
    GatewayApiKey,
    ApiKeyModelPermission,
    LLMModel,
    PermissionRequest,
    PermissionRequestStatus,
    Provider,
)
from app.common.schemas import (
    GatewayApiKeyCreate,
    GatewayApiKeyOut,
    GatewayApiKeyCreated,
    ApiKeyPermissionOut,
    PermissionRequestCreate,
    PermissionRequestOut,
    LLMModelOut,
)
from app.utils.dependencies import require_user
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/user", tags=["User"])


# ═══════════════════════════════════════════════════════════════════════════
# Gateway API Key Management
# ═══════════════════════════════════════════════════════════════════════════


def _generate_gateway_key() -> tuple[str, str, str]:
    """
    Generate a new gateway API key.
    Returns (raw_key, key_hash, key_prefix).
    The raw key is shown once to the user; we store only the hash.
    """
    raw_key = "gw-" + secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    key_prefix = raw_key[:10] + "***"
    return raw_key, key_hash, key_prefix


@router.post("/api-keys", response_model=GatewayApiKeyCreated)
def create_api_key(
    request: GatewayApiKeyCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_user),
):
    """
    Create a new gateway API key.
    The raw key is returned ONCE in this response — store it securely.
    """
    raw_key, key_hash, key_prefix = _generate_gateway_key()

    api_key = GatewayApiKey(
        user_id=current_user.id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        label=request.label,
    )
    db.add(api_key)
    db.commit()
    db.refresh(api_key)

    return GatewayApiKeyCreated(
        id=api_key.id,
        key=raw_key,
        key_prefix=key_prefix,
        label=api_key.label,
    )


@router.get("/api-keys", response_model=List[GatewayApiKeyOut])
def list_api_keys(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_user),
):
    """List all gateway API keys for the current user."""
    keys = (
        db.query(GatewayApiKey)
        .filter(GatewayApiKey.user_id == current_user.id)
        .order_by(GatewayApiKey.id)
        .all()
    )
    result = []
    for k in keys:
        perms = []
        for p in k.permissions:
            perm_out = ApiKeyPermissionOut(
                id=p.id,
                model_id=p.model_id,
                model_display_name=p.model.display_name if p.model else None,
                provider_name=p.model.provider.name
                if p.model and p.model.provider
                else None,
                max_input_tokens=p.max_input_tokens,
                max_output_tokens=p.max_output_tokens,
                is_active=p.is_active,
            )
            perms.append(perm_out)

        result.append(
            GatewayApiKeyOut(
                id=k.id,
                key_prefix=k.key_prefix,
                label=k.label,
                is_active=k.is_active,
                created_at=k.created_at,
                last_used_at=k.last_used_at,
                permissions=perms,
            )
        )
    return result


@router.delete("/api-keys/{key_id}")
def revoke_api_key(
    key_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_user),
):
    """Revoke (delete) a gateway API key."""
    api_key = (
        db.query(GatewayApiKey)
        .filter(GatewayApiKey.id == key_id, GatewayApiKey.user_id == current_user.id)
        .first()
    )
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")
    db.delete(api_key)
    db.commit()
    return {"success": True, "message": "API key revoked"}


@router.patch("/api-keys/{key_id}/toggle")
def toggle_api_key(
    key_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_user),
):
    """Toggle active/inactive status of a gateway API key."""
    api_key = (
        db.query(GatewayApiKey)
        .filter(GatewayApiKey.id == key_id, GatewayApiKey.user_id == current_user.id)
        .first()
    )
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")
    api_key.is_active = not api_key.is_active
    db.commit()
    return {"success": True, "is_active": api_key.is_active}


# ═══════════════════════════════════════════════════════════════════════════
# Permission Requests
# ═══════════════════════════════════════════════════════════════════════════


@router.post("/permission-requests", response_model=PermissionRequestOut)
def create_permission_request(
    request: PermissionRequestCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_user),
):
    """Request access to a model for one of your API keys."""
    # Verify the API key belongs to the current user
    api_key = (
        db.query(GatewayApiKey)
        .filter(
            GatewayApiKey.id == request.api_key_id,
            GatewayApiKey.user_id == current_user.id,
        )
        .first()
    )
    if not api_key:
        raise HTTPException(
            status_code=404, detail="API key not found or does not belong to you"
        )

    # Verify the model exists
    model = db.query(LLMModel).filter(LLMModel.id == request.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Check if permission already exists
    existing_perm = (
        db.query(ApiKeyModelPermission)
        .filter(
            ApiKeyModelPermission.api_key_id == request.api_key_id,
            ApiKeyModelPermission.model_id == request.model_id,
        )
        .first()
    )
    if existing_perm:
        raise HTTPException(
            status_code=400, detail="Permission already granted for this model"
        )

    # Check for pending request
    pending = (
        db.query(PermissionRequest)
        .filter(
            PermissionRequest.api_key_id == request.api_key_id,
            PermissionRequest.model_id == request.model_id,
            PermissionRequest.status == PermissionRequestStatus.pending,
        )
        .first()
    )
    if pending:
        raise HTTPException(
            status_code=400, detail="A pending request already exists for this model"
        )

    perm_req = PermissionRequest(
        user_id=current_user.id,
        api_key_id=request.api_key_id,
        model_id=request.model_id,
        request_message=request.request_message,
    )
    db.add(perm_req)
    db.commit()
    db.refresh(perm_req)

    return PermissionRequestOut(
        id=perm_req.id,
        user_id=perm_req.user_id,
        username=current_user.username,
        api_key_id=perm_req.api_key_id,
        api_key_label=api_key.label,
        model_id=perm_req.model_id,
        model_display_name=model.display_name,
        provider_name=model.provider.name if model.provider else None,
        status=perm_req.status.value,
        request_message=perm_req.request_message,
        admin_message=perm_req.admin_message,
        reviewed_by=perm_req.reviewed_by,
        created_at=perm_req.created_at,
    )


@router.get("/permission-requests", response_model=List[PermissionRequestOut])
def list_my_permission_requests(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_user),
):
    """List all permission requests made by the current user."""
    requests = (
        db.query(PermissionRequest)
        .filter(PermissionRequest.user_id == current_user.id)
        .order_by(PermissionRequest.created_at.desc())
        .all()
    )
    result = []
    for r in requests:
        result.append(
            PermissionRequestOut(
                id=r.id,
                user_id=r.user_id,
                username=current_user.username,
                api_key_id=r.api_key_id,
                api_key_label=r.api_key.label if r.api_key else None,
                model_id=r.model_id,
                model_display_name=r.model.display_name if r.model else None,
                provider_name=r.model.provider.name
                if r.model and r.model.provider
                else None,
                status=r.status.value,
                request_message=r.request_message,
                admin_message=r.admin_message,
                reviewed_by=r.reviewed_by,
                created_at=r.created_at,
            )
        )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Available Models (read-only, any user can see)
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/models", response_model=List[LLMModelOut])
def list_available_models(db: Session = Depends(get_db)):
    """List all active models (available to any authenticated user for browsing)."""
    models = (
        db.query(LLMModel)
        .join(Provider)
        .filter(LLMModel.is_active == True, Provider.is_active == True)
        .order_by(Provider.name, LLMModel.model_id)
        .all()
    )
    result = []
    for m in models:
        out = LLMModelOut.model_validate(m)
        out.provider_name = m.provider.name if m.provider else None
        result.append(out)
    return result
