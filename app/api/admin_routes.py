"""
Admin routes — only accessible to users with admin role.
Covers:
  - User management (create, list, delete)
  - Provider management (CRUD)
  - Provider API key management (add, list, toggle, delete)
  - LLM model management (CRUD)
  - Environment variables (encrypted KV store)
  - Permission request review (approve / reject)
  - Direct permission grants to API keys
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.db.database import get_db
from app.db.models import (
    User,
    UserRole,
    Provider,
    ProviderApiKey,
    LLMModel,
    EnvironmentVariable,
    PermissionRequest,
    PermissionRequestStatus,
    GatewayApiKey,
    ApiKeyModelPermission,
)
from app.common.schemas import (
    CreateUserRequest,
    UserOut,
    ProviderCreate,
    ProviderUpdate,
    ProviderOut,
    ProviderApiKeyCreate,
    ProviderApiKeyOut,
    LLMModelCreate,
    LLMModelUpdate,
    LLMModelOut,
    EnvVarCreate,
    EnvVarUpdate,
    EnvVarOut,
    PermissionRequestOut,
    PermissionRequestReview,
    AdminGrantPermission,
)
from app.services.auth_service import create_user, maybe_delete_default_admin
from app.utils.dependencies import require_admin
from app.utils.encryption import encrypt_value, decrypt_value
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/admin", tags=["Admin"], dependencies=[Depends(require_admin)]
)


# ═══════════════════════════════════════════════════════════════════════════
# User Management
# ═══════════════════════════════════════════════════════════════════════════


@router.post("/users", response_model=UserOut)
def admin_create_user(
    request: CreateUserRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Create a new user (admin or regular). Only admins can do this."""
    existing = db.query(User).filter(User.username == request.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    user = create_user(
        db,
        request.username,
        request.password,
        role=request.role,
        display_name=request.display_name or "",
    )

    # If we just created a new admin, try to delete the default admin
    if user.role == UserRole.admin:
        maybe_delete_default_admin(db, user.id)

    return UserOut.model_validate(user)


@router.get("/users", response_model=List[UserOut])
def admin_list_users(db: Session = Depends(get_db)):
    """List all users."""
    users = db.query(User).order_by(User.id).all()
    return [UserOut.model_validate(u) for u in users]


@router.delete("/users/{user_id}")
def admin_delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Delete a user (cannot delete yourself)."""
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)
    db.commit()
    return {"success": True, "message": f"User '{user.username}' deleted"}


# ═══════════════════════════════════════════════════════════════════════════
# Providers
# ═══════════════════════════════════════════════════════════════════════════


@router.post("/providers", response_model=ProviderOut)
def admin_create_provider(request: ProviderCreate, db: Session = Depends(get_db)):
    """Add a new LLM provider."""
    existing = db.query(Provider).filter(Provider.name == request.name).first()
    if existing:
        raise HTTPException(
            status_code=400, detail=f"Provider '{request.name}' already exists"
        )
    provider = Provider(**request.model_dump())
    db.add(provider)
    db.commit()
    db.refresh(provider)
    return ProviderOut.model_validate(provider)


@router.get("/providers", response_model=List[ProviderOut])
def admin_list_providers(db: Session = Depends(get_db)):
    """List all providers."""
    providers = db.query(Provider).order_by(Provider.id).all()
    return [ProviderOut.model_validate(p) for p in providers]


@router.put("/providers/{provider_id}", response_model=ProviderOut)
def admin_update_provider(
    provider_id: int, request: ProviderUpdate, db: Session = Depends(get_db)
):
    """Update provider details."""
    provider = db.query(Provider).filter(Provider.id == provider_id).first()
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    for field, value in request.model_dump(exclude_unset=True).items():
        setattr(provider, field, value)
    db.commit()
    db.refresh(provider)
    return ProviderOut.model_validate(provider)


@router.delete("/providers/{provider_id}")
def admin_delete_provider(provider_id: int, db: Session = Depends(get_db)):
    """Delete a provider and all its API keys and models (cascade)."""
    provider = db.query(Provider).filter(Provider.id == provider_id).first()
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    db.delete(provider)
    db.commit()
    return {"success": True, "message": f"Provider '{provider.name}' deleted"}


# ═══════════════════════════════════════════════════════════════════════════
# Provider API Keys
# ═══════════════════════════════════════════════════════════════════════════


@router.post("/provider-api-keys", response_model=ProviderApiKeyOut)
def admin_add_provider_api_key(
    request: ProviderApiKeyCreate, db: Session = Depends(get_db)
):
    """Add an encrypted API key for a provider."""
    provider = db.query(Provider).filter(Provider.id == request.provider_id).first()
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    encrypted = encrypt_value(request.api_key)
    key_obj = ProviderApiKey(
        provider_id=request.provider_id,
        label=request.label,
        encrypted_key=encrypted,
    )
    db.add(key_obj)
    db.commit()
    db.refresh(key_obj)
    return ProviderApiKeyOut.model_validate(key_obj)


@router.get("/provider-api-keys/{provider_id}", response_model=List[ProviderApiKeyOut])
def admin_list_provider_api_keys(provider_id: int, db: Session = Depends(get_db)):
    """List all API keys for a provider (metadata only — never exposes the actual key)."""
    keys = (
        db.query(ProviderApiKey)
        .filter(ProviderApiKey.provider_id == provider_id)
        .order_by(ProviderApiKey.id)
        .all()
    )
    return [ProviderApiKeyOut.model_validate(k) for k in keys]


@router.delete("/provider-api-keys/{key_id}")
def admin_delete_provider_api_key(key_id: int, db: Session = Depends(get_db)):
    """Delete a provider API key."""
    key_obj = db.query(ProviderApiKey).filter(ProviderApiKey.id == key_id).first()
    if not key_obj:
        raise HTTPException(status_code=404, detail="Provider API key not found")
    db.delete(key_obj)
    db.commit()
    return {"success": True, "message": "Provider API key deleted"}


@router.patch("/provider-api-keys/{key_id}/toggle")
def admin_toggle_provider_api_key(key_id: int, db: Session = Depends(get_db)):
    """Toggle active/inactive status of a provider API key."""
    key_obj = db.query(ProviderApiKey).filter(ProviderApiKey.id == key_id).first()
    if not key_obj:
        raise HTTPException(status_code=404, detail="Provider API key not found")
    key_obj.is_active = not key_obj.is_active
    db.commit()
    return {"success": True, "is_active": key_obj.is_active}


# ═══════════════════════════════════════════════════════════════════════════
# LLM Models
# ═══════════════════════════════════════════════════════════════════════════


@router.post("/models", response_model=LLMModelOut)
def admin_create_model(request: LLMModelCreate, db: Session = Depends(get_db)):
    """Register a new LLM model under a provider."""
    provider = db.query(Provider).filter(Provider.id == request.provider_id).first()
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    existing = (
        db.query(LLMModel)
        .filter(
            LLMModel.provider_id == request.provider_id,
            LLMModel.model_id == request.model_id,
        )
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model_id}' already exists for this provider",
        )

    model = LLMModel(**request.model_dump())
    db.add(model)
    db.commit()
    db.refresh(model)

    out = LLMModelOut.model_validate(model)
    out.provider_name = provider.name
    return out


@router.get("/models", response_model=List[LLMModelOut])
def admin_list_models(db: Session = Depends(get_db)):
    """List all models across all providers."""
    models = db.query(LLMModel).order_by(LLMModel.id).all()
    result = []
    for m in models:
        out = LLMModelOut.model_validate(m)
        out.provider_name = m.provider.name if m.provider else None
        result.append(out)
    return result


@router.put("/models/{model_id}", response_model=LLMModelOut)
def admin_update_model(
    model_id: int, request: LLMModelUpdate, db: Session = Depends(get_db)
):
    """Update a model's metadata."""
    model = db.query(LLMModel).filter(LLMModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    for field, value in request.model_dump(exclude_unset=True).items():
        setattr(model, field, value)
    db.commit()
    db.refresh(model)
    out = LLMModelOut.model_validate(model)
    out.provider_name = model.provider.name if model.provider else None
    return out


@router.delete("/models/{model_id}")
def admin_delete_model(model_id: int, db: Session = Depends(get_db)):
    """Delete a model."""
    model = db.query(LLMModel).filter(LLMModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    db.delete(model)
    db.commit()
    return {"success": True, "message": f"Model '{model.model_id}' deleted"}


# ═══════════════════════════════════════════════════════════════════════════
# Environment Variables
# ═══════════════════════════════════════════════════════════════════════════


@router.post("/env-vars", response_model=EnvVarOut)
def admin_create_env_var(request: EnvVarCreate, db: Session = Depends(get_db)):
    """Create a new encrypted environment variable."""
    existing = (
        db.query(EnvironmentVariable)
        .filter(EnvironmentVariable.key == request.key)
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=400, detail=f"Key '{request.key}' already exists"
        )

    env_var = EnvironmentVariable(
        key=request.key,
        encrypted_value=encrypt_value(request.value),
        description=request.description,
        is_secret=request.is_secret,
    )
    db.add(env_var)
    db.commit()
    db.refresh(env_var)
    return _env_var_to_out(env_var)


@router.get("/env-vars", response_model=List[EnvVarOut])
def admin_list_env_vars(db: Session = Depends(get_db)):
    """List all environment variables. Secret values are masked."""
    env_vars = db.query(EnvironmentVariable).order_by(EnvironmentVariable.key).all()
    return [_env_var_to_out(ev) for ev in env_vars]


@router.put("/env-vars/{env_id}", response_model=EnvVarOut)
def admin_update_env_var(
    env_id: int, request: EnvVarUpdate, db: Session = Depends(get_db)
):
    """Update an environment variable."""
    env_var = (
        db.query(EnvironmentVariable).filter(EnvironmentVariable.id == env_id).first()
    )
    if not env_var:
        raise HTTPException(status_code=404, detail="Environment variable not found")
    if request.value is not None:
        env_var.encrypted_value = encrypt_value(request.value)
    if request.description is not None:
        env_var.description = request.description
    if request.is_secret is not None:
        env_var.is_secret = request.is_secret
    db.commit()
    db.refresh(env_var)
    return _env_var_to_out(env_var)


@router.delete("/env-vars/{env_id}")
def admin_delete_env_var(env_id: int, db: Session = Depends(get_db)):
    """Delete an environment variable."""
    env_var = (
        db.query(EnvironmentVariable).filter(EnvironmentVariable.id == env_id).first()
    )
    if not env_var:
        raise HTTPException(status_code=404, detail="Environment variable not found")
    db.delete(env_var)
    db.commit()
    return {"success": True, "message": f"Env var '{env_var.key}' deleted"}


def _env_var_to_out(ev: EnvironmentVariable) -> EnvVarOut:
    """Convert an EnvironmentVariable to its API representation, masking secrets."""
    value = "******" if ev.is_secret else decrypt_value(ev.encrypted_value)
    return EnvVarOut(
        id=ev.id,
        key=ev.key,
        value=value,
        description=ev.description,
        is_secret=ev.is_secret,
        created_at=ev.created_at,
        updated_at=ev.updated_at,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Permission Request Review
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/permission-requests", response_model=List[PermissionRequestOut])
def admin_list_permission_requests(
    status_filter: str = None,
    db: Session = Depends(get_db),
):
    """List permission requests. Optionally filter by status (pending/approved/rejected)."""
    query = db.query(PermissionRequest).order_by(PermissionRequest.created_at.desc())
    if status_filter:
        query = query.filter(PermissionRequest.status == status_filter)
    requests = query.all()
    return [_perm_request_to_out(r) for r in requests]


@router.put("/permission-requests/{request_id}")
def admin_review_permission_request(
    request_id: int,
    review: PermissionRequestReview,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Approve or reject a permission request."""
    perm_req = (
        db.query(PermissionRequest).filter(PermissionRequest.id == request_id).first()
    )
    if not perm_req:
        raise HTTPException(status_code=404, detail="Permission request not found")
    if perm_req.status != PermissionRequestStatus.pending:
        raise HTTPException(status_code=400, detail="Request has already been reviewed")

    perm_req.status = PermissionRequestStatus(review.status)
    perm_req.admin_message = review.admin_message
    perm_req.reviewed_by = admin.id
    db.commit()

    # If approved, create the actual permission
    if review.status == "approved":
        # Check if permission already exists
        existing = (
            db.query(ApiKeyModelPermission)
            .filter(
                ApiKeyModelPermission.api_key_id == perm_req.api_key_id,
                ApiKeyModelPermission.model_id == perm_req.model_id,
            )
            .first()
        )
        if not existing:
            perm = ApiKeyModelPermission(
                api_key_id=perm_req.api_key_id,
                model_id=perm_req.model_id,
                max_input_tokens=review.max_input_tokens,
                max_output_tokens=review.max_output_tokens,
            )
            db.add(perm)
            db.commit()

    return {"success": True, "message": f"Request {review.status}"}


# ═══════════════════════════════════════════════════════════════════════════
# Direct Permission Grant
# ═══════════════════════════════════════════════════════════════════════════


@router.post("/grant-permission")
def admin_grant_permission(
    request: AdminGrantPermission,
    db: Session = Depends(get_db),
):
    """Admin directly grants a model permission to a gateway API key."""
    api_key = (
        db.query(GatewayApiKey).filter(GatewayApiKey.id == request.api_key_id).first()
    )
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")
    model = db.query(LLMModel).filter(LLMModel.id == request.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    existing = (
        db.query(ApiKeyModelPermission)
        .filter(
            ApiKeyModelPermission.api_key_id == request.api_key_id,
            ApiKeyModelPermission.model_id == request.model_id,
        )
        .first()
    )
    if existing:
        raise HTTPException(status_code=400, detail="Permission already exists")

    perm = ApiKeyModelPermission(
        api_key_id=request.api_key_id,
        model_id=request.model_id,
        max_input_tokens=request.max_input_tokens,
        max_output_tokens=request.max_output_tokens,
    )
    db.add(perm)
    db.commit()
    return {"success": True, "message": "Permission granted"}


def _perm_request_to_out(r: PermissionRequest) -> PermissionRequestOut:
    """Convert a PermissionRequest ORM object to the API schema."""
    return PermissionRequestOut(
        id=r.id,
        user_id=r.user_id,
        username=r.user.username if r.user else None,
        api_key_id=r.api_key_id,
        api_key_label=r.api_key.label if r.api_key else None,
        model_id=r.model_id,
        model_display_name=r.model.display_name if r.model else None,
        provider_name=r.model.provider.name if r.model and r.model.provider else None,
        status=r.status.value,
        request_message=r.request_message,
        admin_message=r.admin_message,
        reviewed_by=r.reviewed_by,
        created_at=r.created_at,
    )
