"""
Auth routes — login, signup (disabled for self-service; admin creates users),
refresh, logout, change-password.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import User, UserRole
from app.common.schemas import (
    LoginRequest,
    AuthResponse,
    RefreshRequest,
    ChangePasswordRequest,
)
from app.services.auth_service import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    rotate_refresh_token,
    revoke_refresh_token,
    change_password,
    maybe_delete_default_admin,
    verify_password,
)
from app.utils.dependencies import get_current_user
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=AuthResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    """
    Login with username and password.
    Returns tokens plus metadata the UI needs (role, is_default_admin, must_change_password).
    """
    logger.info(f"Login attempt for username: {request.username}")
    user = authenticate_user(db, request.username, request.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    access_token = create_access_token(user.id, role=user.role.value)
    refresh_token = create_refresh_token(db, user.id)

    # Build response messages for the UI
    messages = []
    if user.is_default_admin:
        messages.append(
            "You are logged in as the default admin (admin/admin). "
            "Please create a new admin user immediately and then log in with that account. "
            "The default admin will be deleted once a real admin logs in."
        )
    if user.must_change_password:
        messages.append("You must change your password before proceeding.")

    # If this is a real (non-default) admin logging in, try to clean up the default admin
    if user.role == UserRole.admin and not user.is_default_admin:
        deleted = maybe_delete_default_admin(db, user.id)
        if deleted:
            messages.append("The default admin account has been removed.")

    return AuthResponse(
        accessToken=access_token,
        refreshToken=refresh_token,
        message=" | ".join(messages) if messages else "Login successful",
        role=user.role.value,
        is_default_admin=user.is_default_admin,
        must_change_password=user.must_change_password,
    )


@router.post("/refresh", response_model=AuthResponse)
def refresh_token(request: RefreshRequest, db: Session = Depends(get_db)):
    result = rotate_refresh_token(db, request.refreshToken)
    if not result:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    user_id, new_refresh_token = result
    user = db.query(User).filter(User.id == user_id).first()
    role = user.role.value if user else "user"

    return AuthResponse(
        accessToken=create_access_token(user_id, role=role),
        refreshToken=new_refresh_token,
        message="Token refreshed",
        role=role,
    )


@router.post("/logout")
def logout(request: RefreshRequest, db: Session = Depends(get_db)):
    """Logout and revoke refresh token"""
    revoke_refresh_token(db, request.refreshToken)
    return {"success": True, "message": "Logged out successfully"}


@router.post("/change-password")
def change_password_route(
    request: ChangePasswordRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Change the current user's password."""
    if not verify_password(request.current_password, current_user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    change_password(db, current_user, request.new_password)
    return {"success": True, "message": "Password changed successfully"}


@router.get("/me")
def me(current_user: User = Depends(get_current_user)):
    """Return the current user's profile."""
    return {
        "id": current_user.id,
        "username": current_user.username,
        "role": current_user.role.value,
        "display_name": current_user.display_name,
        "is_default_admin": current_user.is_default_admin,
        "must_change_password": current_user.must_change_password,
    }
