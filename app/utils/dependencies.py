"""
FastAPI dependencies for authentication and authorization.
Extracts the current user from the Authorization header and provides
role-based guards (require_admin, require_user).
"""

from fastapi import Depends, HTTPException, status, Header
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import User, UserRole
from app.services.auth_service import verify_access_token, get_user_by_id
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_current_user(
    authorization: str = Header(..., description="Bearer <access_token>"),
    db: Session = Depends(get_db),
) -> User:
    """
    Extract and validate the JWT from the Authorization header.
    Returns the full User ORM object.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must start with 'Bearer '",
        )

    token = authorization[len("Bearer ") :]
    payload = verify_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
        )

    user = get_user_by_id(db, payload["user_id"])
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Dependency that ensures the current user has admin role."""
    if current_user.role != UserRole.admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user


def require_user(current_user: User = Depends(get_current_user)) -> User:
    """Dependency that just confirms the user is authenticated (any role)."""
    return current_user
