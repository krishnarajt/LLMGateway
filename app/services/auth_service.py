from datetime import datetime, timedelta, timezone
from typing import Optional
import uuid
import jwt
import hashlib
import secrets
from sqlalchemy.orm import Session

from app.db.models import User, RefreshToken
from app.utils.logging_utils import get_logger

# bring in configuration constants
from app.common import constants

logger = get_logger(__name__)

# Configuration
SECRET_KEY = constants.SECRET_KEY

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 30

# Password hashing configuration
SALT_LENGTH = 32  # 32 bytes = 256 bits
HASH_ITERATIONS = 100000  # OWASP recommended minimum


def _now() -> datetime:
    """Return current UTC time (timezone-aware)"""
    return datetime.now(timezone.utc)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a stored hash"""
    try:
        # Format: iterations$salt$hash
        parts = hashed_password.split("$")
        if len(parts) != 3:
            return False

        iterations = int(parts[0])
        salt = bytes.fromhex(parts[1])
        stored_hash = parts[2]

        # Hash the provided password with the same salt and iterations
        computed_hash = hashlib.pbkdf2_hmac(
            "sha256", plain_password.encode("utf-8"), salt, iterations
        ).hex()

        # Constant-time comparison to prevent timing attacks
        return secrets.compare_digest(computed_hash, stored_hash)
    except (ValueError, IndexError):
        return False


def get_password_hash(password: str) -> str:
    """Hash a password using PBKDF2-SHA256"""
    salt = secrets.token_bytes(SALT_LENGTH)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, HASH_ITERATIONS
    ).hex()
    # Format: iterations$salt$hash
    return f"{HASH_ITERATIONS}${salt.hex()}${password_hash}"


def create_access_token(user_id: int, expires_delta: Optional[timedelta] = None) -> str:
    """Create a short-lived JWT access token"""
    expire = _now() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode = {"sub": str(user_id), "exp": expire, "type": "access"}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(db: Session, user_id: int) -> str:
    """
    Create a long-lived refresh token.
    Only the jti (unique ID) is stored in the DB — not the full JWT.
    This keeps the DB row small and makes revocation O(1).
    """
    jti = str(uuid.uuid4())
    expires_at = _now() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode = {
        "sub": str(user_id),
        "exp": expires_at,
        "type": "refresh",
        "jti": jti,
    }
    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    # Store only the jti in DB, not the full token
    db_token = RefreshToken(user_id=user_id, token=jti, expires_at=expires_at)
    db.add(db_token)
    db.commit()

    return token


def verify_access_token(token: str) -> Optional[int]:
    """Verify an access token and return the user_id, or None if invalid"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "access":
            return None
        return int(payload["sub"])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError, KeyError, ValueError):
        return None


def verify_refresh_token(db: Session, token: str) -> Optional[int]:
    """
    Verify a refresh token by:
    1. Checking the JWT signature + expiry
    2. Confirming the jti exists in the DB (not revoked)
    Returns user_id if valid, None otherwise.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            return None

        user_id = int(payload["sub"])
        jti = payload.get("jti")

        if not jti:
            return None

        # Check jti exists in DB and is not expired
        db_token = (
            db.query(RefreshToken)
            .filter(
                RefreshToken.token == jti,
                RefreshToken.user_id == user_id,
                RefreshToken.expires_at > _now(),
            )
            .first()
        )

        return user_id if db_token else None

    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError, KeyError, ValueError):
        return None


def rotate_refresh_token(db: Session, old_token: str) -> Optional[tuple[int, str]]:
    """
    Verify the old refresh token, revoke it, and issue a new one.
    Returns (user_id, new_token) or None if the old token is invalid.
    This should be used in the /refresh endpoint instead of verify alone.
    """
    user_id = verify_refresh_token(db, old_token)
    if not user_id:
        return None

    # Revoke old token
    try:
        old_jti = jwt.decode(old_token, SECRET_KEY, algorithms=[ALGORITHM]).get("jti")
        db.query(RefreshToken).filter(RefreshToken.token == old_jti).delete()
        db.commit()
    except jwt.InvalidTokenError:
        return None

    # Issue new token
    new_token = create_refresh_token(db, user_id)
    return user_id, new_token


def revoke_refresh_token(db: Session, token: str) -> bool:
    """Revoke a single refresh token (logout)"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        jti = payload.get("jti")
        if not jti:
            return False
        deleted = db.query(RefreshToken).filter(RefreshToken.token == jti).delete()
        db.commit()
        return deleted > 0
    except jwt.InvalidTokenError:
        return False


def revoke_all_user_tokens(db: Session, user_id: int) -> None:
    """Revoke all refresh tokens for a user (logout everywhere)"""
    db.query(RefreshToken).filter(RefreshToken.user_id == user_id).delete()
    db.commit()


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Verify username + password, return User or None"""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def create_user(db: Session, username: str, password: str) -> User:
    """Create a new user with a hashed password"""
    hashed_password = get_password_hash(password)
    db_user = User(username=username, password_hash=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()
