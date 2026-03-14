"""
Encryption utility for securing sensitive data (API keys, env vars) in the database.

Uses Fernet symmetric encryption with a key derived from the application SECRET_KEY.
All values stored in the database that are sensitive should go through encrypt_value/decrypt_value.
"""

import base64
import hashlib

from cryptography.fernet import Fernet

from app.common import constants
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _derive_fernet_key(secret: str) -> bytes:
    """
    Derive a 32-byte Fernet-compatible key from the application SECRET_KEY.
    Uses SHA-256 to produce a deterministic 32-byte digest, then base64-encodes it
    because Fernet requires a url-safe base64-encoded 32-byte key.
    """
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


# Build the Fernet cipher once at module load time
_fernet = Fernet(_derive_fernet_key(constants.SECRET_KEY))


def encrypt_value(plaintext: str) -> str:
    """Encrypt a plaintext string and return a base64-encoded ciphertext string."""
    return _fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt_value(ciphertext: str) -> str:
    """Decrypt a base64-encoded ciphertext string back to plaintext."""
    return _fernet.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
