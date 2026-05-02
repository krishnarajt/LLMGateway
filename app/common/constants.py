"""Application-wide constants loaded from environment variables.

Every other module should import needed values from here rather than
accessing :mod:`os.environ` directly.  This centralises configuration and
makes it easier to mock during tests.

The module also ensures ``.env`` files are loaded early via :func:`load_dotenv`.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List

from dotenv import load_dotenv

# Load environment variables from .env file (if present).  This happens
# on import so any module importing constants will have the vars available.
load_dotenv()

# ---------------------------------------------------------------------------
# Generic environment flags
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR: str = os.getenv("LOG_DIR", "logs")

# ---------------------------------------------------------------------------
# Server / FastAPI settings
# ---------------------------------------------------------------------------
PORT: int = int(os.getenv("PORT", "8000"))
RELOAD: bool = os.getenv("RELOAD", "false").lower() == "true"

# CORS origins are a comma-separated list.  ``*`` means allow all.
CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "*").split(",")


def _split_key_list(value: str) -> List[str]:
    """Split comma, semicolon, or newline separated provider key lists."""
    return [item.strip() for item in re.split(r"[,;\n]+", value or "") if item.strip()]


# ---------------------------------------------------------------------------
# Database settings
# ---------------------------------------------------------------------------
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/postgres",
)
DB_SCHEMA: str = os.getenv("DB_SCHEMA", "public")

# ---------------------------------------------------------------------------
# Provider API key pools
# ---------------------------------------------------------------------------
# These optional env vars are appended after DB-managed provider keys and are
# used for internal provider-key rotation before model fallback is attempted.
PROVIDER_API_KEYS_BY_TYPE: Dict[str, List[str]] = {
    "openai": _split_key_list(os.getenv("OPENAI_API_KEYS", "")),
    "gemini": _split_key_list(os.getenv("GEMINI_API_KEYS", "")),
    "groq": _split_key_list(os.getenv("GROQ_API_KEYS", "")),
    "huggingface": _split_key_list(os.getenv("HUGGINGFACE_API_KEYS", "")),
}

PROVIDER_API_KEY_ENV_ALIASES: Dict[str, List[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "groq": ["GROQ_API_KEY"],
    "huggingface": [
        "HF_TOKEN",
        "HUGGINGFACE_API_KEY",
        "HUGGING_FACE_API_KEY",
        "HUGGING_FACE_API_KEYS",
        "HUGGINGFACEHUB_API_TOKEN",
    ],
}


def _env_name(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def get_provider_api_keys_from_env(provider_name: str, provider_type: str) -> List[str]:
    """Last-resort provider-key pool from container env vars."""
    candidates = [
        f"{_env_name(provider_name)}_API_KEYS",
        f"{_env_name(provider_type)}_API_KEYS",
        f"{_env_name(provider_name)}_API_KEY",
        f"{_env_name(provider_type)}_API_KEY",
    ]

    keys: List[str] = []
    for env_key in dict.fromkeys(candidates):
        keys.extend(_split_key_list(os.getenv(env_key, "")))
    if not keys:
        for env_key in PROVIDER_API_KEY_ENV_ALIASES.get(provider_type, []):
            keys.extend(_split_key_list(os.getenv(env_key, "")))
    if not keys:
        keys.extend(PROVIDER_API_KEYS_BY_TYPE.get(provider_type, []))
    return list(dict.fromkeys(keys))

# ---------------------------------------------------------------------------
# Security / auth
# ---------------------------------------------------------------------------
SECRET_KEY: str = os.getenv("SECRET_KEY") or ""
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY environment variable is not set!")
