"""Application-wide constants loaded from environment variables.

Every other module should import needed values from here rather than
accessing :mod:`os.environ` directly.  This centralises configuration and
makes it easier to mock during tests.

The module also ensures ``.env`` files are loaded early via :func:`load_dotenv`.
"""

from __future__ import annotations

import os
from typing import List

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

# ---------------------------------------------------------------------------
# Database settings
# ---------------------------------------------------------------------------
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/postgres",
)
DB_SCHEMA: str = os.getenv("DB_SCHEMA", "public")

# ---------------------------------------------------------------------------
# Security / auth
# ---------------------------------------------------------------------------
SECRET_KEY: str = os.getenv("SECRET_KEY") or ""
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY environment variable is not set!")
