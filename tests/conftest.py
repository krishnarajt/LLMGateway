"""
Test fixtures using SQLite in-memory database for fast, isolated tests.
"""

import os
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Ensure app package is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Required at import time by app.common.constants
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-testing-only")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.db.database import Base, get_db
from app.db.models import User, UserRole, Provider, LLMModel, ProviderApiKey
from app.services.auth_service import get_password_hash, create_access_token
from app.utils.encryption import encrypt_value

from app.api.auth_routes import router as auth_router
from app.api.admin_routes import router as admin_router
from app.api.user_routes import router as user_router
from app.api.chat_routes import router as chat_router


# ---------------------------------------------------------------------------
# SQLite in-memory engine for tests
# ---------------------------------------------------------------------------

TEST_ENGINE = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

# SQLite doesn't have schemas — strip the schema from metadata AND every table
Base.metadata.schema = None
for table in Base.metadata.tables.values():
    table.schema = None

TestSession = sessionmaker(autocommit=False, autoflush=False, bind=TEST_ENGINE)


@pytest.fixture(autouse=True)
def setup_database():
    """Create all tables before each test, drop after."""
    Base.metadata.create_all(bind=TEST_ENGINE)
    yield
    Base.metadata.drop_all(bind=TEST_ENGINE)


@pytest.fixture
def db():
    """Provide a fresh DB session for each test."""
    session = TestSession()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def app(db):
    """FastAPI app with all routers and the test DB session."""
    _app = FastAPI()
    _app.include_router(auth_router, prefix="/api")
    _app.include_router(admin_router, prefix="/api")
    _app.include_router(user_router, prefix="/api")
    _app.include_router(chat_router, prefix="/api")

    def override_get_db():
        yield db

    _app.dependency_overrides[get_db] = override_get_db
    return _app


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def admin_user(db) -> User:
    """Create an admin user and return it."""
    user = User(
        username="testadmin",
        password_hash=get_password_hash("adminpass"),
        role=UserRole.admin,
        display_name="Test Admin",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def regular_user(db) -> User:
    """Create a regular user and return it."""
    user = User(
        username="testuser",
        password_hash=get_password_hash("userpass"),
        role=UserRole.user,
        display_name="Test User",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def admin_token(admin_user) -> str:
    """JWT access token for the admin user."""
    return create_access_token(admin_user.id, role="admin")


@pytest.fixture
def user_token(regular_user) -> str:
    """JWT access token for the regular user."""
    return create_access_token(regular_user.id, role="user")


@pytest.fixture
def admin_headers(admin_token) -> dict:
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture
def user_headers(user_token) -> dict:
    return {"Authorization": f"Bearer {user_token}"}


@pytest.fixture
def openai_provider(db) -> Provider:
    """Seed an OpenAI provider."""
    p = Provider(
        name="openai",
        display_name="OpenAI",
        base_url="https://api.openai.com/v1",
        provider_type="openai",
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


@pytest.fixture
def gemini_provider(db) -> Provider:
    """Seed a Gemini provider."""
    p = Provider(
        name="gemini",
        display_name="Google Gemini",
        base_url="https://generativelanguage.googleapis.com",
        provider_type="gemini",
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


@pytest.fixture
def ollama_provider(db) -> Provider:
    """Seed an Ollama provider."""
    p = Provider(
        name="ollama",
        display_name="Ollama",
        base_url="http://localhost:11434",
        provider_type="ollama",
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


@pytest.fixture
def gpt4_model(db, openai_provider) -> LLMModel:
    """Seed a GPT-4o model."""
    m = LLMModel(
        provider_id=openai_provider.id,
        model_id="gpt-4o",
        display_name="GPT-4o",
        max_context_tokens=128000,
    )
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


@pytest.fixture
def openai_api_key(db, openai_provider) -> ProviderApiKey:
    """Seed an encrypted API key for OpenAI."""
    k = ProviderApiKey(
        provider_id=openai_provider.id,
        label="test-key",
        encrypted_key=encrypt_value("sk-test-fake-key-12345"),
    )
    db.add(k)
    db.commit()
    db.refresh(k)
    return k
