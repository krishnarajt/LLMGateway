import os
import sys
from pathlib import Path
from typing import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.auth_routes import router as auth_router
from app.db.database import get_db


# Ensure `app` package is importable when pytest is run from repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Required at import time by app.common.constants
os.environ.setdefault("SECRET_KEY", "test-secret-key")


class FakeQuery:
    def __init__(self, db: "FakeDB") -> None:
        self.db = db

    def filter(self, *args, **kwargs) -> "FakeQuery":
        return self

    def first(self):
        return self.db.existing_user

    def delete(self) -> int:
        return 1


class FakeDB:
    def __init__(self) -> None:
        self.existing_user = None

    def query(self, model) -> FakeQuery:
        return FakeQuery(self)

    def add(self, obj) -> None:
        return None

    def commit(self) -> None:
        return None

    def refresh(self, obj) -> None:
        return None

    def close(self) -> None:
        return None


@pytest.fixture
def fake_db() -> FakeDB:
    return FakeDB()


@pytest.fixture
def app(fake_db: FakeDB) -> FastAPI:
    app = FastAPI()
    app.include_router(auth_router, prefix="/api")

    def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    return app


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
