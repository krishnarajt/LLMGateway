from types import SimpleNamespace

import pytest

from app.api import auth_routes


def test_signup_success(client, monkeypatch):
    monkeypatch.setattr(
        auth_routes, "create_user", lambda db, username, password: SimpleNamespace(id=1)
    )
    monkeypatch.setattr(auth_routes, "create_access_token", lambda user_id: "access-1")
    monkeypatch.setattr(
        auth_routes, "create_refresh_token", lambda db, user_id: "refresh-1"
    )

    response = client.post(
        "/api/auth/signup", json={"username": "alice", "password": "password123"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "accessToken": "access-1",
        "refreshToken": "refresh-1",
        "message": "Account created successfully",
    }


def test_signup_duplicate_username_returns_400(client, fake_db):
    fake_db.existing_user = SimpleNamespace(id=99, username="alice")

    response = client.post(
        "/api/auth/signup", json={"username": "alice", "password": "password123"}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Username already exists"


def test_login_success(client, monkeypatch):
    monkeypatch.setattr(
        auth_routes,
        "authenticate_user",
        lambda db, username, password: SimpleNamespace(id=7),
    )
    monkeypatch.setattr(auth_routes, "create_access_token", lambda user_id: "access-7")
    monkeypatch.setattr(
        auth_routes, "create_refresh_token", lambda db, user_id: "refresh-7"
    )

    response = client.post(
        "/api/auth/login", json={"username": "alice", "password": "password123"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "accessToken": "access-7",
        "refreshToken": "refresh-7",
        "message": "Login successful",
    }


def test_login_invalid_credentials_returns_401(client, monkeypatch):
    monkeypatch.setattr(
        auth_routes, "authenticate_user", lambda db, username, password: None
    )

    response = client.post(
        "/api/auth/login", json={"username": "alice", "password": "wrong"}
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid username or password"


def test_refresh_success(client, monkeypatch):
    monkeypatch.setattr(
        auth_routes,
        "rotate_refresh_token",
        lambda db, token: (42, "refresh-new-42"),
    )
    monkeypatch.setattr(auth_routes, "create_access_token", lambda user_id: "access-42")

    response = client.post("/api/auth/refresh", json={"refreshToken": "refresh-old-42"})

    assert response.status_code == 200
    assert response.json() == {
        "accessToken": "access-42",
        "refreshToken": "refresh-new-42",
        "message": "Token refreshed",
    }


def test_refresh_invalid_token_returns_401(client, monkeypatch):
    monkeypatch.setattr(auth_routes, "rotate_refresh_token", lambda db, token: None)

    response = client.post("/api/auth/refresh", json={"refreshToken": "bad-token"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or expired refresh token"


def test_logout_success(client, monkeypatch):
    called = {"count": 0}

    def _revoke(db, token):
        called["count"] += 1
        return True

    monkeypatch.setattr(auth_routes, "revoke_refresh_token", _revoke)

    response = client.post("/api/auth/logout", json={"refreshToken": "refresh-1"})

    assert response.status_code == 200
    assert response.json() == {"success": True, "message": "Logged out successfully"}
    assert called["count"] == 1


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/api/auth/signup", {"username": "alice"}),
        ("/api/auth/login", {"username": "alice"}),
        ("/api/auth/refresh", {}),
        ("/api/auth/logout", {}),
    ],
)
def test_auth_routes_validation_errors(client, path, payload):
    response = client.post(path, json=payload)
    assert response.status_code == 422
