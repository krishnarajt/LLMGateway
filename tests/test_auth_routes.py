"""Tests for authentication routes."""

from app.db.models import User, UserRole
from app.services.auth_service import get_password_hash


class TestLogin:
    def test_login_success(self, client, admin_user):
        resp = client.post(
            "/api/auth/login", json={"username": "testadmin", "password": "adminpass"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "accessToken" in data
        assert "refreshToken" in data
        assert data["role"] == "admin"

    def test_login_invalid_password(self, client, admin_user):
        resp = client.post(
            "/api/auth/login", json={"username": "testadmin", "password": "wrong"}
        )
        assert resp.status_code == 401

    def test_login_nonexistent_user(self, client):
        resp = client.post(
            "/api/auth/login", json={"username": "ghost", "password": "pass"}
        )
        assert resp.status_code == 401

    def test_login_default_admin_warning(self, client, db):
        """When logging in as the default admin, the response should contain a warning."""
        user = User(
            username="admin",
            password_hash=get_password_hash("admin"),
            role=UserRole.admin,
            is_default_admin=True,
            must_change_password=True,
        )
        db.add(user)
        db.commit()

        resp = client.post(
            "/api/auth/login", json={"username": "admin", "password": "admin"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_default_admin"] is True
        assert data["must_change_password"] is True
        assert "default admin" in data["message"].lower()


class TestRefreshAndLogout:
    def test_refresh_token(self, client, admin_user):
        # Login first
        login_resp = client.post(
            "/api/auth/login", json={"username": "testadmin", "password": "adminpass"}
        )
        refresh_token = login_resp.json()["refreshToken"]

        # Refresh
        resp = client.post("/api/auth/refresh", json={"refreshToken": refresh_token})
        assert resp.status_code == 200
        assert "accessToken" in resp.json()

    def test_refresh_invalid_token(self, client):
        resp = client.post("/api/auth/refresh", json={"refreshToken": "invalid-token"})
        assert resp.status_code == 401

    def test_logout(self, client, admin_user):
        login_resp = client.post(
            "/api/auth/login", json={"username": "testadmin", "password": "adminpass"}
        )
        refresh_token = login_resp.json()["refreshToken"]

        resp = client.post("/api/auth/logout", json={"refreshToken": refresh_token})
        assert resp.status_code == 200
        assert resp.json()["success"] is True


class TestChangePassword:
    def test_change_password(self, client, admin_user, admin_headers):
        resp = client.post(
            "/api/auth/change-password",
            json={"current_password": "adminpass", "new_password": "newpass123"},
            headers=admin_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

        # Login with new password
        resp = client.post(
            "/api/auth/login", json={"username": "testadmin", "password": "newpass123"}
        )
        assert resp.status_code == 200

    def test_change_password_wrong_current(self, client, admin_user, admin_headers):
        resp = client.post(
            "/api/auth/change-password",
            json={"current_password": "wrongpass", "new_password": "newpass123"},
            headers=admin_headers,
        )
        assert resp.status_code == 400


class TestMe:
    def test_me_endpoint(self, client, admin_user, admin_headers):
        resp = client.get("/api/auth/me", headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "testadmin"
        assert data["role"] == "admin"

    def test_me_unauthorized(self, client):
        resp = client.get("/api/auth/me", headers={"Authorization": "Bearer invalid"})
        assert resp.status_code == 401
