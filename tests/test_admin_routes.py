"""Tests for admin routes."""


class TestAdminUserManagement:
    def test_create_user(self, client, admin_headers):
        resp = client.post(
            "/api/admin/users",
            json={"username": "newuser", "password": "pass123", "role": "user"},
            headers=admin_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "newuser"
        assert data["role"] == "user"

    def test_create_duplicate_user(self, client, admin_headers, regular_user):
        resp = client.post(
            "/api/admin/users",
            json={"username": "testuser", "password": "pass123"},
            headers=admin_headers,
        )
        assert resp.status_code == 400

    def test_list_users(self, client, admin_headers, regular_user):
        resp = client.get("/api/admin/users", headers=admin_headers)
        assert resp.status_code == 200
        users = resp.json()
        # At least the admin and regular user
        assert len(users) >= 2

    def test_delete_user(self, client, admin_headers, regular_user):
        resp = client.delete(
            f"/api/admin/users/{regular_user.id}", headers=admin_headers
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_cannot_delete_self(self, client, admin_headers, admin_user):
        resp = client.delete(f"/api/admin/users/{admin_user.id}", headers=admin_headers)
        assert resp.status_code == 400

    def test_regular_user_cannot_access_admin(self, client, user_headers):
        resp = client.get("/api/admin/users", headers=user_headers)
        assert resp.status_code == 403


class TestProviderManagement:
    def test_create_provider(self, client, admin_headers):
        resp = client.post(
            "/api/admin/providers",
            json={
                "name": "anthropic",
                "display_name": "Anthropic",
                "base_url": "https://api.anthropic.com",
                "provider_type": "openai",
            },
            headers=admin_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "anthropic"

    def test_list_providers(self, client, admin_headers, openai_provider):
        resp = client.get("/api/admin/providers", headers=admin_headers)
        assert resp.status_code == 200
        providers = resp.json()
        assert len(providers) >= 1

    def test_update_provider(self, client, admin_headers, openai_provider):
        resp = client.put(
            f"/api/admin/providers/{openai_provider.id}",
            json={"display_name": "OpenAI Updated"},
            headers=admin_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["display_name"] == "OpenAI Updated"

    def test_delete_provider(self, client, admin_headers, openai_provider):
        resp = client.delete(
            f"/api/admin/providers/{openai_provider.id}", headers=admin_headers
        )
        assert resp.status_code == 200


class TestProviderApiKeys:
    def test_add_provider_api_key(self, client, admin_headers, openai_provider):
        resp = client.post(
            "/api/admin/provider-api-keys",
            json={
                "provider_id": openai_provider.id,
                "label": "my-key",
                "api_key": "sk-test123",
            },
            headers=admin_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "my-key"
        # Actual key should NOT be in the response
        assert "encrypted_key" not in data
        assert "api_key" not in data

    def test_list_provider_api_keys(
        self, client, admin_headers, openai_provider, openai_api_key
    ):
        resp = client.get(
            f"/api/admin/provider-api-keys/{openai_provider.id}",
            headers=admin_headers,
        )
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    def test_toggle_provider_api_key(self, client, admin_headers, openai_api_key):
        resp = client.patch(
            f"/api/admin/provider-api-keys/{openai_api_key.id}/toggle",
            headers=admin_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["is_active"] is False  # Was True, now toggled

    def test_delete_provider_api_key(self, client, admin_headers, openai_api_key):
        resp = client.delete(
            f"/api/admin/provider-api-keys/{openai_api_key.id}",
            headers=admin_headers,
        )
        assert resp.status_code == 200


class TestModelManagement:
    def test_create_model(self, client, admin_headers, openai_provider):
        resp = client.post(
            "/api/admin/models",
            json={
                "provider_id": openai_provider.id,
                "model_id": "gpt-4o-mini",
                "display_name": "GPT-4o Mini",
                "max_context_tokens": 128000,
            },
            headers=admin_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == "gpt-4o-mini"
        assert data["provider_name"] == "openai"

    def test_create_duplicate_model(self, client, admin_headers, gpt4_model):
        resp = client.post(
            "/api/admin/models",
            json={
                "provider_id": gpt4_model.provider_id,
                "model_id": "gpt-4o",
                "display_name": "GPT-4o Duplicate",
            },
            headers=admin_headers,
        )
        assert resp.status_code == 400

    def test_list_models(self, client, admin_headers, gpt4_model):
        resp = client.get("/api/admin/models", headers=admin_headers)
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    def test_update_model(self, client, admin_headers, gpt4_model):
        resp = client.put(
            f"/api/admin/models/{gpt4_model.id}",
            json={"display_name": "GPT-4o Updated"},
            headers=admin_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["display_name"] == "GPT-4o Updated"

    def test_delete_model(self, client, admin_headers, gpt4_model):
        resp = client.delete(
            f"/api/admin/models/{gpt4_model.id}", headers=admin_headers
        )
        assert resp.status_code == 200


class TestEnvironmentVariables:
    def test_create_env_var(self, client, admin_headers):
        resp = client.post(
            "/api/admin/env-vars",
            json={"key": "MY_SECRET", "value": "super-secret-123", "is_secret": True},
            headers=admin_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["key"] == "MY_SECRET"
        # Secret value should be masked
        assert data["value"] == "******"

    def test_create_non_secret_env_var(self, client, admin_headers):
        resp = client.post(
            "/api/admin/env-vars",
            json={"key": "PUBLIC_VAR", "value": "not-secret", "is_secret": False},
            headers=admin_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["value"] == "not-secret"

    def test_list_env_vars(self, client, admin_headers):
        # Create one first
        client.post(
            "/api/admin/env-vars",
            json={"key": "TEST_KEY", "value": "test-val"},
            headers=admin_headers,
        )
        resp = client.get("/api/admin/env-vars", headers=admin_headers)
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    def test_update_env_var(self, client, admin_headers):
        create_resp = client.post(
            "/api/admin/env-vars",
            json={"key": "UPD_KEY", "value": "old-val", "is_secret": False},
            headers=admin_headers,
        )
        env_id = create_resp.json()["id"]

        resp = client.put(
            f"/api/admin/env-vars/{env_id}",
            json={"value": "new-val"},
            headers=admin_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["value"] == "new-val"

    def test_delete_env_var(self, client, admin_headers):
        create_resp = client.post(
            "/api/admin/env-vars",
            json={"key": "DEL_KEY", "value": "delete-me"},
            headers=admin_headers,
        )
        env_id = create_resp.json()["id"]

        resp = client.delete(f"/api/admin/env-vars/{env_id}", headers=admin_headers)
        assert resp.status_code == 200

    def test_regular_user_cannot_access_env_vars(self, client, user_headers):
        resp = client.get("/api/admin/env-vars", headers=user_headers)
        assert resp.status_code == 403


class TestAdminDirectPermissionGrant:
    def test_grant_permission(
        self, client, admin_headers, db, regular_user, gpt4_model
    ):
        """Admin can directly grant model permission to a user's API key."""
        from app.db.models import GatewayApiKey
        import hashlib

        # Create a gateway API key for the user
        key_hash = hashlib.sha256(b"gw-test-key").hexdigest()
        gw_key = GatewayApiKey(
            user_id=regular_user.id,
            key_hash=key_hash,
            key_prefix="gw-test***",
            label="test",
        )
        db.add(gw_key)
        db.commit()
        db.refresh(gw_key)

        resp = client.post(
            "/api/admin/grant-permission",
            json={
                "api_key_id": gw_key.id,
                "model_id": gpt4_model.id,
                "max_output_tokens": 4096,
            },
            headers=admin_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True
