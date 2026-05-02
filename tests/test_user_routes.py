"""Tests for user routes — API key management, permission requests."""

from app.db.models import ApiKeyModelPermission, LLMModel


class TestGatewayApiKeys:
    def test_create_api_key(self, client, user_headers):
        resp = client.post(
            "/api/user/api-keys",
            json={"label": "my-key"},
            headers=user_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "my-key"
        # The raw key should be returned only this once
        assert data["key"].startswith("gw-")
        assert "***" in data["key_prefix"]

    def test_list_api_keys(self, client, user_headers):
        # Create a key first
        client.post(
            "/api/user/api-keys", json={"label": "list-test"}, headers=user_headers
        )

        resp = client.get("/api/user/api-keys", headers=user_headers)
        assert resp.status_code == 200
        keys = resp.json()
        assert len(keys) >= 1
        # Raw key should NOT be in the list response
        for k in keys:
            assert "key" not in k or k.get("key") is None

    def test_toggle_api_key(self, client, user_headers):
        create_resp = client.post(
            "/api/user/api-keys", json={"label": "toggle-test"}, headers=user_headers
        )
        key_id = create_resp.json()["id"]

        resp = client.patch(f"/api/user/api-keys/{key_id}/toggle", headers=user_headers)
        assert resp.status_code == 200
        assert resp.json()["is_active"] is False

    def test_revoke_api_key(self, client, user_headers):
        create_resp = client.post(
            "/api/user/api-keys", json={"label": "revoke-test"}, headers=user_headers
        )
        key_id = create_resp.json()["id"]

        resp = client.delete(f"/api/user/api-keys/{key_id}", headers=user_headers)
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_cannot_revoke_other_users_key(self, client, admin_headers, user_headers):
        # Create key as regular user
        create_resp = client.post(
            "/api/user/api-keys", json={"label": "other"}, headers=user_headers
        )
        key_id = create_resp.json()["id"]

        # Try to delete as admin (different user) — should 404 because
        # the endpoint filters by current_user.id
        resp = client.delete(f"/api/user/api-keys/{key_id}", headers=admin_headers)
        assert resp.status_code == 404


class TestPermissionRequests:
    def test_create_permission_request(self, client, user_headers, gpt4_model):
        # Create an API key first
        key_resp = client.post(
            "/api/user/api-keys", json={"label": "perm-test"}, headers=user_headers
        )
        key_id = key_resp.json()["id"]

        resp = client.post(
            "/api/user/permission-requests",
            json={
                "api_key_id": key_id,
                "model_id": gpt4_model.id,
                "request_message": "I need GPT-4o for my project",
            },
            headers=user_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"
        assert data["model_display_name"] == "GPT-4o"

    def test_duplicate_pending_request_rejected(self, client, user_headers, gpt4_model):
        key_resp = client.post(
            "/api/user/api-keys", json={"label": "dup-test"}, headers=user_headers
        )
        key_id = key_resp.json()["id"]

        # First request
        client.post(
            "/api/user/permission-requests",
            json={"api_key_id": key_id, "model_id": gpt4_model.id},
            headers=user_headers,
        )

        # Duplicate request should fail
        resp = client.post(
            "/api/user/permission-requests",
            json={"api_key_id": key_id, "model_id": gpt4_model.id},
            headers=user_headers,
        )
        assert resp.status_code == 400

    def test_list_my_permission_requests(self, client, user_headers, gpt4_model):
        key_resp = client.post(
            "/api/user/api-keys", json={"label": "list-perm"}, headers=user_headers
        )
        key_id = key_resp.json()["id"]

        client.post(
            "/api/user/permission-requests",
            json={"api_key_id": key_id, "model_id": gpt4_model.id},
            headers=user_headers,
        )

        resp = client.get("/api/user/permission-requests", headers=user_headers)
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    def test_admin_approves_permission_request(
        self, client, user_headers, admin_headers, gpt4_model
    ):
        """Full flow: user requests, admin approves, permission is created."""
        # User creates API key and requests permission
        key_resp = client.post(
            "/api/user/api-keys", json={"label": "approve-test"}, headers=user_headers
        )
        key_id = key_resp.json()["id"]

        req_resp = client.post(
            "/api/user/permission-requests",
            json={"api_key_id": key_id, "model_id": gpt4_model.id},
            headers=user_headers,
        )
        req_id = req_resp.json()["id"]

        # Admin lists pending requests
        list_resp = client.get(
            "/api/admin/permission-requests?status_filter=pending",
            headers=admin_headers,
        )
        assert len(list_resp.json()) >= 1

        # Admin approves
        approve_resp = client.put(
            f"/api/admin/permission-requests/{req_id}",
            json={"status": "approved", "max_output_tokens": 4096},
            headers=admin_headers,
        )
        assert approve_resp.status_code == 200

        # Check the permission is now on the API key
        keys_resp = client.get("/api/user/api-keys", headers=user_headers)
        keys = keys_resp.json()
        target_key = [k for k in keys if k["id"] == key_id][0]
        assert len(target_key["permissions"]) == 1
        assert target_key["permissions"][0]["model_id"] == gpt4_model.id

    def test_admin_rejects_permission_request(
        self, client, user_headers, admin_headers, gpt4_model
    ):
        key_resp = client.post(
            "/api/user/api-keys", json={"label": "reject-test"}, headers=user_headers
        )
        key_id = key_resp.json()["id"]

        req_resp = client.post(
            "/api/user/permission-requests",
            json={"api_key_id": key_id, "model_id": gpt4_model.id},
            headers=user_headers,
        )
        req_id = req_resp.json()["id"]

        resp = client.put(
            f"/api/admin/permission-requests/{req_id}",
            json={"status": "rejected", "admin_message": "Not approved at this time"},
            headers=admin_headers,
        )
        assert resp.status_code == 200

        # Check the user's request is now rejected
        my_reqs = client.get(
            "/api/user/permission-requests", headers=user_headers
        ).json()
        rejected = [r for r in my_reqs if r["id"] == req_id][0]
        assert rejected["status"] == "rejected"
        assert rejected["admin_message"] == "Not approved at this time"


class TestAvailableModels:
    def test_list_available_models(self, client, user_headers, gpt4_model):
        resp = client.get("/api/user/models", headers=user_headers)
        assert resp.status_code == 200
        models = resp.json()
        assert len(models) >= 1
        assert any(m["model_id"] == "gpt-4o" for m in models)


class TestModelMultiplexing:
    def test_update_and_list_multiplexing_rules(
        self, client, user_headers, db, regular_user, gpt4_model, openai_provider
    ):
        key_resp = client.post(
            "/api/user/api-keys",
            json={"label": "mux-test"},
            headers=user_headers,
        )
        key_id = key_resp.json()["id"]

        fallback_model = LLMModel(
            provider_id=openai_provider.id,
            model_id="gpt-4o-mini",
            display_name="GPT-4o Mini",
        )
        db.add(fallback_model)
        db.commit()
        db.refresh(fallback_model)

        db.add_all(
            [
                ApiKeyModelPermission(api_key_id=key_id, model_id=gpt4_model.id),
                ApiKeyModelPermission(
                    api_key_id=key_id,
                    model_id=fallback_model.id,
                ),
            ]
        )
        db.commit()

        list_resp = client.get("/api/user/multiplexing", headers=user_headers)
        assert list_resp.status_code == 200
        routes = list_resp.json()
        primary_route = [
            route
            for route in routes
            if route["api_key_id"] == key_id
            and route["primary_model_id"] == gpt4_model.id
        ][0]
        assert primary_route["enabled"] is True
        assert primary_route["fallback_model_ids"] == []

        update_resp = client.put(
            f"/api/user/api-keys/{key_id}/multiplexing/{gpt4_model.id}",
            json={"enabled": True, "fallback_model_ids": [fallback_model.id]},
            headers=user_headers,
        )
        assert update_resp.status_code == 200
        data = update_resp.json()
        assert data["fallback_model_ids"] == [fallback_model.id]
        assert data["fallback_models"][0]["model_id"] == "gpt-4o-mini"

    def test_fallback_must_be_accessible_to_api_key(
        self, client, user_headers, db, gpt4_model, openai_provider
    ):
        key_resp = client.post(
            "/api/user/api-keys",
            json={"label": "mux-reject"},
            headers=user_headers,
        )
        key_id = key_resp.json()["id"]

        inaccessible_model = LLMModel(
            provider_id=openai_provider.id,
            model_id="gpt-4.1",
            display_name="GPT-4.1",
        )
        db.add(inaccessible_model)
        db.commit()
        db.refresh(inaccessible_model)
        db.add(ApiKeyModelPermission(api_key_id=key_id, model_id=gpt4_model.id))
        db.commit()

        resp = client.put(
            f"/api/user/api-keys/{key_id}/multiplexing/{gpt4_model.id}",
            json={"enabled": True, "fallback_model_ids": [inaccessible_model.id]},
            headers=user_headers,
        )
        assert resp.status_code == 400
