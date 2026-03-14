"""Tests for the chat service and encryption utilities."""

import hashlib
import pytest
from unittest.mock import patch, MagicMock

from app.db.models import GatewayApiKey, ApiKeyModelPermission
from app.services.chat_service import (
    resolve_api_key,
    resolve_model,
    check_permission,
    get_provider_key,
    ChatServiceError,
    execute_chat,
)
from app.common.schemas import ChatConfig
from app.utils.encryption import encrypt_value, decrypt_value


class TestEncryption:
    def test_encrypt_decrypt_roundtrip(self):
        original = "sk-my-super-secret-api-key"
        encrypted = encrypt_value(original)
        decrypted = decrypt_value(encrypted)
        assert decrypted == original
        assert encrypted != original  # Ensure it's actually encrypted

    def test_different_values_produce_different_ciphertexts(self):
        enc1 = encrypt_value("value1")
        enc2 = encrypt_value("value2")
        assert enc1 != enc2


class TestResolveApiKey:
    def test_valid_key(self, db, regular_user):
        raw_key = "gw-test-key-for-chat"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        gw_key = GatewayApiKey(
            user_id=regular_user.id,
            key_hash=key_hash,
            key_prefix="gw-test***",
            label="chat-test",
        )
        db.add(gw_key)
        db.commit()

        result = resolve_api_key(db, raw_key)
        assert result.id == gw_key.id

    def test_invalid_key(self, db):
        with pytest.raises(ChatServiceError, match="Invalid or inactive"):
            resolve_api_key(db, "gw-nonexistent-key")

    def test_inactive_key(self, db, regular_user):
        raw_key = "gw-inactive-key"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        gw_key = GatewayApiKey(
            user_id=regular_user.id,
            key_hash=key_hash,
            key_prefix="gw-inac***",
            label="inactive",
            is_active=False,
        )
        db.add(gw_key)
        db.commit()

        with pytest.raises(ChatServiceError, match="Invalid or inactive"):
            resolve_api_key(db, raw_key)


class TestResolveModel:
    def test_valid_model(self, db, gpt4_model):
        result = resolve_model(db, "gpt-4o")
        assert result.id == gpt4_model.id

    def test_nonexistent_model(self, db):
        with pytest.raises(ChatServiceError, match="not found"):
            resolve_model(db, "nonexistent-model")


class TestCheckPermission:
    def test_has_permission(self, db, regular_user, gpt4_model):
        raw_key = "gw-perm-check"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        gw_key = GatewayApiKey(
            user_id=regular_user.id,
            key_hash=key_hash,
            key_prefix="gw-perm***",
            label="perm-check",
        )
        db.add(gw_key)
        db.commit()
        db.refresh(gw_key)

        perm = ApiKeyModelPermission(
            api_key_id=gw_key.id,
            model_id=gpt4_model.id,
            max_output_tokens=4096,
        )
        db.add(perm)
        db.commit()

        result = check_permission(db, gw_key, gpt4_model)
        assert result.max_output_tokens == 4096

    def test_no_permission(self, db, regular_user, gpt4_model):
        raw_key = "gw-no-perm"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        gw_key = GatewayApiKey(
            user_id=regular_user.id,
            key_hash=key_hash,
            key_prefix="gw-nope***",
            label="no-perm",
        )
        db.add(gw_key)
        db.commit()
        db.refresh(gw_key)

        with pytest.raises(ChatServiceError, match="does not have permission"):
            check_permission(db, gw_key, gpt4_model)


class TestGetProviderKey:
    def test_get_key(self, db, openai_provider, openai_api_key):
        key = get_provider_key(db, openai_provider)
        assert key == "sk-test-fake-key-12345"

    def test_ollama_no_key_needed(self, db, ollama_provider):
        key = get_provider_key(db, ollama_provider)
        assert key == ""

    def test_missing_key(self, db, gemini_provider):
        with pytest.raises(ChatServiceError, match="No active API key"):
            get_provider_key(db, gemini_provider)


class TestExecuteChat:
    def test_full_chat_flow(
        self, db, regular_user, gpt4_model, openai_provider, openai_api_key
    ):
        """Full integration test of the chat flow (with mocked provider call)."""
        raw_key = "gw-full-chat-test"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        gw_key = GatewayApiKey(
            user_id=regular_user.id,
            key_hash=key_hash,
            key_prefix="gw-full***",
            label="full-chat",
        )
        db.add(gw_key)
        db.commit()
        db.refresh(gw_key)

        perm = ApiKeyModelPermission(
            api_key_id=gw_key.id,
            model_id=gpt4_model.id,
        )
        db.add(perm)
        db.commit()

        config = ChatConfig(model="gpt-4o")

        # Mock the actual HTTP call to OpenAI
        mock_result = {
            "content": "Hello! I'm a test response.",
            "usage": {"total_tokens": 42},
        }
        with patch("app.services.chat_service.get_provider_adapter") as mock_adapter:
            mock_provider = MagicMock()
            mock_provider.chat.return_value = mock_result
            mock_adapter.return_value = mock_provider

            result = execute_chat(
                db=db,
                raw_api_key=raw_key,
                system_prompt="You are helpful.",
                user_prompt="Hello!",
                image_base64=None,
                image_media_type="image/png",
                config=config,
            )

        assert result["content"] == "Hello! I'm a test response."
        assert result["model"] == "gpt-4o"
        assert result["provider"] == "openai"
