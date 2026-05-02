"""Tests for the chat service and encryption utilities."""

import hashlib
import pytest
from unittest.mock import patch, MagicMock

from app.db.models import (
    GatewayApiKey,
    ApiKeyModelPermission,
    LLMModel,
    ModelMultiplexFallback,
    ModelMultiplexRule,
    ProviderApiKey,
    EnvironmentVariable,
    Provider,
)
from app.services.chat_service import (
    resolve_api_key,
    resolve_model,
    check_permission,
    get_provider_key,
    get_provider_adapter,
    ChatServiceError,
    execute_chat,
)
from app.common.schemas import ChatConfig
from app.llm_providers.gemini_provider import GeminiProvider
from app.llm_providers.groq_provider import GroqProvider
from app.llm_providers.huggingface_provider import HuggingFaceProvider
from app.llm_providers.openai_provider import OpenAIProvider
from app.utils.encryption import encrypt_value, decrypt_value


class FakeLLMResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


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

    def test_get_env_var_backed_provider_keys(self, db, openai_provider):
        env_var = EnvironmentVariable(
            key="OPENAI_PRIMARY",
            encrypted_value=encrypt_value("sk-env-primary"),
            is_secret=True,
        )
        db.add(env_var)
        db.commit()
        db.refresh(env_var)

        db.add(
            ProviderApiKey(
                provider_id=openai_provider.id,
                label="primary",
                env_var_id=env_var.id,
                order_index=0,
            )
        )
        db.commit()

        assert get_provider_key(db, openai_provider) == "sk-env-primary"

    def test_container_env_used_only_when_no_db_keys(
        self, db, openai_provider, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEYS", "sk-container-1,sk-container-2")

        assert get_provider_key(db, openai_provider) == "sk-container-1"

    def test_provider_env_aliases(self, db, groq_provider, huggingface_provider, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-container")
        monkeypatch.setenv("HF_TOKEN", "hf-container")

        assert get_provider_key(db, groq_provider) == "gsk-container"
        assert get_provider_key(db, huggingface_provider) == "hf-container"


class TestProviderAdapters:
    def test_registry_supports_groq_and_huggingface_defaults(self):
        groq = Provider(
            name="groq",
            display_name="Groq",
            provider_type="groq",
        )
        huggingface = Provider(
            name="huggingface",
            display_name="Hugging Face",
            provider_type="huggingface",
        )

        groq_adapter = get_provider_adapter(groq, "gsk-test")
        huggingface_adapter = get_provider_adapter(huggingface, "hf-test")

        assert isinstance(groq_adapter, GroqProvider)
        assert groq_adapter.base_url == "https://api.groq.com/openai/v1"
        assert isinstance(huggingface_adapter, HuggingFaceProvider)
        assert huggingface_adapter.base_url == "https://router.huggingface.co/v1"

    def test_openai_compatible_adapter_strips_thinking_by_default(self):
        provider = OpenAIProvider(api_key="sk-test", base_url="https://example.test/v1")
        response = FakeLLMResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "<think>private chain</think>\nFinal answer.",
                            "reasoning_content": "separate reasoning",
                        }
                    }
                ],
                "usage": {"total_tokens": 7},
            }
        )

        with patch("app.llm_providers.openai_provider.httpx.post") as mock_post:
            mock_post.return_value = response
            result = provider.chat(model_id="test-model", user_prompt="Hello")

        assert result == {
            "content": "Final answer.",
            "usage": {"total_tokens": 7},
        }

    def test_openai_compatible_adapter_returns_thinking_when_requested(self):
        provider = OpenAIProvider(api_key="sk-test", base_url="https://example.test/v1")
        response = FakeLLMResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "<think>inline reasoning</think>\nFinal answer.",
                            "reasoning_content": "separate reasoning",
                        }
                    }
                ]
            }
        )

        with patch("app.llm_providers.openai_provider.httpx.post") as mock_post:
            mock_post.return_value = response
            result = provider.chat(
                model_id="test-model",
                user_prompt="Hello",
                include_thinking=True,
            )

        assert result["content"] == "Final answer."
        assert "separate reasoning" in result["thinking"]
        assert "inline reasoning" in result["thinking"]

    def test_groq_controls_reasoning_payload(self):
        provider = GroqProvider(api_key="gsk-test")
        response = FakeLLMResponse({"choices": [{"message": {"content": "Done"}}]})

        with patch("app.llm_providers.openai_provider.httpx.post") as mock_post:
            mock_post.return_value = response
            provider.chat(
                model_id="llama-3.3-70b-versatile",
                user_prompt="Hello",
                extra={"reasoning_format": "raw"},
            )

        payload = mock_post.call_args.kwargs["json"]
        assert "include_reasoning" not in payload
        assert payload["reasoning_format"] == "hidden"

        with patch("app.llm_providers.openai_provider.httpx.post") as mock_post:
            mock_post.return_value = response
            provider.chat(
                model_id="llama-3.3-70b-versatile",
                user_prompt="Hello",
                include_thinking=True,
            )

        payload = mock_post.call_args.kwargs["json"]
        assert payload["include_reasoning"] is True

    def test_gemini_skips_thought_parts_by_default(self):
        provider = GeminiProvider(api_key="test-key")
        response = FakeLLMResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "private summary", "thought": True},
                                {"text": "Final answer."},
                            ]
                        }
                    }
                ],
                "usageMetadata": {"totalTokenCount": 9},
            }
        )

        with patch("app.llm_providers.gemini_provider.httpx.post") as mock_post:
            mock_post.return_value = response
            result = provider.chat(model_id="gemini-test", user_prompt="Hello")

        assert result == {
            "content": "Final answer.",
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": 9,
            },
        }

    def test_gemini_returns_thought_parts_when_requested(self):
        provider = GeminiProvider(api_key="test-key")
        response = FakeLLMResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "private summary", "thought": True},
                                {"text": "Final answer."},
                            ]
                        }
                    }
                ]
            }
        )

        with patch("app.llm_providers.gemini_provider.httpx.post") as mock_post:
            mock_post.return_value = response
            result = provider.chat(
                model_id="gemini-test",
                user_prompt="Hello",
                include_thinking=True,
            )

        payload = mock_post.call_args.kwargs["json"]
        assert payload["generationConfig"]["thinkingConfig"]["includeThoughts"] is True
        assert result["content"] == "Final answer."
        assert result["thinking"] == "private summary"


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
        assert "thinking" not in result
        assert mock_provider.chat.call_args.kwargs["include_thinking"] is False

    def test_full_chat_flow_can_return_thinking_when_requested(
        self, db, regular_user, gpt4_model, openai_provider, openai_api_key
    ):
        raw_key = "gw-thinking-chat-test"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        gw_key = GatewayApiKey(
            user_id=regular_user.id,
            key_hash=key_hash,
            key_prefix="gw-think***",
            label="thinking-chat",
        )
        db.add(gw_key)
        db.commit()
        db.refresh(gw_key)

        db.add(ApiKeyModelPermission(api_key_id=gw_key.id, model_id=gpt4_model.id))
        db.commit()

        mock_result = {
            "content": "Final answer only.",
            "thinking": "Reasoning summary.",
            "usage": {"total_tokens": 42},
        }
        with patch("app.services.chat_service.get_provider_adapter") as mock_adapter:
            mock_provider = MagicMock()
            mock_provider.chat.return_value = mock_result
            mock_adapter.return_value = mock_provider

            result = execute_chat(
                db=db,
                raw_api_key=raw_key,
                system_prompt=None,
                user_prompt="Hello!",
                image_base64=None,
                image_media_type="image/png",
                config=ChatConfig(model="gpt-4o", thinking=True),
            )

        assert result["content"] == "Final answer only."
        assert result["thinking"] == "Reasoning summary."
        assert mock_provider.chat.call_args.kwargs["include_thinking"] is True

    def test_provider_key_rotation_happens_before_model_fallback(
        self, db, regular_user, gpt4_model, openai_provider, openai_api_key
    ):
        raw_key = "gw-provider-key-rotation"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        gw_key = GatewayApiKey(
            user_id=regular_user.id,
            key_hash=key_hash,
            key_prefix="gw-rot***",
            label="rotation",
        )
        db.add(gw_key)
        db.commit()
        db.refresh(gw_key)

        db.add(ApiKeyModelPermission(api_key_id=gw_key.id, model_id=gpt4_model.id))
        db.add(
            ProviderApiKey(
                provider_id=openai_provider.id,
                label="second-key",
                encrypted_key=encrypt_value("sk-second-fake-key"),
            )
        )
        db.commit()

        first_provider = MagicMock()
        first_provider.chat.side_effect = RuntimeError("rate limited")
        second_provider = MagicMock()
        second_provider.chat.return_value = {
            "content": "Recovered with second provider key.",
            "usage": {"total_tokens": 12},
        }

        with patch("app.services.chat_service.get_provider_adapter") as mock_adapter:
            mock_adapter.side_effect = [first_provider, second_provider]

            result = execute_chat(
                db=db,
                raw_api_key=raw_key,
                system_prompt=None,
                user_prompt="Hello!",
                image_base64=None,
                image_media_type="image/png",
                config=ChatConfig(model="gpt-4o"),
            )

        assert result["content"] == "Recovered with second provider key."
        assert result["model"] == "gpt-4o"
        assert mock_adapter.call_count == 2

    def test_model_fallback_after_all_primary_provider_keys_fail(
        self, db, regular_user, gpt4_model, openai_provider, openai_api_key
    ):
        raw_key = "gw-model-fallback"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        gw_key = GatewayApiKey(
            user_id=regular_user.id,
            key_hash=key_hash,
            key_prefix="gw-fall***",
            label="fallback",
        )
        fallback_model = LLMModel(
            provider_id=openai_provider.id,
            model_id="gpt-4o-mini",
            display_name="GPT-4o Mini",
        )
        db.add_all([gw_key, fallback_model])
        db.commit()
        db.refresh(gw_key)
        db.refresh(fallback_model)

        db.add_all(
            [
                ApiKeyModelPermission(
                    api_key_id=gw_key.id,
                    model_id=gpt4_model.id,
                ),
                ApiKeyModelPermission(
                    api_key_id=gw_key.id,
                    model_id=fallback_model.id,
                ),
            ]
        )
        db.flush()
        rule = ModelMultiplexRule(
            api_key_id=gw_key.id,
            primary_model_id=gpt4_model.id,
            is_enabled=True,
        )
        db.add(rule)
        db.flush()
        db.add(
            ModelMultiplexFallback(
                rule_id=rule.id,
                fallback_model_id=fallback_model.id,
                order_index=0,
            )
        )
        db.commit()

        primary_provider = MagicMock()
        primary_provider.chat.side_effect = RuntimeError("primary failed")
        fallback_provider = MagicMock()
        fallback_provider.chat.return_value = {
            "content": "Recovered with fallback model.",
            "usage": {"total_tokens": 18},
        }

        with patch("app.services.chat_service.get_provider_adapter") as mock_adapter:
            mock_adapter.side_effect = [primary_provider, fallback_provider]

            result = execute_chat(
                db=db,
                raw_api_key=raw_key,
                system_prompt=None,
                user_prompt="Hello!",
                image_base64=None,
                image_media_type="image/png",
                config=ChatConfig(model="gpt-4o"),
            )

        assert result["content"] == "Recovered with fallback model."
        assert result["model"] == "gpt-4o-mini"
        assert result["provider"] == "openai"
