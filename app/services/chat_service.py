"""
Chat service — the core logic for the /chat endpoint.

Flow:
  1. Validate the gateway API key (from X-API-Key header)
  2. Resolve the requested model from the config
  3. Check the API key has permission for that model
  4. Get the provider and a decrypted API key for it
  5. Instantiate the right provider adapter
  6. Call the LLM and return the response
"""

import hashlib
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.db.models import (
    GatewayApiKey, ApiKeyModelPermission, LLMModel, Provider,
    ProviderApiKey,
)
from app.common.schemas import ChatConfig
from app.llm_providers.openai_provider import OpenAIProvider
from app.llm_providers.gemini_provider import GeminiProvider
from app.llm_providers.ollama_provider import OllamaProvider
from app.utils.encryption import decrypt_value
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Registry of provider_type -> adapter class
_PROVIDER_REGISTRY = {
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
}


class ChatServiceError(Exception):
    """Raised when something goes wrong in the chat service."""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def resolve_api_key(db: Session, raw_key: str) -> GatewayApiKey:
    """
    Validate a raw gateway API key by hashing it and looking up the hash.
    Returns the GatewayApiKey ORM object.
    """
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    api_key = (
        db.query(GatewayApiKey)
        .filter(GatewayApiKey.key_hash == key_hash, GatewayApiKey.is_active == True)
        .first()
    )
    if not api_key:
        raise ChatServiceError("Invalid or inactive API key", status_code=401)

    # Update last_used_at
    api_key.last_used_at = datetime.now(timezone.utc)
    db.commit()

    return api_key


def resolve_model(db: Session, model_id_str: str) -> LLMModel:
    """
    Resolve a model_id string (e.g. "gpt-4o") to the LLMModel ORM object.
    Searches across all active providers.
    """
    model = (
        db.query(LLMModel)
        .join(Provider)
        .filter(
            LLMModel.model_id == model_id_str,
            LLMModel.is_active == True,
            Provider.is_active == True,
        )
        .first()
    )
    if not model:
        raise ChatServiceError(f"Model '{model_id_str}' not found or inactive", status_code=404)
    return model


def check_permission(db: Session, api_key: GatewayApiKey, model: LLMModel) -> ApiKeyModelPermission:
    """
    Check the API key has permission to use the given model.
    Returns the permission row (which may contain token limits).
    """
    perm = (
        db.query(ApiKeyModelPermission)
        .filter(
            ApiKeyModelPermission.api_key_id == api_key.id,
            ApiKeyModelPermission.model_id == model.id,
            ApiKeyModelPermission.is_active == True,
        )
        .first()
    )
    if not perm:
        raise ChatServiceError(
            f"API key '{api_key.key_prefix}' does not have permission to use model '{model.model_id}'",
            status_code=403,
        )
    return perm


def get_provider_key(db: Session, provider: Provider) -> str:
    """
    Get a decrypted API key for the provider.
    Picks the first active key. In the future this could do round-robin or load balancing.
    """
    key_row = (
        db.query(ProviderApiKey)
        .filter(ProviderApiKey.provider_id == provider.id, ProviderApiKey.is_active == True)
        .first()
    )
    if not key_row:
        # For Ollama, API key may not be needed
        if provider.provider_type == "ollama":
            return ""
        raise ChatServiceError(
            f"No active API key configured for provider '{provider.name}'",
            status_code=500,
        )
    return decrypt_value(key_row.encrypted_key)


def get_provider_adapter(provider: Provider, api_key: str):
    """Instantiate the appropriate provider adapter."""
    adapter_cls = _PROVIDER_REGISTRY.get(provider.provider_type)
    if not adapter_cls:
        raise ChatServiceError(
            f"Unsupported provider type: '{provider.provider_type}'",
            status_code=500,
        )
    base_url = provider.base_url or ""
    return adapter_cls(api_key=api_key, base_url=base_url)


def execute_chat(
    db: Session,
    raw_api_key: str,
    system_prompt: Optional[str],
    user_prompt: str,
    image_base64: Optional[str],
    image_media_type: str,
    config: ChatConfig,
) -> dict:
    """
    Full chat execution pipeline:
      1. Validate API key
      2. Resolve model
      3. Check permissions (and apply token limits)
      4. Get provider adapter
      5. Call LLM
      6. Return response
    """
    # Step 1: Validate gateway API key
    api_key_obj = resolve_api_key(db, raw_api_key)

    # Step 2: Resolve model
    model = resolve_model(db, config.model)
    provider = model.provider

    # Step 3: Check permission
    perm = check_permission(db, api_key_obj, model)

    # Apply permission-level token limits (the stricter of permission vs request)
    effective_max_output = config.max_output_tokens
    if perm.max_output_tokens is not None:
        if effective_max_output is None or effective_max_output > perm.max_output_tokens:
            effective_max_output = perm.max_output_tokens

    # Step 4: Get provider adapter
    provider_api_key = get_provider_key(db, provider)
    adapter = get_provider_adapter(provider, provider_api_key)

    # Step 5: Call LLM
    logger.info(
        f"Chat request: user_key={api_key_obj.key_prefix}, "
        f"model={model.model_id}, provider={provider.name}"
    )
    result = adapter.chat(
        model_id=model.model_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_base64=image_base64,
        image_media_type=image_media_type,
        temperature=config.temperature,
        max_output_tokens=effective_max_output,
        top_p=config.top_p,
        extra=config.extra,
    )

    # Step 6: Return enriched response
    return {
        "content": result["content"],
        "model": model.model_id,
        "provider": provider.name,
        "usage": result.get("usage"),
    }
