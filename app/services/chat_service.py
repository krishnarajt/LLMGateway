"""
Chat service — the core logic for the /chat endpoint.

Flow:
  1. Validate the gateway API key (from X-API-Key header)
  2. Resolve the requested model from the config
  3. Check the API key has permission for that model
  4. Try all provider API keys for that model's provider
  5. If those fail, try the configured model fallback chain
  6. Return the first successful LLM response
"""

import hashlib
from datetime import datetime, timezone
from typing import Optional

import httpx
from sqlalchemy.orm import Session

from app.common import constants
from app.db.models import (
    GatewayApiKey,
    ApiKeyModelPermission,
    LLMModel,
    ModelMultiplexRule,
    Provider,
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


class ProviderAttemptFailure(Exception):
    """Internal marker for a model attempt after all provider keys fail."""

    def __init__(self, model: LLMModel, failures: list[str], status_code: int = 502):
        self.model = model
        self.failures = failures
        self.status_code = status_code
        super().__init__("; ".join(failures))


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
        raise ChatServiceError(
            f"Model '{model_id_str}' not found or inactive", status_code=404
        )
    return model


def check_permission(
    db: Session, api_key: GatewayApiKey, model: LLMModel
) -> ApiKeyModelPermission:
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


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def get_provider_keys(db: Session, provider: Provider) -> list[str]:
    """
    Get decrypted API keys for the provider, in retry order.
    DB-managed Env Var backed provider keys are tried first. Container env vars
    are only a last-resort fallback when no usable DB key rows exist.
    """
    key_rows = (
        db.query(ProviderApiKey)
        .filter(
            ProviderApiKey.provider_id == provider.id, ProviderApiKey.is_active == True
        )
        .order_by(ProviderApiKey.order_index, ProviderApiKey.id)
        .all()
    )

    db_keys = []
    for key_row in key_rows:
        if key_row.env_var and key_row.env_var.encrypted_value:
            db_keys.append(decrypt_value(key_row.env_var.encrypted_value))
        elif key_row.encrypted_key:
            # Backward compatibility for provider keys created before Env Var routing.
            db_keys.append(decrypt_value(key_row.encrypted_key))

    keys = _dedupe_preserve_order(db_keys)
    if not keys:
        keys = constants.get_provider_api_keys_from_env(
            provider_name=provider.name,
            provider_type=provider.provider_type,
        )

    # For Ollama, API key may not be needed.
    if provider.provider_type == "ollama":
        return keys or [""]

    if not keys:
        raise ChatServiceError(
            f"No active API key configured for provider '{provider.name}'",
            status_code=500,
        )
    return keys


def get_provider_key(db: Session, provider: Provider) -> str:
    """Backward-compatible helper returning the first provider key."""
    return get_provider_keys(db, provider)[0]


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


def _effective_max_output(config: ChatConfig, perm: ApiKeyModelPermission) -> int | None:
    effective_max_output = config.max_output_tokens
    if perm.max_output_tokens is not None:
        if (
            effective_max_output is None
            or effective_max_output > perm.max_output_tokens
        ):
            effective_max_output = perm.max_output_tokens
    return effective_max_output


def _failure_status_code(exc: Exception) -> int | None:
    if isinstance(exc, ChatServiceError):
        return exc.status_code
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    return None


def _failure_message(exc: Exception) -> str:
    if isinstance(exc, ChatServiceError):
        return exc.message
    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        method = exc.request.method if exc.request else "HTTP"
        url = str(exc.request.url) if exc.request else ""
        return f"{method} {url} returned {response.status_code}"
    return str(exc) or exc.__class__.__name__


def _provider_attempt_status(failures: list[tuple[int | None, str]]) -> int:
    status_codes = [status for status, _ in failures if status is not None]
    if status_codes and all(status == 429 for status in status_codes):
        return 429
    return 502


def _call_model_with_provider_keys(
    db: Session,
    api_key_obj: GatewayApiKey,
    model: LLMModel,
    perm: ApiKeyModelPermission,
    system_prompt: Optional[str],
    user_prompt: str,
    image_base64: Optional[str],
    image_media_type: str,
    config: ChatConfig,
) -> dict:
    provider = model.provider
    try:
        provider_keys = get_provider_keys(db, provider)
    except ChatServiceError as exc:
        raise ProviderAttemptFailure(
            model=model,
            failures=[_failure_message(exc)],
            status_code=exc.status_code,
        ) from exc

    effective_max_output = _effective_max_output(config, perm)
    failures: list[tuple[int | None, str]] = []

    for key_index, provider_api_key in enumerate(provider_keys, start=1):
        try:
            adapter = get_provider_adapter(provider, provider_api_key)
            logger.info(
                f"Chat request: user_key={api_key_obj.key_prefix}, "
                f"model={model.model_id}, provider={provider.name}, "
                f"provider_key_attempt={key_index}/{len(provider_keys)}"
            )
            return adapter.chat(
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
        except Exception as exc:
            status_code = _failure_status_code(exc)
            message = _failure_message(exc)
            failures.append((status_code, message))
            logger.warning(
                f"Provider attempt failed: user_key={api_key_obj.key_prefix}, "
                f"model={model.model_id}, provider={provider.name}, "
                f"provider_key_attempt={key_index}/{len(provider_keys)}, "
                f"error={message}"
            )

    failure_lines = [
        f"{provider.name}/{model.model_id} key#{index}: {message}"
        for index, (_, message) in enumerate(failures, start=1)
    ]
    raise ProviderAttemptFailure(
        model=model,
        failures=failure_lines,
        status_code=_provider_attempt_status(failures),
    )


def _configured_fallbacks(
    db: Session, api_key_obj: GatewayApiKey, primary_model: LLMModel
) -> list[tuple[LLMModel, ApiKeyModelPermission]]:
    rule = (
        db.query(ModelMultiplexRule)
        .filter(
            ModelMultiplexRule.api_key_id == api_key_obj.id,
            ModelMultiplexRule.primary_model_id == primary_model.id,
        )
        .first()
    )
    if not rule or not rule.is_enabled:
        return []

    fallbacks = []
    for fallback in rule.fallbacks:
        model = fallback.fallback_model
        if (
            not model
            or not model.is_active
            or not model.provider
            or not model.provider.is_active
        ):
            logger.warning(
                f"Skipping inactive fallback model for key={api_key_obj.key_prefix}, "
                f"primary_model={primary_model.model_id}"
            )
            continue

        perm = (
            db.query(ApiKeyModelPermission)
            .filter(
                ApiKeyModelPermission.api_key_id == api_key_obj.id,
                ApiKeyModelPermission.model_id == model.id,
                ApiKeyModelPermission.is_active == True,
            )
            .first()
        )
        if not perm:
            logger.warning(
                f"Skipping fallback without active permission: key={api_key_obj.key_prefix}, "
                f"fallback_model={model.model_id}"
            )
            continue
        fallbacks.append((model, perm))
    return fallbacks


def _final_failure_status(failures: list[ProviderAttemptFailure]) -> int:
    if failures and all(failure.status_code == 429 for failure in failures):
        return 429
    return 502


def _format_failures(requested_model: str, failures: list[ProviderAttemptFailure]) -> str:
    details = []
    for failure in failures:
        details.extend(failure.failures)
    detail_text = "; ".join(details) if details else "no provider attempts were made"
    return (
        f"All provider keys and model fallbacks failed for requested model "
        f"'{requested_model}'. Attempts: {detail_text}"
    )


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

    # Step 3: Check permission
    perm = check_permission(db, api_key_obj, model)

    failures = []
    attempts: list[tuple[LLMModel, ApiKeyModelPermission]] = [(model, perm)]
    attempts.extend(_configured_fallbacks(db, api_key_obj, model))

    for attempt_index, (attempt_model, attempt_perm) in enumerate(attempts):
        if attempt_index > 0:
            logger.info(
                f"Model multiplexing fallback: user_key={api_key_obj.key_prefix}, "
                f"requested_model={model.model_id}, fallback_model={attempt_model.model_id}"
            )

        try:
            result = _call_model_with_provider_keys(
                db=db,
                api_key_obj=api_key_obj,
                model=attempt_model,
                perm=attempt_perm,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_base64=image_base64,
                image_media_type=image_media_type,
                config=config,
            )
            return {
                "content": result["content"],
                "model": attempt_model.model_id,
                "provider": attempt_model.provider.name,
                "usage": result.get("usage"),
            }
        except ProviderAttemptFailure as failure:
            failures.append(failure)

    raise ChatServiceError(
        _format_failures(config.model, failures),
        status_code=_final_failure_status(failures),
    )
