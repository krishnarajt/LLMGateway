"""
Google Gemini provider adapter.
Uses the Gemini REST API (generativelanguage.googleapis.com) via httpx.
Supports text and vision (image) inputs.
"""

import httpx
import time
from typing import Optional

from app.llm_providers import LLMProviderBase
from app.llm_providers.response_utils import normalized_response
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

_TIMEOUT = httpx.Timeout(120.0, connect=10.0)
_RETRYABLE_STATUS_CODES = {500, 502, 503, 504}
_MAX_RETRIES = 2
_RETRY_BACKOFF_SECONDS = 0.5


def _key_fingerprint(api_key: str) -> str:
    if not api_key:
        return "none"
    if len(api_key) <= 8:
        return f"{api_key[:2]}***"
    return f"{api_key[:4]}***{api_key[-4:]}"


def _sanitize_extra_payload(extra: Optional[dict]) -> dict:
    """Drop internal gateway metadata before forwarding provider extras."""
    if not extra:
        return {}
    return {
        key: value
        for key, value in extra.items()
        if not str(key).startswith("_gateway_")
    }


class GeminiProvider(LLMProviderBase):
    def __init__(
        self, api_key: str, base_url: str = "https://generativelanguage.googleapis.com"
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def chat(
        self,
        model_id: str,
        system_prompt: Optional[str] = None,
        user_prompt: str = "",
        image_base64: Optional[str] = None,
        image_media_type: str = "image/png",
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        extra: Optional[dict] = None,
        include_thinking: bool = False,
    ) -> dict:
        # Gemini uses a different request structure than OpenAI
        # Endpoint: POST /v1beta/models/{model}:generateContent?key=API_KEY

        # Build user content parts
        parts = [{"text": user_prompt}]
        if image_base64:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": image_media_type,
                        "data": image_base64,
                    }
                }
            )

        contents = []
        # System instruction is passed at the top level in Gemini API
        if system_prompt:
            # For Gemini, system instructions go via the systemInstruction field
            pass  # handled below

        contents.append({"role": "user", "parts": parts})

        payload = {"contents": contents}

        # System instruction support
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        # Generation config
        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_output_tokens is not None:
            generation_config["maxOutputTokens"] = max_output_tokens
        if top_p is not None:
            generation_config["topP"] = top_p

        extra_payload = _sanitize_extra_payload(extra)
        extra_generation_config = extra_payload.pop(
            "generationConfig", extra_payload.pop("generation_config", None)
        )
        if extra_generation_config:
            generation_config.update(extra_generation_config)

        thinking_config = dict(
            generation_config.pop("thinkingConfig", None)
            or generation_config.pop("thinking_config", None)
            or {}
        )
        if include_thinking:
            thinking_config["includeThoughts"] = True
            # Gemma 4 docs require thinking to be enabled via thinkingLevel.
            if model_id.startswith("gemma-4-"):
                thinking_config.setdefault("thinkingLevel", "high")
        elif "includeThoughts" in thinking_config:
            thinking_config["includeThoughts"] = False
        if thinking_config:
            generation_config["thinkingConfig"] = thinking_config

        if generation_config:
            payload["generationConfig"] = generation_config
        if extra_payload:
            payload.update(extra_payload)

        url = f"{self.base_url}/v1beta/models/{model_id}:generateContent"
        params = {"key": self.api_key}
        trace_id = None
        if extra:
            trace_id = extra.get("_gateway_trace_id")
        key_fingerprint = _key_fingerprint(self.api_key)

        logger.info(
            "Gemini request starting: "
            f"trace_id={trace_id}, model={model_id}, "
            f"provider_key={key_fingerprint}, url={url}, "
            f"internal_retries={_MAX_RETRIES + 1}"
        )

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                logger.info(
                    "Gemini HTTP attempt: "
                    f"trace_id={trace_id}, model={model_id}, "
                    f"provider_key={key_fingerprint}, "
                    f"http_attempt={attempt + 1}/{_MAX_RETRIES + 1}, "
                    "scope=same_provider_key_retry"
                )
                resp = httpx.post(
                    url,
                    json=payload,
                    params=params,
                    timeout=_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
                logger.info(
                    "Gemini HTTP attempt succeeded: "
                    f"trace_id={trace_id}, model={model_id}, "
                    f"provider_key={key_fingerprint}, "
                    f"http_attempt={attempt + 1}/{_MAX_RETRIES + 1}, "
                    "scope=same_provider_key_retry"
                )
                break
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                status_code = exc.response.status_code
                if (
                    status_code not in _RETRYABLE_STATUS_CODES
                    or attempt >= _MAX_RETRIES
                ):
                    logger.warning(
                        "Gemini HTTP attempt failed without further same-key retries: "
                        f"trace_id={trace_id}, model={model_id}, "
                        f"provider_key={key_fingerprint}, status={status_code}, "
                        f"http_attempt={attempt + 1}/{_MAX_RETRIES + 1}, "
                        "scope=same_provider_key_retry"
                    )
                    raise
                logger.warning(
                    "Gemini transient HTTP error, retrying same provider key: "
                    f"trace_id={trace_id}, model={model_id}, "
                    f"provider_key={key_fingerprint}, status={status_code}, "
                    f"http_attempt={attempt + 1}/{_MAX_RETRIES + 1}, "
                    "scope=same_provider_key_retry"
                )
            except httpx.RequestError as exc:
                last_exc = exc
                if attempt >= _MAX_RETRIES:
                    logger.warning(
                        "Gemini transport error failed without further same-key retries: "
                        f"trace_id={trace_id}, model={model_id}, "
                        f"provider_key={key_fingerprint}, "
                        f"error={exc.__class__.__name__}, "
                        f"http_attempt={attempt + 1}/{_MAX_RETRIES + 1}, "
                        "scope=same_provider_key_retry"
                    )
                    raise
                logger.warning(
                    "Gemini transport error, retrying same provider key: "
                    f"trace_id={trace_id}, model={model_id}, "
                    f"provider_key={key_fingerprint}, "
                    f"error={exc.__class__.__name__}, "
                    f"http_attempt={attempt + 1}/{_MAX_RETRIES + 1}, "
                    "scope=same_provider_key_retry"
                )

            time.sleep(_RETRY_BACKOFF_SECONDS * (2**attempt))
        else:
            raise last_exc or RuntimeError("Gemini request failed without exception")

        # Parse the response
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini returned no candidates")

        content_parts = candidates[0].get("content", {}).get("parts", [])
        answer_parts = []
        thinking_parts = []
        for part in content_parts:
            text = part.get("text", "")
            if not text:
                continue
            if part.get("thought"):
                thinking_parts.append(text)
            else:
                answer_parts.append(text)
        content = "".join(answer_parts)

        # Token usage
        usage_meta = data.get("usageMetadata", {})
        usage = {
            "prompt_tokens": usage_meta.get("promptTokenCount"),
            "completion_tokens": usage_meta.get("candidatesTokenCount"),
            "total_tokens": usage_meta.get("totalTokenCount"),
        }

        return normalized_response(
            content=content,
            usage=usage,
            include_thinking=include_thinking,
            thinking_parts=thinking_parts,
        )
