"""
OpenAI provider adapter.
Uses raw httpx calls to the OpenAI-compatible /v1/chat/completions endpoint.
This means it also works with any OpenAI-compatible API (Azure, local proxies, etc.).
"""

import httpx
from typing import Optional

from app.llm_providers import LLMProviderBase
from app.llm_providers.response_utils import (
    THINKING_FIELD_NAMES,
    normalized_response,
)
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Timeout for LLM calls — these can be slow
_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class OpenAIProvider(LLMProviderBase):
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _apply_thinking_options(self, payload: dict, include_thinking: bool) -> None:
        """Let provider subclasses request/disable separate reasoning output."""
        if include_thinking:
            return

        for key in ("include_reasoning", "include_thinking", "includeThoughts"):
            payload.pop(key, None)
        if payload.get("reasoning_format") in {"raw", "parsed"}:
            payload["reasoning_format"] = "hidden"

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
        messages = []

        # System message
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # User message — text only or multimodal (text + image)
        if image_base64:
            # Vision-style multimodal request
            user_content = [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_media_type};base64,{image_base64}"
                    },
                },
            ]
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": user_prompt})

        # Build the request payload
        payload = {"model": model_id, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        # Merge any extra provider-specific params
        if extra:
            payload.update(extra)
        self._apply_thinking_options(payload, include_thinking)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.info(f"OpenAI request: model={model_id}, base_url={self.base_url}")

        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        message = data["choices"][0].get("message", {})
        content = message.get("content", "")
        thinking_parts = [message.get(field) for field in THINKING_FIELD_NAMES]
        usage = data.get("usage")

        return normalized_response(
            content=content,
            usage=usage,
            include_thinking=include_thinking,
            thinking_parts=thinking_parts,
        )
