"""
Google Gemini provider adapter.
Uses the Gemini REST API (generativelanguage.googleapis.com) via httpx.
Supports text and vision (image) inputs.
"""

import httpx
from typing import Optional

from app.llm_providers import LLMProviderBase
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


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
        if generation_config:
            payload["generationConfig"] = generation_config

        url = f"{self.base_url}/v1beta/models/{model_id}:generateContent"
        params = {"key": self.api_key}

        logger.info(f"Gemini request: model={model_id}")

        resp = httpx.post(
            url,
            json=payload,
            params=params,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        # Parse the response
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini returned no candidates")

        content_parts = candidates[0].get("content", {}).get("parts", [])
        content = "".join(p.get("text", "") for p in content_parts)

        # Token usage
        usage_meta = data.get("usageMetadata", {})
        usage = {
            "prompt_tokens": usage_meta.get("promptTokenCount"),
            "completion_tokens": usage_meta.get("candidatesTokenCount"),
            "total_tokens": usage_meta.get("totalTokenCount"),
        }

        return {"content": content, "usage": usage}
