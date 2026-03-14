"""
Ollama provider adapter.
Uses the Ollama REST API (typically http://localhost:11434) via httpx.
Ollama uses an OpenAI-compatible /api/chat endpoint.
"""

import httpx
from typing import Optional

from app.llm_providers import LLMProviderBase
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Ollama can be slow for large models — generous timeout
_TIMEOUT = httpx.Timeout(300.0, connect=10.0)


class OllamaProvider(LLMProviderBase):
    def __init__(self, api_key: str = "", base_url: str = "http://localhost:11434"):
        # Ollama doesn't need an API key, but we accept one for interface consistency
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
        # Ollama /api/chat format
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_msg = {"role": "user", "content": user_prompt}
        # Ollama supports images via the "images" field (list of base64 strings)
        if image_base64:
            user_msg["images"] = [image_base64]
        messages.append(user_msg)

        payload = {
            "model": model_id,
            "messages": messages,
            "stream": False,  # We want the full response at once for the non-streaming endpoint
        }

        # Options
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_output_tokens is not None:
            options["num_predict"] = max_output_tokens
        if top_p is not None:
            options["top_p"] = top_p
        if options:
            payload["options"] = options

        logger.info(f"Ollama request: model={model_id}, base_url={self.base_url}")

        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        content = data.get("message", {}).get("content", "")

        # Ollama provides some usage info
        usage = {
            "prompt_tokens": data.get("prompt_eval_count"),
            "completion_tokens": data.get("eval_count"),
            "total_tokens": (data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0),
        }

        return {"content": content, "usage": usage}
