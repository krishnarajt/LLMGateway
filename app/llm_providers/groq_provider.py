"""Groq provider adapter."""

from app.llm_providers.openai_provider import OpenAIProvider


class GroqProvider(OpenAIProvider):
    """Groq chat completions use an OpenAI-compatible API surface."""

    def __init__(self, api_key: str, base_url: str = "https://api.groq.com/openai/v1"):
        super().__init__(api_key=api_key, base_url=base_url)

    def _apply_thinking_options(self, payload: dict, include_thinking: bool) -> None:
        if include_thinking:
            if "reasoning_format" not in payload:
                payload.setdefault("include_reasoning", True)
            return

        if "reasoning_format" in payload:
            payload.pop("include_reasoning", None)
            if payload["reasoning_format"] in {"raw", "parsed"}:
                payload["reasoning_format"] = "hidden"
        else:
            payload["include_reasoning"] = False
