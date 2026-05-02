"""
Abstract base class for all LLM provider adapters.
Each provider (OpenAI, Gemini, Ollama, etc.) implements this interface.
"""

from abc import ABC, abstractmethod
from typing import Optional


class LLMProviderBase(ABC):
    """Base class that every provider adapter must implement."""

    @abstractmethod
    def chat(
        self,
        model_id: str,
        system_prompt: Optional[str],
        user_prompt: str,
        image_base64: Optional[str] = None,
        image_media_type: str = "image/png",
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        extra: Optional[dict] = None,
        include_thinking: bool = False,
    ) -> dict:
        """
        Send a chat completion request and return the result.

        Returns a dict with at least:
          - content: str (the model's text response)
          - usage: dict | None (token usage info)
          - thinking: str | None (only when explicitly requested)
        """
        ...
