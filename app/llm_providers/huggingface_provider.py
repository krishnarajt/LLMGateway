"""Hugging Face provider adapter."""

from app.llm_providers.openai_provider import OpenAIProvider


class HuggingFaceProvider(OpenAIProvider):
    """Hugging Face Inference Providers expose OpenAI-compatible chat."""

    def __init__(
        self, api_key: str, base_url: str = "https://router.huggingface.co/v1"
    ):
        super().__init__(api_key=api_key, base_url=base_url)
