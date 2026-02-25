"""
DeepSeek LLM Backend Implementation

DeepSeek is OpenAI-compatible, so we inherit from OpenAI backend.
"""

from aces.llm_backends.openai_backend import OpenAIBackend


class DeepSeekBackend(OpenAIBackend):
    """
    DeepSeek LLM backend (OpenAI-compatible).
    
    Simply sets the base_url to DeepSeek's API endpoint.
    """
    
    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str = None,
        temperature: float = 1.0,
        max_tokens: int = None,
        base_url: str = "https://api.deepseek.com",
        **kwargs
    ):
        """Initialize DeepSeek backend."""
        # Import OpenAI client
        from openai import OpenAI
        
        # Override client with custom base_url
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._kwargs = kwargs
        
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
