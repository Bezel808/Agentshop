"""
LLM backend factory for centralized creation logic.
"""

from __future__ import annotations

from typing import Any, Dict

from aces.llm_backends.openai_backend import OpenAIBackend
from aces.llm_backends.deepseek_backend import DeepSeekBackend
from aces.llm_backends.qwen_backend import QwenBackend


class LLMBackendFactory:
    @staticmethod
    def create(config: Dict[str, Any]):
        backend = (config.get("backend") or "deepseek").lower()
        model = config.get("model")
        api_key = config.get("api_key")
        temperature = config.get("temperature", 1.0)
        max_tokens = config.get("max_tokens")
        base_url = config.get("base_url")

        if backend == "openai":
            return OpenAIBackend(
                model=model or "gpt-4o-mini",
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if backend == "deepseek":
            return DeepSeekBackend(
                model=model or "deepseek-chat",
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=base_url or "https://api.deepseek.com",
            )
        if backend == "qwen":
            return QwenBackend(
                model=model or "qwen-turbo",
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        raise ValueError(f"Unsupported LLM backend: {backend}")
