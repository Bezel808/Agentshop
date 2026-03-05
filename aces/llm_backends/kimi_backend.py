"""
Kimi (Moonshot 月之暗面) Backend

OpenAI 兼容 API，base_url 指向 Moonshot。
- 文本: moonshot-v1-8k, kimi-k2
- 视觉: moonshot-v1-8k-vision-preview, moonshot-v1-32k-vision-preview
"""

from typing import List
from aces.llm_backends.openai_backend import OpenAIBackend


class KimiBackend(OpenAIBackend):
    """Kimi/Moonshot LLM backend，支持文本与视觉模型"""

    def __init__(
        self,
        model: str = "moonshot-v1-8k",
        api_key: str = None,
        temperature: float = 0.7,
        max_tokens: int = None,
        base_url: str = "https://api.moonshot.cn/v1",
        **kwargs
    ):
        from openai import OpenAI

        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._kwargs = kwargs

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def _convert_messages(self, messages: List) -> list:
        """
        转换消息格式，处理图像 content（Kimi 视觉模型支持 data URL）。
        """
        openai_messages = []

        for msg in messages:
            if isinstance(msg.content, str):
                if msg.content.startswith("data:image/"):
                    openai_messages.append({
                        "role": msg.role,
                        "content": [
                            {"type": "image_url", "image_url": {"url": msg.content}}
                        ],
                    })
                else:
                    openai_messages.append({"role": msg.role, "content": msg.content})
            else:
                openai_messages.append({
                    "role": msg.role,
                    "content": str(msg.content),
                })

        return openai_messages
