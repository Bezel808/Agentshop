"""
Qwen Backend (支持 VL 模型)
"""

from typing import List
from aces.llm_backends.openai_backend import OpenAIBackend


class QwenBackend(OpenAIBackend):
    """Qwen LLM backend，支持 VL 模型"""
    
    def __init__(
        self,
        model: str = "qwen-turbo",
        api_key: str = None,
        temperature: float = 1.0,
        max_tokens: int = None,
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
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    
    def _convert_messages(self, messages: List) -> list:
        """
        转换消息格式，处理图像内容。
        
        Qwen VL 支持 OpenAI 的多模态格式。
        """
        openai_messages = []
        
        for msg in messages:
            if isinstance(msg.content, str):
                # 检查是否是 data URL（图像）
                if msg.content.startswith("data:image/"):
                    # 这是图像 data URL，转换为多模态格式
                    openai_messages.append({
                        "role": msg.role,
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": msg.content}
                            }
                        ]
                    })
                else:
                    # 普通文本
                    openai_messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            else:
                # 其他类型，使用父类处理
                openai_messages.append({
                    "role": msg.role,
                    "content": str(msg.content)
                })
        
        return openai_messages
