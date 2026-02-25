"""
OpenAI LLM Backend Implementation
"""

from typing import List, Optional
import logging

from aces.core.protocols import LLMBackend, Message, ToolSchema


logger = logging.getLogger(__name__)


class OpenAIBackend(LLMBackend):
    """
    OpenAI LLM backend implementation.
    
    This is a "slot implementation" for the LLM backend interface.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize OpenAI backend.
        
        Args:
            model: Model identifier (e.g., "gpt-4o", "gpt-4o-mini")
            api_key: OpenAI API key (or uses OPENAI_API_KEY env var)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
        """
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._kwargs = kwargs
        
        # Lazy import to avoid dependency if not used
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "OpenAI backend requires 'openai' package. "
                "Install with: pip install openai"
            )
        
        logger.info(f"Initialized OpenAI backend with model={model}")
    
    def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSchema]] = None,
        **kwargs
    ) -> Message:
        """Generate response using OpenAI API."""
        # Convert our Message format to OpenAI format
        openai_messages = self._convert_messages(messages)
        
        # Convert tool schemas to OpenAI function format
        openai_tools = None
        if tools:
            openai_tools = self._convert_tools(tools)
        
        # Merge kwargs
        call_kwargs = {
            "model": self._model,
            "messages": openai_messages,
            "temperature": self._temperature,
            **self._kwargs,
            **kwargs,
        }
        
        if self._max_tokens:
            call_kwargs["max_tokens"] = self._max_tokens
        
        if openai_tools:
            call_kwargs["tools"] = openai_tools
            call_kwargs["tool_choice"] = "auto"
        
        # Make API call
        response = self._client.chat.completions.create(**call_kwargs)
        
        # Convert response back to our Message format
        return self._convert_response(response)
    
    async def agenerate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSchema]] = None,
        **kwargs
    ) -> Message:
        """Async version of generate."""
        # Would use AsyncOpenAI client
        raise NotImplementedError("Async generate not yet implemented")
    
    def count_tokens(self, messages: List[Message]) -> int:
        """Count tokens using tiktoken."""
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self._model)
            
            # Rough approximation: stringify and encode
            text = " ".join(
                str(msg.content) for msg in messages
            )
            return len(encoding.encode(text))
        except ImportError:
            logger.warning("tiktoken not available, returning rough estimate")
            # Rough estimate: 1 token ≈ 4 characters
            text = " ".join(str(msg.content) for msg in messages)
            return len(text) // 4
    
    @property
    def model_name(self) -> str:
        """Return model identifier."""
        return self._model
    
    # ========================================================================
    # Private Helper Methods
    # ========================================================================
    
    def _convert_messages(self, messages: List[Message]) -> list:
        """Convert our Message format to OpenAI format."""
        openai_messages = []
        
        for msg in messages:
            # Handle different content types
            if isinstance(msg.content, dict):
                # Multimodal content (e.g., image)
                if msg.content.get("type") == "image":
                    openai_messages.append({
                        "role": msg.role,
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": msg.content["image_data"],
                                    "detail": "high",
                                }
                            }
                        ]
                    })
                else:
                    # Other structured content
                    openai_messages.append({
                        "role": msg.role,
                        "content": str(msg.content),
                    })
            else:
                # Text content
                openai_messages.append({
                    "role": msg.role,
                    "content": str(msg.content),
                })
        
        return openai_messages
    
    def _convert_tools(self, tools: List[ToolSchema]) -> list:
        """Convert tool schemas to OpenAI function format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                }
            }
            for tool in tools
        ]
    
    def _convert_response(self, response) -> Message:
        """Convert OpenAI response to our Message format."""
        choice = response.choices[0]
        message = choice.message
        
        # Check if there are tool calls
        if message.tool_calls:
            tool_call = message.tool_calls[0]  # Take first one
            
            # Parse arguments safely
            import json
            try:
                parameters = json.loads(tool_call.function.arguments)
            except:
                parameters = {}
            
            content = {
                "tool_call": {
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "parameters": parameters,
                },
                "reasoning": message.content or "",
            }
        else:
            content = message.content or ""
        
        return Message(
            role="assistant",
            content=content,
            metadata={
                "finish_reason": choice.finish_reason,
                "model": response.model,
                "has_tool_call": message.tool_calls is not None,
            }
        )
