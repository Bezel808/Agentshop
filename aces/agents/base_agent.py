"""
Base Agent Implementation

This module provides a concrete implementation of the Agent interface
that composes the three key slots: LLM, Perception, and Tools.
"""

from typing import List, Optional, Dict, Any
import logging

from aces.core.protocols import (
    Agent,
    LLMBackend,
    PerceptionMode,
    Tool,
    Observation,
    Action,
    Message,
    AgentState,
    ToolResult,
)


logger = logging.getLogger(__name__)


class ComposableAgent(Agent):
    """
    A modular agent composed of pluggable components.
    
    This is the "Dependency Injection Container" that wires together:
    - An LLM backend (the brain)
    - A perception mode (the sensor)
    - A toolset (the capabilities)
    
    Usage:
        agent = ComposableAgent(
            llm=OpenAIBackend(model="gpt-4o"),
            perception=VisualPerception(),
            tools=[SearchTool(), CartTool()],
            system_prompt="You are a helpful shopping assistant."
        )
    """
    
    def __init__(
        self,
        llm: LLMBackend,
        perception: PerceptionMode,
        tools: List[Tool],
        system_prompt: Optional[str] = None,
        max_history: int = 50,
    ):
        """
        Initialize the agent.
        
        Args:
            llm: The LLM backend to use for decision making
            perception: The perception mode for processing observations
            tools: List of tools available to the agent
            system_prompt: Optional system prompt to guide behavior
            max_history: Maximum number of messages to keep in history
        """
        self.llm = llm
        self.perception = perception
        self.tools = {tool.get_schema().name: tool for tool in tools}
        self.system_prompt = system_prompt
        self.max_history = max_history
        
        # Agent state
        self.message_history: List[Message] = []
        self.observations: List[Observation] = []
        self.step_count = 0
        
        # Initialize with system prompt
        if system_prompt:
            self.message_history.append(
                Message(role="system", content=system_prompt)
            )
        
        logger.info(
            f"Initialized agent with LLM={llm.model_name}, "
            f"Perception={perception.get_modality()}, "
            f"Tools={list(self.tools.keys())}"
        )
    
    def act(self, observation: Observation) -> Action:
        """
        Decide on an action given an observation.
        
        This is the core decision-making loop:
        1. Validate observation matches perception mode
        2. Convert observation to message
        3. Query LLM with tools
        4. Parse LLM response into action
        """
        # Validate observation
        if not self.perception.validate_observation(observation):
            raise ValueError(
                f"Observation modality mismatch: expected {self.perception.get_modality()}, "
                f"got {observation.modality}"
            )
        
        # Store observation
        self.observations.append(observation)
        
        # Convert observation to message
        obs_message = self._observation_to_message(observation)
        self.message_history.append(obs_message)
        
        # Trim history if needed
        self._trim_history()
        
        # Get tool schemas for LLM
        tool_schemas = [tool.get_schema() for tool in self.tools.values()]
        
        # Query LLM
        response = self.llm.generate(
            messages=self.message_history,
            tools=tool_schemas,
        )
        
        # Store LLM response
        self.message_history.append(response)
        
        # Parse response into action
        action = self._parse_action(response)
        
        self.step_count += 1
        
        return action
    
    async def aact(self, observation: Observation) -> Action:
        """Async version of act (implementation similar to act)."""
        # For brevity, implementation omitted but would mirror act()
        # with async LLM calls
        raise NotImplementedError("Async act not yet implemented")
    
    def reset(self) -> None:
        """Reset agent to initial state."""
        self.message_history = []
        self.observations = []
        self.step_count = 0
        
        # Re-add system prompt
        if self.system_prompt:
            self.message_history.append(
                Message(role="system", content=self.system_prompt)
            )
        
        logger.info("Agent reset")
    
    def get_state(self) -> AgentState:
        """Get current agent state for checkpointing."""
        return AgentState(
            message_history=self.message_history.copy(),
            observations=self.observations.copy(),
            step_count=self.step_count,
            metadata={
                "llm": self.llm.model_name,
                "perception": self.perception.get_modality(),
                "tools": list(self.tools.keys()),
            }
        )
    
    def load_state(self, state: AgentState) -> None:
        """Restore agent from checkpoint."""
        self.message_history = state.message_history
        self.observations = state.observations
        self.step_count = state.step_count
        
        logger.info(f"Loaded state at step {self.step_count}")
    
    # ========================================================================
    # Private Helper Methods
    # ========================================================================
    
    def _observation_to_message(self, observation: Observation) -> Message:
        """
        Convert an observation to a message format the LLM understands.
        
        This is where perception mode matters:
        - Visual mode: Convert to image message (data URL)
        - Verbal mode: Convert to text message
        """
        if observation.modality == "visual":
            # For visual, observation.data is already a data URL string
            # Pass it directly so LLM backend can handle it
            content = observation.data
        elif observation.modality == "verbal":
            # For verbal, extract text content
            content = observation.data
        else:
            # Default: stringify
            content = str(observation.data)
        
        return Message(
            role="user",
            content=content,
            metadata={"timestamp": observation.timestamp}
        )
    
    def _parse_action(self, response: Message) -> Action:
        """
        Parse LLM response into an action.
        
        The response may contain:
        - A tool call (structured)
        - Plain text (we extract intent)
        """
        content = response.content
        
        # Check if response contains tool call
        if isinstance(content, dict) and "tool_call" in content:
            tool_call = content["tool_call"]
            
            # Validate tool exists
            tool_name = tool_call["name"]
            if tool_name not in self.tools:
                logger.warning(f"Unknown tool: {tool_name}. Available: {list(self.tools.keys())}")
                # Try fuzzy match
                for available_tool in self.tools.keys():
                    if available_tool in tool_name or tool_name in available_tool:
                        tool_name = available_tool
                        break
            
            return Action(
                tool_name=tool_name,
                parameters=tool_call["parameters"],
                reasoning=content.get("reasoning"),
            )
        
        # Otherwise, try to extract tool intent from text
        # (This is a fallback - ideally LLM returns structured tool calls)
        if isinstance(content, str):
            # Simple heuristic: look for tool names in response
            for tool_name in self.tools.keys():
                if tool_name.lower() in content.lower():
                    logger.info(f"Extracted tool '{tool_name}' from text response")
                    return Action(
                        tool_name=tool_name,
                        parameters={},  # Empty params, tool will handle
                        reasoning=content,
                    )
        
        # If no tool found, return a "no-op" action
        logger.warning(f"Could not parse action from response: {str(content)[:200]}")
        return Action(
            tool_name="noop",
            parameters={},
            reasoning="No valid action found",
        )
    
    def _trim_history(self) -> None:
        """Trim message history to stay within max_history limit."""
        if len(self.message_history) > self.max_history:
            # Keep system message and recent messages
            system_messages = [
                msg for msg in self.message_history if msg.role == "system"
            ]
            recent_messages = self.message_history[-self.max_history:]
            
            # Combine
            self.message_history = system_messages + recent_messages
            
            logger.debug(f"Trimmed history to {len(self.message_history)} messages")
