"""
Core Protocol Definitions

This module defines the fundamental protocols (interfaces) that all
components must implement. These are the "contracts" that enable
modular composition.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Union
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# Data Transfer Objects (DTOs)
# ============================================================================

@dataclass
class Message:
    """A message in agent-environment communication."""
    role: str  # "user", "assistant", "system", "tool"
    content: Union[str, Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Observation:
    """What the agent perceives from the environment."""
    data: Any  # Could be image bytes, text, structured data
    modality: str  # "visual", "verbal", "multimodal"
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Action:
    """An action the agent wants to take."""
    tool_name: str
    parameters: Dict[str, Any]
    reasoning: Optional[str] = None


@dataclass
class ToolResult:
    """Result of executing a tool."""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentState:
    """Internal state of an agent (for checkpointing)."""
    message_history: List[Message]
    observations: List[Observation]
    step_count: int
    metadata: Dict[str, Any]


# ============================================================================
# The Agent Slot: LLM Backend Interface
# ============================================================================

class LLMBackend(ABC):
    """
    Abstract base class for LLM backends.
    
    This is the "Agent Slot" - it defines how we interact with any LLM
    without coupling to a specific provider.
    """
    
    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        tools: Optional[List["ToolSchema"]] = None,
        **kwargs
    ) -> Message:
        """
        Generate a response given conversation history.
        
        Args:
            messages: Conversation history
            tools: Available tools (for function calling)
            **kwargs: Provider-specific parameters
            
        Returns:
            Generated message (may contain tool calls)
        """
        pass
    
    @abstractmethod
    async def agenerate(
        self,
        messages: List[Message],
        tools: Optional[List["ToolSchema"]] = None,
        **kwargs
    ) -> Message:
        """Async version of generate."""
        pass
    
    @abstractmethod
    def count_tokens(self, messages: List[Message]) -> int:
        """Count tokens in message history."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        pass


# ============================================================================
# The Perception Slot: How Agents Sense the World
# ============================================================================

class PerceptionMode(ABC):
    """
    Abstract base class for perception strategies.
    
    This is the "Perception Slot" - it defines how raw environment state
    is transformed into agent observations.
    """
    
    @abstractmethod
    def encode(self, raw_state: Any) -> Observation:
        """
        Convert raw environment state into an observation.
        
        Args:
            raw_state: Raw state from environment (e.g., HTML, product dict)
            
        Returns:
            Observation in the mode's format (visual/verbal/etc)
        """
        pass
    
    @abstractmethod
    def get_modality(self) -> str:
        """Return the modality type (e.g., 'visual', 'verbal')."""
        pass
    
    @abstractmethod
    def validate_observation(self, obs: Observation) -> bool:
        """Check if observation conforms to this mode's requirements."""
        pass


# ============================================================================
# The Tool Slot: MCP-Compliant Tool Interface
# ============================================================================

@dataclass
class ToolSchema:
    """
    MCP-compliant tool definition.
    
    Based on Model Context Protocol specification:
    https://spec.modelcontextprotocol.io/
    """
    name: str
    description: str
    input_schema: Dict[str, Any]  # JSON Schema
    output_schema: Optional[Dict[str, Any]] = None


class Tool(ABC):
    """
    Abstract base class for MCP-compliant tools.
    
    This is the "Tool Slot" - it defines the interface for any capability
    an agent can use to interact with the environment.
    """
    
    @abstractmethod
    def get_schema(self) -> ToolSchema:
        """Return the tool's schema for LLM function calling."""
        pass
    
    @abstractmethod
    def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """
        Execute the tool with given parameters.
        
        Args:
            parameters: Tool parameters (validated against schema)
            
        Returns:
            Result of tool execution
        """
        pass
    
    @abstractmethod
    async def aexecute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Async version of execute."""
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters against the tool's schema."""
        pass


# ============================================================================
# The Environment Interface
# ============================================================================

class Environment(ABC):
    """
    Abstract base class for environments (marketplaces).
    
    The environment is what agents interact with through tools.
    It maintains state and provides observations.
    """
    
    @abstractmethod
    def reset(self) -> Observation:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation
        """
        pass
    
    @abstractmethod
    def step(self, action: Action) -> tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Execute an action and return the result.
        
        Args:
            action: Action to execute
            
        Returns:
            (observation, reward, done, info) tuple (RL-style)
        """
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current environment state (for logging/analysis)."""
        pass
    
    @abstractmethod
    def get_available_tools(self) -> List[Tool]:
        """Return list of tools available in this environment."""
        pass


# ============================================================================
# The Agent Interface
# ============================================================================

class Agent(ABC):
    """
    Abstract base class for agents.
    
    An agent is composed of:
    - An LLM backend (the brain)
    - A perception mode (the sensor)
    - A set of tools (the capabilities)
    """
    
    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """
        Decide on an action given an observation.
        
        Args:
            observation: Current observation from environment
            
        Returns:
            Action to take
        """
        pass
    
    @abstractmethod
    async def aact(self, observation: Observation) -> Action:
        """Async version of act."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset agent state (clear history, etc)."""
        pass
    
    @abstractmethod
    def get_state(self) -> AgentState:
        """Get agent state for checkpointing."""
        pass
    
    @abstractmethod
    def load_state(self, state: AgentState) -> None:
        """Restore agent from checkpoint."""
        pass


# ============================================================================
# The Experiment Interface
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    agent_config: Dict[str, Any]
    environment_config: Dict[str, Any]
    perception_mode: str
    max_steps: int
    random_seed: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExperimentResult:
    """Result of running an experiment."""
    success: bool
    steps_taken: int
    final_state: Dict[str, Any]
    trajectory: List[tuple[Observation, Action, ToolResult]]
    metrics: Dict[str, float]
    metadata: Dict[str, Any]


class Experiment(Protocol):
    """
    Protocol for experiment runners.
    
    Experiments orchestrate agent-environment interactions.
    """
    
    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment."""
        ...
    
    async def arun(self, config: ExperimentConfig) -> ExperimentResult:
        """Async version of run."""
        ...
