"""
Experiment System Protocols

Defines interfaces for experiment orchestration, logging, and metrics.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# ============================================================================
# Trajectory Data Structures
# ============================================================================

class StepType(Enum):
    """Types of steps in agent trajectory."""
    OBSERVATION = "observation"
    THOUGHT = "thought"
    ACTION = "action"
    TOOL_RESULT = "tool_result"


@dataclass
class TrajectoryStep:
    """
    A single step in the agent's trajectory.
    
    This is the fundamental unit of logged data for analysis.
    """
    step_number: int
    timestamp: float
    step_type: StepType
    
    # Content (varies by type)
    content: Any
    
    # Modality tracking (crucial for perception research)
    input_modality: Optional[str] = None  # "visual", "verbal", "multimodal"
    output_modality: Optional[str] = None
    
    # Context
    agent_state: Optional[Dict[str, Any]] = None
    environment_state: Optional[Dict[str, Any]] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "step_number": self.step_number,
            "timestamp": self.timestamp,
            "step_type": self.step_type.value,
            "content": self.content,
            "input_modality": self.input_modality,
            "output_modality": self.output_modality,
            "agent_state": self.agent_state,
            "environment_state": self.environment_state,
            "metadata": self.metadata,
        }


@dataclass
class Trajectory:
    """
    Complete trajectory of an agent-environment interaction.
    
    This is what gets saved for each experimental trial.
    """
    experiment_id: str
    trial_number: int
    agent_config: Dict[str, Any]
    environment_config: Dict[str, Any]
    
    # The sequence of steps
    steps: List[TrajectoryStep] = field(default_factory=list)
    
    # Outcome
    success: bool = False
    final_action: Optional[Any] = None
    
    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        return self.end_time - self.start_time
    
    @property
    def num_steps(self) -> int:
        """Number of steps taken."""
        return len(self.steps)
    
    def add_step(self, step: TrajectoryStep) -> None:
        """Add a step to the trajectory."""
        self.steps.append(step)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment_id": self.experiment_id,
            "trial_number": self.trial_number,
            "agent_config": self.agent_config,
            "environment_config": self.environment_config,
            "steps": [step.to_dict() for step in self.steps],
            "success": self.success,
            "final_action": self.final_action,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "num_steps": self.num_steps,
            "metadata": self.metadata,
        }


# ============================================================================
# Metrics System
# ============================================================================

@dataclass
class MetricResult:
    """Result of a metric calculation."""
    metric_name: str
    value: Any
    unit: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricCalculator(ABC):
    """
    Abstract base class for metric calculators.
    
    Metrics are computed from trajectories after experiments complete.
    """
    
    @abstractmethod
    def calculate(self, trajectory: Trajectory) -> MetricResult:
        """
        Calculate metric from a trajectory.
        
        Args:
            trajectory: Complete trajectory data
            
        Returns:
            Calculated metric
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return metric name."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Return metric description."""
        pass


class MetricRegistry:
    """Registry for metric calculators."""
    
    def __init__(self):
        self._metrics: Dict[str, MetricCalculator] = {}
    
    def register(self, calculator: MetricCalculator) -> None:
        """Register a metric calculator."""
        name = calculator.get_name()
        self._metrics[name] = calculator
    
    def calculate_all(self, trajectory: Trajectory) -> List[MetricResult]:
        """Calculate all registered metrics."""
        return [
            calc.calculate(trajectory)
            for calc in self._metrics.values()
        ]
    
    def get_metric(self, name: str) -> Optional[MetricCalculator]:
        """Get a specific metric calculator."""
        return self._metrics.get(name)


# ============================================================================
# Intervention System
# ============================================================================

class InterventionHook(ABC):
    """
    Abstract base class for experiment interventions.
    
    Interventions modify the environment *before* the agent starts.
    This is crucial for causal inference in experiments.
    """
    
    @abstractmethod
    def apply(
        self,
        environment: Any,  # MarketplaceProvider
        trial_metadata: Dict[str, Any]
    ) -> None:
        """
        Apply intervention to environment.
        
        Args:
            environment: The marketplace to modify
            trial_metadata: Metadata about this trial (trial number, condition, etc.)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return intervention name."""
        pass
    
    @abstractmethod
    def should_apply(self, trial_number: int) -> bool:
        """
        Determine if this intervention should apply to this trial.
        
        Args:
            trial_number: Current trial number
            
        Returns:
            True if intervention should be applied
        """
        pass


# ============================================================================
# Experiment Configuration
# ============================================================================

@dataclass
class ExperimentConfig:
    """
    Complete configuration for an experiment.
    
    This extends the basic config with experiment-specific settings.
    """
    # Identification
    experiment_id: str
    name: str
    description: str
    
    # Components
    agent_config: Dict[str, Any]
    environment_config: Dict[str, Any]
    
    # Execution
    num_trials: int = 1
    max_steps_per_trial: int = 10
    random_seed: Optional[int] = None
    
    # Interventions
    interventions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metrics to track
    metrics: List[str] = field(default_factory=list)
    
    # Output
    output_dir: str = "experiment_results"
    save_screenshots: bool = True
    save_trajectories: bool = True
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """
    Results from running an experiment.
    
    Contains all trajectories and aggregated metrics.
    """
    experiment_id: str
    config: ExperimentConfig
    
    # All trial trajectories
    trajectories: List[Trajectory] = field(default_factory=list)
    
    # Aggregated metrics
    metrics: Dict[str, List[MetricResult]] = field(default_factory=dict)
    
    # Summary statistics
    success_rate: float = 0.0
    avg_steps: float = 0.0
    avg_duration: float = 0.0
    
    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = field(default_factory=datetime.now)
    
    def add_trajectory(self, trajectory: Trajectory) -> None:
        """Add a trajectory from a completed trial."""
        self.trajectories.append(trajectory)
    
    def compute_summary_statistics(self) -> None:
        """Compute summary statistics from trajectories."""
        if not self.trajectories:
            return
        
        self.success_rate = sum(
            1 for t in self.trajectories if t.success
        ) / len(self.trajectories)
        
        self.avg_steps = sum(
            t.num_steps for t in self.trajectories
        ) / len(self.trajectories)
        
        self.avg_duration = sum(
            t.duration for t in self.trajectories
        ) / len(self.trajectories)


# ============================================================================
# Experiment Runner Protocol
# ============================================================================

class ExperimentRunner(ABC):
    """
    Abstract base class for experiment runners.
    
    Orchestrates the "game loop" between agent and environment,
    with comprehensive logging and intervention support.
    """
    
    @abstractmethod
    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Run a complete experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Complete experiment results
        """
        pass
    
    @abstractmethod
    async def arun(self, config: ExperimentConfig) -> ExperimentResult:
        """Async version of run."""
        pass
    
    @abstractmethod
    def run_single_trial(
        self,
        trial_number: int,
        agent: Any,
        environment: Any,
    ) -> Trajectory:
        """
        Run a single trial.
        
        Args:
            trial_number: Trial number
            agent: Configured agent
            environment: Configured environment
            
        Returns:
            Trajectory from this trial
        """
        pass
