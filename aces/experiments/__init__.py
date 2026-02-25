"""
ACES Experiment System

Provides experiment orchestration, logging, metrics, and interventions.
"""

from aces.experiments.protocols import (
    StepType,
    TrajectoryStep,
    Trajectory,
    MetricResult,
    MetricCalculator,
    MetricRegistry,
    InterventionHook,
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
)

from aces.experiments.runner import StandardExperimentRunner
from aces.experiments.logger import TrajectoryLogger

from aces.experiments.metrics import (
    DecisionTimeMetric,
    SelectedProductRankMetric,
    ToolUsageCountMetric,
    ModalityUsageMetric,
    PriceSensitivityMetric,
    ReasoningQualityMetric,
)

from aces.experiments.interventions import (
    ConditionalIntervention,
    KeywordInjectionIntervention,
    PriceManipulationIntervention,
    PositionShuffleIntervention,
    BadgeManipulationIntervention,
)


__all__ = [
    # Protocols
    "StepType",
    "TrajectoryStep",
    "Trajectory",
    "MetricResult",
    "MetricCalculator",
    "MetricRegistry",
    "InterventionHook",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentRunner",
    
    # Implementations
    "StandardExperimentRunner",
    "TrajectoryLogger",
    
    # Metrics
    "DecisionTimeMetric",
    "SelectedProductRankMetric",
    "ToolUsageCountMetric",
    "ModalityUsageMetric",
    "PriceSensitivityMetric",
    "ReasoningQualityMetric",
    
    # Interventions
    "ConditionalIntervention",
    "KeywordInjectionIntervention",
    "PriceManipulationIntervention",
    "PositionShuffleIntervention",
    "BadgeManipulationIntervention",
]
