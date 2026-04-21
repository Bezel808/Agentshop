"""
ACES Experiment System

Lightweight package exports with lazy attribute loading.
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


__all__ = [
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
    "StandardExperimentRunner",
    "TrajectoryLogger",
    "DecisionTimeMetric",
    "SelectedProductRankMetric",
    "ToolUsageCountMetric",
    "ModalityUsageMetric",
    "PriceSensitivityMetric",
    "ReasoningQualityMetric",
    "InvalidToolCallRateMetric",
    "RetryRateMetric",
    "EnvErrorRateMetric",
    "StepsToSuccessMetric",
    "ConditionalIntervention",
    "KeywordInjectionIntervention",
    "PriceManipulationIntervention",
    "PositionShuffleIntervention",
    "BadgeManipulationIntervention",
]


def __getattr__(name):
    if name == "StandardExperimentRunner":
        from aces.experiments.runner import StandardExperimentRunner
        return StandardExperimentRunner
    if name == "TrajectoryLogger":
        from aces.experiments.logger import TrajectoryLogger
        return TrajectoryLogger

    if name in {
        "DecisionTimeMetric",
        "SelectedProductRankMetric",
        "ToolUsageCountMetric",
        "ModalityUsageMetric",
        "PriceSensitivityMetric",
        "ReasoningQualityMetric",
        "InvalidToolCallRateMetric",
        "RetryRateMetric",
        "EnvErrorRateMetric",
        "StepsToSuccessMetric",
    }:
        from aces.experiments import metrics as _metrics
        return getattr(_metrics, name)

    if name in {
        "ConditionalIntervention",
        "KeywordInjectionIntervention",
        "PriceManipulationIntervention",
        "PositionShuffleIntervention",
        "BadgeManipulationIntervention",
    }:
        from aces.experiments import interventions as _interventions
        return getattr(_interventions, name)

    raise AttributeError(name)
