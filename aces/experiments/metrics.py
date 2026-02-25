"""
Standard Metric Calculators

Provides common metrics for e-commerce agent experiments.
"""

import logging
from typing import List

from aces.experiments.protocols import (
    MetricCalculator,
    MetricResult,
    Trajectory,
    StepType,
)


logger = logging.getLogger(__name__)


class DecisionTimeMetric(MetricCalculator):
    """
    Measures time from initial observation to final decision.
    
    Research question: Do visual vs verbal agents decide faster?
    """
    
    def calculate(self, trajectory: Trajectory) -> MetricResult:
        """Calculate decision time."""
        decision_time = trajectory.duration
        
        return MetricResult(
            metric_name="decision_time",
            value=decision_time,
            unit="seconds",
            metadata={
                "num_steps": trajectory.num_steps,
            }
        )
    
    def get_name(self) -> str:
        return "decision_time"
    
    def get_description(self) -> str:
        return "Time from start to final decision (seconds)"


class SelectedProductRankMetric(MetricCalculator):
    """
    Measures the rank/position of the selected product.
    
    Research question: Do agents exhibit position bias?
    """
    
    def calculate(self, trajectory: Trajectory) -> MetricResult:
        """Calculate selected product rank."""
        rank = None
        
        if trajectory.success and trajectory.final_action:
            # Extract product info from final action
            product_id = trajectory.final_action.get("product_id")
            
            # Find product in environment state
            for step in trajectory.steps:
                if step.step_type == StepType.OBSERVATION:
                    env_state = step.environment_state
                    if env_state:
                        # Rank would be extracted from product position
                        # For now, placeholder
                        rank = 1  # Would compute actual rank
                        break
        
        return MetricResult(
            metric_name="selected_product_rank",
            value=rank,
            unit="position",
            metadata={
                "success": trajectory.success,
            }
        )
    
    def get_name(self) -> str:
        return "selected_product_rank"
    
    def get_description(self) -> str:
        return "Position/rank of selected product in search results"


class ToolUsageCountMetric(MetricCalculator):
    """
    Counts how many tools the agent used.
    
    Research question: Do verbal agents use more tools (exploration)?
    """
    
    def calculate(self, trajectory: Trajectory) -> MetricResult:
        """Count tool usage."""
        tool_calls = [
            step for step in trajectory.steps
            if step.step_type == StepType.ACTION
        ]
        
        # Count by tool type
        tool_counts = {}
        for step in tool_calls:
            tool_name = step.content.get("tool_name")
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        
        return MetricResult(
            metric_name="tool_usage_count",
            value=len(tool_calls),
            unit="calls",
            metadata={"by_tool": tool_counts}
        )
    
    def get_name(self) -> str:
        return "tool_usage_count"
    
    def get_description(self) -> str:
        return "Number of tool calls made by agent"


class ModalityUsageMetric(MetricCalculator):
    """
    Tracks which modalities were used during the trial.
    
    Research question: How does perception mode affect information processing?
    """
    
    def calculate(self, trajectory: Trajectory) -> MetricResult:
        """Calculate modality usage."""
        modalities = set()
        
        for step in trajectory.steps:
            if step.input_modality:
                modalities.add(step.input_modality)
        
        # Count observations by modality
        modality_counts = {}
        for step in trajectory.steps:
            if step.step_type == StepType.OBSERVATION and step.input_modality:
                mod = step.input_modality
                modality_counts[mod] = modality_counts.get(mod, 0) + 1
        
        return MetricResult(
            metric_name="modality_usage",
            value=list(modalities),
            metadata={"counts": modality_counts}
        )
    
    def get_name(self) -> str:
        return "modality_usage"
    
    def get_description(self) -> str:
        return "Which modalities (visual/verbal) were used"


class PriceSensitivityMetric(MetricCalculator):
    """
    Measures if agent selected cheaper vs expensive products.
    
    Research question: Are agents price-sensitive?
    """
    
    def calculate(self, trajectory: Trajectory) -> MetricResult:
        """Calculate price sensitivity."""
        if not trajectory.success:
            return MetricResult(
                metric_name="price_sensitivity",
                value=None,
                metadata={"success": False}
            )
        
        # Would compute: (selected_price - avg_price) / std_price
        # For now, placeholder
        
        return MetricResult(
            metric_name="price_sensitivity",
            value=0.0,
            unit="z-score",
            metadata={"placeholder": True}
        )
    
    def get_name(self) -> str:
        return "price_sensitivity"
    
    def get_description(self) -> str:
        return "Whether agent selected cheaper products (z-score)"


class ReasoningQualityMetric(MetricCalculator):
    """
    Assesses quality of agent's reasoning.
    
    Research question: Do verbal agents provide better reasoning?
    """
    
    def calculate(self, trajectory: Trajectory) -> MetricResult:
        """Calculate reasoning quality."""
        # Extract all thought steps
        thoughts = [
            step for step in trajectory.steps
            if step.step_type == StepType.THOUGHT
        ]
        
        # Simple metric: length and coherence
        total_reasoning_length = sum(
            len(str(step.content)) for step in thoughts
        )
        
        # Quality score (simplified - would use LLM judge in practice)
        quality_score = min(total_reasoning_length / 100, 1.0)
        
        return MetricResult(
            metric_name="reasoning_quality",
            value=quality_score,
            unit="score",
            metadata={
                "num_thoughts": len(thoughts),
                "total_length": total_reasoning_length,
            }
        )
    
    def get_name(self) -> str:
        return "reasoning_quality"
    
    def get_description(self) -> str:
        return "Quality of agent's reasoning (0-1 score)"
