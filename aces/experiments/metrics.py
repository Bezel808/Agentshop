"""
Standard Metric Calculators

Provides research + engineering metrics for agent experiments.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import math

from aces.experiments.protocols import (
    MetricCalculator,
    MetricResult,
    Trajectory,
    StepType,
)


logger = logging.getLogger(__name__)


def _action_steps(trajectory: Trajectory):
    return [s for s in trajectory.steps if s.step_type == StepType.ACTION]


def _tool_result_steps(trajectory: Trajectory):
    return [s for s in trajectory.steps if s.step_type == StepType.TOOL_RESULT]


def _thought_steps(trajectory: Trajectory):
    return [s for s in trajectory.steps if s.step_type == StepType.THOUGHT]


def _extract_recommended_product_id(trajectory: Trajectory) -> Optional[str]:
    if isinstance(trajectory.final_action, dict):
        pid = trajectory.final_action.get("product_id")
        if pid:
            return pid

    for step in reversed(_tool_result_steps(trajectory)):
        content = step.content if isinstance(step.content, dict) else {}
        data = content.get("data") if isinstance(content.get("data"), dict) else {}
        if data.get("recommended") and data.get("product_id"):
            return data.get("product_id")
    return None


def _extract_last_candidate_products(trajectory: Trajectory) -> List[Dict[str, Any]]:
    """Try to extract latest candidate product set from search tool results."""
    for step in reversed(_tool_result_steps(trajectory)):
        content = step.content if isinstance(step.content, dict) else {}
        data = content.get("data") if isinstance(content.get("data"), dict) else {}

        products = data.get("products")
        if isinstance(products, list) and products:
            return products

        # Some tool wrappers may nest payload.
        nested = data.get("result") if isinstance(data.get("result"), dict) else {}
        products = nested.get("products")
        if isinstance(products, list) and products:
            return products

    return []


class DecisionTimeMetric(MetricCalculator):
    def calculate(self, trajectory: Trajectory) -> MetricResult:
        return MetricResult(
            metric_name="decision_time",
            value=trajectory.duration,
            unit="seconds",
            metadata={"num_steps": trajectory.num_steps},
        )

    def get_name(self) -> str:
        return "decision_time"

    def get_description(self) -> str:
        return "Time from start to final decision (seconds)"


class SelectedProductRankMetric(MetricCalculator):
    def calculate(self, trajectory: Trajectory) -> MetricResult:
        pid = _extract_recommended_product_id(trajectory)
        products = _extract_last_candidate_products(trajectory)

        rank = None
        if pid and products:
            for idx, p in enumerate(products, 1):
                if str(p.get("id")) == str(pid):
                    rank = idx
                    break

        return MetricResult(
            metric_name="selected_product_rank",
            value=rank,
            unit="position",
            metadata={
                "success": trajectory.success,
                "recommended_product_id": pid,
                "candidate_count": len(products),
            },
        )

    def get_name(self) -> str:
        return "selected_product_rank"

    def get_description(self) -> str:
        return "Position/rank of selected product in candidate list"


class ToolUsageCountMetric(MetricCalculator):
    def calculate(self, trajectory: Trajectory) -> MetricResult:
        tool_calls = _action_steps(trajectory)
        tool_counts: Dict[str, int] = {}
        for step in tool_calls:
            tool_name = (step.content or {}).get("tool_name")
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

        return MetricResult(
            metric_name="tool_usage_count",
            value=len(tool_calls),
            unit="calls",
            metadata={"by_tool": tool_counts},
        )

    def get_name(self) -> str:
        return "tool_usage_count"

    def get_description(self) -> str:
        return "Number of tool calls made by agent"


class ModalityUsageMetric(MetricCalculator):
    def calculate(self, trajectory: Trajectory) -> MetricResult:
        modalities = set()
        modality_counts: Dict[str, int] = {}

        for step in trajectory.steps:
            if step.input_modality:
                modalities.add(step.input_modality)
            if step.step_type == StepType.OBSERVATION and step.input_modality:
                mod = step.input_modality
                modality_counts[mod] = modality_counts.get(mod, 0) + 1

        return MetricResult(
            metric_name="modality_usage",
            value=list(modalities),
            metadata={"counts": modality_counts},
        )

    def get_name(self) -> str:
        return "modality_usage"

    def get_description(self) -> str:
        return "Which modalities (visual/verbal) were used"


class PriceSensitivityMetric(MetricCalculator):
    def calculate(self, trajectory: Trajectory) -> MetricResult:
        pid = _extract_recommended_product_id(trajectory)
        products = _extract_last_candidate_products(trajectory)

        if not pid or not products:
            return MetricResult(
                metric_name="price_sensitivity",
                value=None,
                unit="z-score",
                metadata={"success": trajectory.success, "reason": "no_recommendation_or_candidates"},
            )

        prices = []
        selected_price = None
        for p in products:
            price = p.get("price")
            if price is None:
                continue
            try:
                price_f = float(price)
                prices.append(price_f)
                if str(p.get("id")) == str(pid):
                    selected_price = price_f
            except (TypeError, ValueError):
                continue

        if selected_price is None or len(prices) < 2:
            return MetricResult(
                metric_name="price_sensitivity",
                value=None,
                unit="z-score",
                metadata={"success": trajectory.success, "reason": "insufficient_price_data"},
            )

        mean_price = sum(prices) / len(prices)
        variance = sum((x - mean_price) ** 2 for x in prices) / len(prices)
        std = math.sqrt(variance)
        z_score = 0.0 if std == 0 else (selected_price - mean_price) / std

        return MetricResult(
            metric_name="price_sensitivity",
            value=z_score,
            unit="z-score",
            metadata={
                "selected_price": selected_price,
                "mean_price": mean_price,
                "num_candidates": len(prices),
            },
        )

    def get_name(self) -> str:
        return "price_sensitivity"

    def get_description(self) -> str:
        return "Selected-price z-score within candidate set"


class ReasoningQualityMetric(MetricCalculator):
    def calculate(self, trajectory: Trajectory) -> MetricResult:
        thoughts = _thought_steps(trajectory)
        if not thoughts:
            return MetricResult(
                metric_name="reasoning_quality",
                value=0.0,
                unit="score",
                metadata={"num_thoughts": 0, "reason": "no_thoughts"},
            )

        texts = [str(step.content or "") for step in thoughts]
        total_len = sum(len(t) for t in texts)
        avg_len = total_len / len(texts)

        # Lightweight heuristic score: length + actionability + diversity.
        action_keywords = ["because", "therefore", "compare", "price", "rating", "recommend", "back"]
        keyword_hits = sum(1 for t in texts for k in action_keywords if k in t.lower())
        diversity = len(set(" ".join(texts).lower().split()))

        length_score = min(avg_len / 180.0, 1.0)
        keyword_score = min(keyword_hits / max(len(texts), 1) / 2.0, 1.0)
        diversity_score = min(diversity / 120.0, 1.0)
        quality_score = round((0.4 * length_score + 0.4 * keyword_score + 0.2 * diversity_score), 4)

        return MetricResult(
            metric_name="reasoning_quality",
            value=quality_score,
            unit="score",
            metadata={
                "num_thoughts": len(thoughts),
                "total_length": total_len,
                "avg_length": avg_len,
                "keyword_hits": keyword_hits,
                "diversity": diversity,
            },
        )

    def get_name(self) -> str:
        return "reasoning_quality"

    def get_description(self) -> str:
        return "Heuristic reasoning quality score (0-1)"


class InvalidToolCallRateMetric(MetricCalculator):
    def calculate(self, trajectory: Trajectory) -> MetricResult:
        action_steps = _action_steps(trajectory)
        if not action_steps:
            return MetricResult(metric_name="invalid_toolcall_rate", value=0.0, unit="ratio", metadata={"total_actions": 0})

        invalid = 0
        for step in action_steps:
            name = str((step.content or {}).get("tool_name", ""))
            if not name or name == "noop":
                invalid += 1

        return MetricResult(
            metric_name="invalid_toolcall_rate",
            value=invalid / len(action_steps),
            unit="ratio",
            metadata={"invalid": invalid, "total_actions": len(action_steps)},
        )

    def get_name(self) -> str:
        return "invalid_toolcall_rate"

    def get_description(self) -> str:
        return "Fraction of invalid/no-op tool calls"


class RetryRateMetric(MetricCalculator):
    def calculate(self, trajectory: Trajectory) -> MetricResult:
        actions = _action_steps(trajectory)
        if len(actions) < 2:
            return MetricResult(metric_name="retry_rate", value=0.0, unit="ratio", metadata={"retries": 0, "total_actions": len(actions)})

        retries = 0
        prev = None
        for step in actions:
            key = (step.content or {}).get("tool_name"), tuple(sorted(((step.content or {}).get("parameters") or {}).items()))
            if prev == key:
                retries += 1
            prev = key

        return MetricResult(
            metric_name="retry_rate",
            value=retries / len(actions),
            unit="ratio",
            metadata={"retries": retries, "total_actions": len(actions)},
        )

    def get_name(self) -> str:
        return "retry_rate"

    def get_description(self) -> str:
        return "Fraction of consecutive repeated actions"


class EnvErrorRateMetric(MetricCalculator):
    def calculate(self, trajectory: Trajectory) -> MetricResult:
        results = _tool_result_steps(trajectory)
        if not results:
            return MetricResult(metric_name="env_error_rate", value=0.0, unit="ratio", metadata={"errors": 0, "total_results": 0})

        errors = 0
        for step in results:
            content = step.content if isinstance(step.content, dict) else {}
            success = content.get("success")
            if success is False:
                errors += 1

        return MetricResult(
            metric_name="env_error_rate",
            value=errors / len(results),
            unit="ratio",
            metadata={"errors": errors, "total_results": len(results)},
        )

    def get_name(self) -> str:
        return "env_error_rate"

    def get_description(self) -> str:
        return "Fraction of failed tool/environment executions"


class StepsToSuccessMetric(MetricCalculator):
    def calculate(self, trajectory: Trajectory) -> MetricResult:
        termination_step = None
        for step in trajectory.steps:
            if step.step_type == StepType.TERMINATION:
                termination_step = step.step_number
                break
        value = termination_step or trajectory.num_steps
        if not trajectory.success:
            value = None

        return MetricResult(
            metric_name="steps_to_success",
            value=value,
            unit="steps",
            metadata={"success": trajectory.success, "total_steps": trajectory.num_steps},
        )

    def get_name(self) -> str:
        return "steps_to_success"

    def get_description(self) -> str:
        return "Number of steps needed to reach success"
