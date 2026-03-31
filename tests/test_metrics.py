import time

from aces.experiments.protocols import Trajectory, TrajectoryStep, StepType
from aces.experiments.metrics import (
    SelectedProductRankMetric,
    PriceSensitivityMetric,
    InvalidToolCallRateMetric,
    RetryRateMetric,
    EnvErrorRateMetric,
)


def _traj():
    t = Trajectory(
        experiment_id="e1",
        trial_number=0,
        agent_config={},
        environment_config={},
        success=True,
        final_action={"product_id": "p2"},
        start_time=time.time() - 2,
        end_time=time.time(),
    )
    t.add_step(TrajectoryStep(step_number=1, timestamp=time.time(), step_type=StepType.ACTION, content={"tool_name": "search", "parameters": {"q": "x"}}))
    t.add_step(TrajectoryStep(step_number=2, timestamp=time.time(), step_type=StepType.TOOL_RESULT, content={"success": True, "data": {"products": [
        {"id": "p1", "price": 10.0},
        {"id": "p2", "price": 20.0},
        {"id": "p3", "price": 30.0},
    ]}}))
    t.add_step(TrajectoryStep(step_number=3, timestamp=time.time(), step_type=StepType.ACTION, content={"tool_name": "recommend", "parameters": {"product_id": "p2"}}))
    t.add_step(TrajectoryStep(step_number=4, timestamp=time.time(), step_type=StepType.TOOL_RESULT, content={"success": False, "error": "timeout"}))
    t.add_step(TrajectoryStep(step_number=5, timestamp=time.time(), step_type=StepType.ACTION, content={"tool_name": "recommend", "parameters": {"product_id": "p2"}}))
    return t


def test_rank_and_price_metrics():
    t = _traj()
    rank = SelectedProductRankMetric().calculate(t)
    assert rank.value == 2

    price = PriceSensitivityMetric().calculate(t)
    assert round(price.value, 6) == 0.0


def test_engineering_metrics():
    t = _traj()
    invalid = InvalidToolCallRateMetric().calculate(t)
    assert invalid.value == 0.0

    retry = RetryRateMetric().calculate(t)
    assert retry.value > 0

    env_err = EnvErrorRateMetric().calculate(t)
    assert env_err.value > 0
