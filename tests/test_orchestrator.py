from aces.orchestration import AgentOrchestrator
from aces.core.protocols import Observation, Action, ToolResult


class _BatchAgent:
    def __init__(self):
        self.calls = 0

    def act_batch(self, observation):
        self.calls += 1
        if self.calls == 1:
            return [Action(tool_name="search", parameters={"q": "watch"})]
        return [Action(tool_name="recommend", parameters={"product_id": "p1"})]


def test_orchestrator_stops_on_recommend():
    agent = _BatchAgent()

    def _obs():
        return Observation(data="x", modality="verbal", timestamp=0.0)

    def _exec(action):
        if action.tool_name == "recommend":
            return ToolResult(success=True, data={"recommended": True})
        return ToolResult(success=True, data={})

    def _stop(action, result, _step):
        if action.tool_name == "recommend" and result.success:
            return "recommend_done"
        return None

    orch = AgentOrchestrator(
        agent=agent,
        get_initial_observation=_obs,
        get_next_observation=_obs,
        execute_action=_exec,
        should_stop=_stop,
        max_steps=5,
    )
    result = orch.run()
    assert result.success is True
    assert result.reason == "recommend_done"
    assert result.action_counts["search"] == 1
    assert result.action_counts["recommend"] == 1
