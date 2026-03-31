# Unified Orchestration

`AgentOrchestrator` 提供统一执行循环：

1. `get_initial_observation()`
2. `agent.act_batch()/act()`
3. `execute_action(action)`
4. `should_stop(action, result, step)`
5. `get_next_observation()`

并统一输出事件：
- `observation`
- `action`
- `tool_result`
- `termination`

该循环已用于：
- `run_browser_agent.py`（实时运行）
- `aces/experiments/runner.py`（实验运行）

当前 Agent 决策链路是“用户需求先被抽取为搜索关键词，再进入 Orchestrator 的 observe → decide → tool → next observation 循环，直到触发 recommend 终止并输出单商品推荐”。
