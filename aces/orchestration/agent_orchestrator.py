"""
Unified Agent Orchestrator

Provides a reusable execution loop:
observe -> decide -> execute -> record -> stop
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import time

from aces.core.protocols import Action, Observation, ToolResult


@dataclass
class OrchestratorEvent:
    """A normalized execution event emitted by the orchestrator."""

    event_type: str
    timestamp: float
    step: int
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorResult:
    """Execution result summary."""

    success: bool
    reason: str
    steps: int
    actions_executed: int
    errors: int
    repeated_actions: int
    action_counts: Dict[str, int]


class AgentOrchestrator:
    """Reusable orchestration loop for both live and experiment runs."""

    def __init__(
        self,
        *,
        agent: Any,
        get_initial_observation: Callable[[], Observation],
        get_next_observation: Callable[[], Observation],
        execute_action: Callable[[Action], ToolResult],
        should_stop: Optional[Callable[[Action, ToolResult, int], Optional[str]]] = None,
        on_event: Optional[Callable[[OrchestratorEvent], None]] = None,
        max_steps: int = 40,
        max_repeated_actions: int = 6,
        max_errors: int = 8,
    ):
        self.agent = agent
        self.get_initial_observation = get_initial_observation
        self.get_next_observation = get_next_observation
        self.execute_action = execute_action
        self.should_stop = should_stop
        self.on_event = on_event
        self.max_steps = max_steps
        self.max_repeated_actions = max_repeated_actions
        self.max_errors = max_errors

    def _emit(self, event_type: str, step: int, **payload: Any) -> None:
        if not self.on_event:
            return
        self.on_event(
            OrchestratorEvent(
                event_type=event_type,
                timestamp=time.time(),
                step=step,
                payload=payload,
            )
        )

    def _get_actions(self, observation: Observation) -> List[Action]:
        if hasattr(self.agent, "act_batch"):
            actions = self.agent.act_batch(observation)
            if actions:
                return actions
        return [self.agent.act(observation)]

    def run(self) -> OrchestratorResult:
        step = 0
        action_counter: Counter[str] = Counter()
        errors = 0
        repeats = 0
        last_action_key: Optional[str] = None

        observation = self.get_initial_observation()
        self._emit("observation", step, observation=observation)

        for _ in range(self.max_steps):
            step += 1
            actions = self._get_actions(observation)
            if not actions:
                self._emit("termination", step, reason="empty_actions")
                return OrchestratorResult(
                    success=False,
                    reason="empty_actions",
                    steps=step,
                    actions_executed=sum(action_counter.values()),
                    errors=errors,
                    repeated_actions=repeats,
                    action_counts=dict(action_counter),
                )

            for action in actions:
                self._emit("action", step, action=action)
                action_counter[action.tool_name] += 1

                action_key = f"{action.tool_name}:{action.parameters}"
                if action_key == last_action_key:
                    repeats += 1
                else:
                    repeats = 0
                last_action_key = action_key

                if repeats >= self.max_repeated_actions:
                    self._emit("termination", step, reason="repeat_threshold")
                    return OrchestratorResult(
                        success=False,
                        reason="repeat_threshold",
                        steps=step,
                        actions_executed=sum(action_counter.values()),
                        errors=errors,
                        repeated_actions=repeats,
                        action_counts=dict(action_counter),
                    )

                result = self.execute_action(action)
                self._emit("tool_result", step, action=action, result=result)

                if not result.success:
                    errors += 1
                    if errors >= self.max_errors:
                        self._emit("termination", step, reason="error_threshold")
                        return OrchestratorResult(
                            success=False,
                            reason="error_threshold",
                            steps=step,
                            actions_executed=sum(action_counter.values()),
                            errors=errors,
                            repeated_actions=repeats,
                            action_counts=dict(action_counter),
                        )

                if self.should_stop:
                    stop_reason = self.should_stop(action, result, step)
                    if stop_reason:
                        self._emit("termination", step, reason=stop_reason)
                        return OrchestratorResult(
                            success=True,
                            reason=stop_reason,
                            steps=step,
                            actions_executed=sum(action_counter.values()),
                            errors=errors,
                            repeated_actions=repeats,
                            action_counts=dict(action_counter),
                        )

            observation = self.get_next_observation()
            self._emit("observation", step, observation=observation)

        self._emit("termination", step, reason="max_steps")
        return OrchestratorResult(
            success=False,
            reason="max_steps",
            steps=step,
            actions_executed=sum(action_counter.values()),
            errors=errors,
            repeated_actions=repeats,
            action_counts=dict(action_counter),
        )
