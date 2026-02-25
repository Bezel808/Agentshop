"""
Structured Trajectory Logger

Provides JSONL logging for agent-environment interactions.
Every observation, thought, and action is recorded with modality tags.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

from aces.experiments.protocols import (
    Trajectory,
    TrajectoryStep,
    StepType,
)


logger = logging.getLogger(__name__)


class TrajectoryLogger:
    """
    Structured logger for agent trajectories.
    
    Logs to JSONL format (one JSON object per line) for easy analysis.
    
    Features:
    - Modality tagging (visual/verbal) at each step
    - Thought process recording (Chain of Thought)
    - Tool call and result logging
    - Timestamp precision for timing analysis
    """
    
    def __init__(self, output_dir: Path, experiment_id: str):
        """
        Initialize trajectory logger.
        
        Args:
            output_dir: Directory to save logs
            experiment_id: Experiment identifier
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_id = experiment_id
        self.current_trajectory: Optional[Trajectory] = None
        
        # JSONL files
        self.trajectories_file = self.output_dir / "trajectories.jsonl"
        self.steps_file = self.output_dir / "steps.jsonl"
        
        logger.info(f"Initialized TrajectoryLogger for {experiment_id}")
    
    def start_trajectory(
        self,
        trial_number: int,
        agent_config: Dict[str, Any],
        environment_config: Dict[str, Any],
    ) -> None:
        """Start logging a new trajectory."""
        self.current_trajectory = Trajectory(
            experiment_id=self.experiment_id,
            trial_number=trial_number,
            agent_config=agent_config,
            environment_config=environment_config,
            start_time=time.time(),
        )
        
        logger.info(f"Started trajectory for trial {trial_number}")
    
    def log_observation(
        self,
        observation: Any,
        modality: str,
        environment_state: Optional[Dict] = None,
    ) -> None:
        """
        Log an observation step.
        
        Args:
            observation: The observation data
            modality: "visual", "verbal", or "multimodal"
            environment_state: Current environment state
        """
        if not self.current_trajectory:
            raise RuntimeError("No active trajectory. Call start_trajectory() first.")
        
        step = TrajectoryStep(
            step_number=len(self.current_trajectory.steps) + 1,
            timestamp=time.time(),
            step_type=StepType.OBSERVATION,
            content=self._serialize_observation(observation),
            input_modality=modality,
            environment_state=environment_state,
            metadata={
                "observation_type": type(observation).__name__,
            }
        )
        
        self.current_trajectory.add_step(step)
        self._write_step_to_jsonl(step)
    
    def log_thought(
        self,
        reasoning: str,
        triggered_by_modality: str,
        agent_state: Optional[Dict] = None,
    ) -> None:
        """
        Log agent's reasoning/thought process.
        
        Args:
            reasoning: Agent's reasoning text
            triggered_by_modality: What modality triggered this thought
            agent_state: Current agent state
        """
        if not self.current_trajectory:
            raise RuntimeError("No active trajectory")
        
        step = TrajectoryStep(
            step_number=len(self.current_trajectory.steps) + 1,
            timestamp=time.time(),
            step_type=StepType.THOUGHT,
            content=reasoning,
            input_modality=triggered_by_modality,
            agent_state=agent_state,
            metadata={
                "reasoning_length": len(reasoning),
            }
        )
        
        self.current_trajectory.add_step(step)
        self._write_step_to_jsonl(step)
    
    def log_action(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_state: Optional[Dict] = None,
    ) -> None:
        """
        Log an action (tool call).
        
        Args:
            tool_name: Name of tool being called
            parameters: Tool parameters
            agent_state: Current agent state
        """
        if not self.current_trajectory:
            raise RuntimeError("No active trajectory")
        
        step = TrajectoryStep(
            step_number=len(self.current_trajectory.steps) + 1,
            timestamp=time.time(),
            step_type=StepType.ACTION,
            content={
                "tool_name": tool_name,
                "parameters": parameters,
            },
            output_modality="action",
            agent_state=agent_state,
            metadata={
                "tool": tool_name,
            }
        )
        
        self.current_trajectory.add_step(step)
        self._write_step_to_jsonl(step)
    
    def log_tool_result(
        self,
        result: Any,
        environment_state: Optional[Dict] = None,
    ) -> None:
        """
        Log tool execution result.
        
        Args:
            result: Tool result data
            environment_state: Environment state after tool execution
        """
        if not self.current_trajectory:
            raise RuntimeError("No active trajectory")
        
        step = TrajectoryStep(
            step_number=len(self.current_trajectory.steps) + 1,
            timestamp=time.time(),
            step_type=StepType.TOOL_RESULT,
            content=self._serialize_tool_result(result),
            environment_state=environment_state,
        )
        
        self.current_trajectory.add_step(step)
        self._write_step_to_jsonl(step)
    
    def end_trajectory(
        self,
        success: bool,
        final_action: Optional[Any] = None,
    ) -> Trajectory:
        """
        Finalize and save trajectory.
        
        Args:
            success: Whether the trial succeeded
            final_action: Final action taken (if any)
            
        Returns:
            Completed trajectory
        """
        if not self.current_trajectory:
            raise RuntimeError("No active trajectory")
        
        self.current_trajectory.success = success
        self.current_trajectory.final_action = final_action
        self.current_trajectory.end_time = time.time()
        
        # Write complete trajectory to JSONL
        self._write_trajectory_to_jsonl(self.current_trajectory)
        
        trajectory = self.current_trajectory
        self.current_trajectory = None
        
        logger.info(
            f"Ended trajectory: {trajectory.num_steps} steps, "
            f"{trajectory.duration:.2f}s, success={success}"
        )
        
        return trajectory
    
    # ========================================================================
    # Private Methods
    # ========================================================================
    
    def _serialize_observation(self, observation: Any) -> Dict[str, Any]:
        """Serialize observation for logging."""
        # Don't log full image data (too large)
        # Just log metadata
        if hasattr(observation, 'to_dict'):
            return observation.to_dict()
        elif hasattr(observation, '__dict__'):
            obs_dict = observation.__dict__.copy()
            # Remove large data
            if 'screenshot' in obs_dict:
                obs_dict['screenshot'] = f"<{len(obs_dict['screenshot'])} bytes>"
            return obs_dict
        else:
            return {"data": str(observation)[:500]}  # Truncate
    
    def _serialize_tool_result(self, result: Any) -> Dict[str, Any]:
        """Serialize tool result for logging."""
        if isinstance(result, dict):
            return result
        else:
            return {"result": str(result)[:500]}
    
    def _write_step_to_jsonl(self, step: TrajectoryStep) -> None:
        """Write a step to the steps JSONL file."""
        with open(self.steps_file, 'a') as f:
            json.dump(step.to_dict(), f)
            f.write('\n')
    
    def _write_trajectory_to_jsonl(self, trajectory: Trajectory) -> None:
        """Write complete trajectory to JSONL file."""
        with open(self.trajectories_file, 'a') as f:
            json.dump(trajectory.to_dict(), f)
            f.write('\n')


# Add missing import
import time
