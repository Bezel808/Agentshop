"""
Experiment Runner Implementation

Orchestrates agent-environment interactions with comprehensive logging.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from aces.experiments.protocols import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
    Trajectory,
    TrajectoryStep,
    StepType,
    InterventionHook,
    MetricRegistry,
)
from aces.core.protocols import Agent, Observation, Action
from aces.environments.protocols import MarketplaceProvider, PageState
from aces.config.loader import ConfigLoader
from aces.environments.router import MarketplaceFactory, MarketplaceAdapter


logger = logging.getLogger(__name__)


class StandardExperimentRunner(ExperimentRunner):
    """
    Standard experiment runner with comprehensive logging.
    
    This orchestrates the "game loop":
    1. Initialize agent and environment
    2. Apply interventions (if any)
    3. Run interaction loop
    4. Log every step with modality tags
    5. Calculate metrics
    6. Save results
    """
    
    def __init__(
        self,
        metric_registry: Optional[MetricRegistry] = None,
        intervention_hooks: Optional[List[InterventionHook]] = None,
    ):
        """
        Initialize experiment runner.
        
        Args:
            metric_registry: Registry of metric calculators
            intervention_hooks: List of intervention hooks
        """
        self.metric_registry = metric_registry or MetricRegistry()
        self.intervention_hooks = intervention_hooks or []
        
        logger.info("Initialized StandardExperimentRunner")
    
    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Run a complete experiment with multiple trials.
        
        Workflow:
        1. Create agent and environment from config
        2. For each trial:
           a. Reset agent and environment
           b. Apply interventions
           c. Run interaction loop
           d. Log trajectory
        3. Calculate metrics
        4. Save results
        """
        logger.info(
            f"Starting experiment: {config.experiment_id} "
            f"({config.num_trials} trials)"
        )
        
        # Create result object
        result = ExperimentResult(
            experiment_id=config.experiment_id,
            config=config,
            start_time=datetime.now(),
        )
        
        # Create output directory
        output_dir = Path(config.output_dir) / config.experiment_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config.__dict__, f, indent=2, default=str)
        
        # Create agent and environment
        agent = ConfigLoader.instantiate_agent(config.agent_config)
        marketplace_provider = MarketplaceFactory.create(config.environment_config)
        environment = MarketplaceAdapter(marketplace_provider)
        
        # Run trials
        for trial_num in range(config.num_trials):
            logger.info(f"Running trial {trial_num + 1}/{config.num_trials}")
            
            try:
                # Run single trial
                trajectory = self.run_single_trial(
                    trial_number=trial_num,
                    agent=agent,
                    environment=environment,
                    config=config,
                    output_dir=output_dir,
                )
                
                # Add to results
                result.add_trajectory(trajectory)
                
                # Save trajectory to JSONL
                self._save_trajectory(trajectory, output_dir)
                
            except Exception as e:
                logger.error(f"Trial {trial_num} failed: {e}")
                # Log failed trial
                failed_trajectory = Trajectory(
                    experiment_id=config.experiment_id,
                    trial_number=trial_num,
                    agent_config=config.agent_config,
                    environment_config=config.environment_config,
                    success=False,
                    metadata={"error": str(e)}
                )
                result.add_trajectory(failed_trajectory)
        
        # Calculate metrics for all trajectories
        self._calculate_metrics(result)
        
        # Compute summary statistics
        result.compute_summary_statistics()
        
        # Save final results
        result.end_time = datetime.now()
        self._save_results(result, output_dir)
        
        # Cleanup
        environment.close()
        
        logger.info(
            f"Experiment complete: {config.experiment_id} "
            f"(success rate: {result.success_rate:.2%})"
        )
        
        return result
    
    async def arun(self, config: ExperimentConfig) -> ExperimentResult:
        """Async version of run (implementation similar)."""
        # For now, just call sync version
        # Full async implementation would use asyncio.gather for parallel trials
        return self.run(config)
    
    def run_single_trial(
        self,
        trial_number: int,
        agent: Agent,
        environment: MarketplaceAdapter,
        config: ExperimentConfig,
        output_dir: Path,
    ) -> Trajectory:
        """
        Run a single trial of agent-environment interaction.
        
        This is the "game loop" with comprehensive logging.
        """
        # Create trajectory
        trajectory = Trajectory(
            experiment_id=config.experiment_id,
            trial_number=trial_number,
            agent_config=config.agent_config,
            environment_config=config.environment_config,
            start_time=time.time(),
        )
        
        step_count = 0
        
        try:
            # Reset agent and environment
            agent.reset()
            page_state = environment.reset()
            
            # === INTERVENTION HOOK ===
            # Apply interventions before agent starts
            for intervention in self.intervention_hooks:
                if intervention.should_apply(trial_number):
                    logger.info(
                        f"Applying intervention: {intervention.get_name()} "
                        f"to trial {trial_number}"
                    )
                    intervention.apply(
                        environment.provider,
                        trial_metadata={
                            "trial_number": trial_number,
                            "experiment_id": config.experiment_id,
                        }
                    )
                    
                    # Log intervention
                    trajectory.metadata[f"intervention_{intervention.get_name()}"] = True
            
            # Get initial observation
            observation = self._get_observation(environment, agent)
            
            # Log initial observation
            step_count += 1
            trajectory.add_step(TrajectoryStep(
                step_number=step_count,
                timestamp=time.time(),
                step_type=StepType.OBSERVATION,
                content=self._serialize_observation(observation),
                input_modality=observation.modality,
                environment_state=self._get_env_state(environment),
                metadata={"initial": True}
            ))
            
            # === MAIN INTERACTION LOOP ===
            for _ in range(config.max_steps_per_trial):
                # Agent decides on action
                action = agent.act(observation)
                
                step_count += 1
                
                # Log agent's thought process (if available in action)
                if action.reasoning:
                    trajectory.add_step(TrajectoryStep(
                        step_number=step_count,
                        timestamp=time.time(),
                        step_type=StepType.THOUGHT,
                        content=action.reasoning,
                        input_modality=observation.modality,  # What modality triggered this thought
                        agent_state=self._get_agent_state(agent),
                    ))
                    step_count += 1
                
                # Log action
                trajectory.add_step(TrajectoryStep(
                    step_number=step_count,
                    timestamp=time.time(),
                    step_type=StepType.ACTION,
                    content={
                        "tool_name": action.tool_name,
                        "parameters": action.parameters,
                    },
                    output_modality="action",
                    agent_state=self._get_agent_state(agent),
                ))
                
                # Execute action in environment
                tool_result = self._execute_action(environment, action, agent)
                
                step_count += 1
                
                # Log tool result
                trajectory.add_step(TrajectoryStep(
                    step_number=step_count,
                    timestamp=time.time(),
                    step_type=StepType.TOOL_RESULT,
                    content=self._serialize_tool_result(tool_result),
                    environment_state=self._get_env_state(environment),
                ))
                
                # Check if done (e.g., product added to cart)
                if action.tool_name == "add_to_cart" and tool_result.get("success"):
                    trajectory.success = True
                    trajectory.final_action = action.parameters
                    break
                
                # Get next observation
                observation = self._get_observation(environment, agent)
                
                step_count += 1
                trajectory.add_step(TrajectoryStep(
                    step_number=step_count,
                    timestamp=time.time(),
                    step_type=StepType.OBSERVATION,
                    content=self._serialize_observation(observation),
                    input_modality=observation.modality,
                    environment_state=self._get_env_state(environment),
                ))
            
            trajectory.end_time = time.time()
            
            # Save screenshot if enabled
            if config.save_screenshots:
                self._save_screenshot(environment, output_dir, trial_number)
            
        except Exception as e:
            logger.error(f"Trial {trial_number} failed: {e}")
            trajectory.success = False
            trajectory.metadata["error"] = str(e)
            trajectory.end_time = time.time()
        
        return trajectory
    
    # ========================================================================
    # Private Helper Methods
    # ========================================================================
    
    def _get_observation(
        self,
        environment: MarketplaceAdapter,
        agent: Agent
    ) -> Observation:
        """Get observation from environment in agent's perception mode."""
        page_state = environment.get_page_state()
        
        # Use agent's perception mode to encode
        # Pass products as the raw state
        if page_state.products:
            observation = agent.perception.encode(page_state.products)
        elif page_state.screenshot:
            observation = agent.perception.encode(page_state.screenshot)
        else:
            # Fallback: encode page state itself
            observation = agent.perception.encode(page_state)
        
        return observation
    
    def _execute_action(
        self,
        environment: MarketplaceAdapter,
        action: Action,
        agent: Agent
    ) -> Dict[str, Any]:
        """Execute action using agent's tools."""
        # Get the tool from agent
        tool = agent.tools.get(action.tool_name)
        
        if not tool:
            logger.error(f"Tool '{action.tool_name}' not found in agent")
            return {"success": False, "error": f"Unknown tool: {action.tool_name}"}
        
        # Execute the tool
        try:
            result = tool.execute(action.parameters)
            
            if result.success:
                return {"success": True, "data": result.data}
            else:
                return {"success": False, "error": result.error}
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_agent_state(self, agent: Agent) -> Dict[str, Any]:
        """Get serializable agent state."""
        state = agent.get_state()
        return {
            "step_count": state.step_count,
            "num_messages": len(state.message_history),
            "num_observations": len(state.observations),
        }
    
    def _get_env_state(self, environment: MarketplaceAdapter) -> Dict[str, Any]:
        """Get serializable environment state."""
        page_state = environment.get_page_state()
        return {
            "num_products": len(page_state.products),
            "query": page_state.query,
            "mode": environment.mode.value,
        }
    
    def _serialize_observation(self, observation: Observation) -> Dict[str, Any]:
        """Serialize observation for logging."""
        return {
            "modality": observation.modality,
            "timestamp": observation.timestamp,
            "data_type": type(observation.data).__name__,
            "metadata": observation.metadata,
            # Don't save full image data in trajectory log (too large)
            "has_image": observation.modality == "visual",
        }
    
    def _serialize_tool_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize tool result for logging."""
        # Simplify for logging
        return {
            "success": result.get("success", False),
            "data_summary": str(result.get("data", ""))[:200],  # Truncate
        }
    
    def _save_trajectory(self, trajectory: Trajectory, output_dir: Path) -> None:
        """Save trajectory to JSONL file."""
        trajectories_file = output_dir / "trajectories.jsonl"
        
        with open(trajectories_file, 'a') as f:
            json.dump(trajectory.to_dict(), f)
            f.write('\n')
        
        logger.debug(f"Saved trajectory for trial {trajectory.trial_number}")
    
    def _save_screenshot(
        self,
        environment: MarketplaceAdapter,
        output_dir: Path,
        trial_number: int
    ) -> None:
        """Save screenshot from environment."""
        page_state = environment.get_page_state()
        
        if page_state.screenshot:
            screenshot_path = output_dir / f"trial_{trial_number}_screenshot.png"
            with open(screenshot_path, 'wb') as f:
                f.write(page_state.screenshot)
    
    def _calculate_metrics(self, result: ExperimentResult) -> None:
        """Calculate all registered metrics for all trajectories."""
        for trajectory in result.trajectories:
            for calc in self.metric_registry._metrics.values():
                metric_result = calc.calculate(trajectory)
                
                if metric_result.metric_name not in result.metrics:
                    result.metrics[metric_result.metric_name] = []
                
                result.metrics[metric_result.metric_name].append(metric_result)
    
    def _save_results(self, result: ExperimentResult, output_dir: Path) -> None:
        """Save aggregated results."""
        # Save summary
        summary = {
            "experiment_id": result.experiment_id,
            "num_trials": len(result.trajectories),
            "success_rate": result.success_rate,
            "avg_steps": result.avg_steps,
            "avg_duration": result.avg_duration,
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat(),
        }
        
        summary_path = output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save metrics
        metrics_path = output_dir / "metrics.json"
        metrics_data = {
            name: [m.value for m in values]
            for name, values in result.metrics.items()
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
