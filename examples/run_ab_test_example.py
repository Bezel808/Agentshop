"""
Example: Running a Visual vs Verbal A/B Test

This demonstrates the complete workflow:
1. Load experiment config
2. Set up interventions and metrics
3. Run experiment
4. Analyze results
"""

import logging
import yaml
from pathlib import Path

from aces.experiments.runner import StandardExperimentRunner
from aces.experiments.protocols import ExperimentConfig
from aces.experiments.metrics import (
    DecisionTimeMetric,
    SelectedProductRankMetric,
    ToolUsageCountMetric,
    ModalityUsageMetric,
    ReasoningQualityMetric,
)
from aces.experiments.interventions import (
    PositionShuffleIntervention,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_visual_vs_verbal_test():
    """
    Run the complete Visual vs Verbal A/B test.
    
    This is a complete example of how to use the experiment system.
    """
    
    print("\n" + "="*80)
    print("ACES v2: Visual vs Verbal A/B Test")
    print("="*80 + "\n")
    
    # ========================================================================
    # Step 1: Load Configuration
    # ========================================================================
    print("Step 1: Loading experiment configuration...")
    
    config_path = "configs/experiments/visual_vs_verbal_ab_test.yaml"
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to ExperimentConfig object
    config = ExperimentConfig(
        experiment_id=config_dict['experiment_id'],
        name=config_dict['name'],
        description=config_dict['description'],
        agent_config=config_dict['agent'],
        environment_config=config_dict['environment'],
        num_trials=config_dict['num_trials'],
        max_steps_per_trial=config_dict['max_steps_per_trial'],
        random_seed=config_dict['random_seed'],
        metrics=config_dict['metrics'],
        output_dir=config_dict['output_dir'],
        save_screenshots=config_dict['save_screenshots'],
        save_trajectories=config_dict['save_trajectories'],
        metadata=config_dict['metadata'],
    )
    
    print(f"✓ Loaded config: {config.name}")
    print(f"  Trials: {config.num_trials}")
    print(f"  Max steps: {config.max_steps_per_trial}\n")
    
    # ========================================================================
    # Step 2: Set Up Metrics
    # ========================================================================
    print("Step 2: Registering metric calculators...")
    
    from aces.experiments.protocols import MetricRegistry
    
    metric_registry = MetricRegistry()
    metric_registry.register(DecisionTimeMetric())
    metric_registry.register(SelectedProductRankMetric())
    metric_registry.register(ToolUsageCountMetric())
    metric_registry.register(ModalityUsageMetric())
    metric_registry.register(ReasoningQualityMetric())
    
    print(f"✓ Registered {len(metric_registry._metrics)} metrics\n")
    
    # ========================================================================
    # Step 3: Set Up Interventions
    # ========================================================================
    print("Step 3: Setting up interventions...")
    
    interventions = [
        # Shuffle positions for all trials (control for position bias)
        PositionShuffleIntervention(seed=42, apply_probability=1.0),
        
        # More interventions would be added here based on config
    ]
    
    print(f"✓ Configured {len(interventions)} interventions\n")
    
    # ========================================================================
    # Step 4: Create Experiment Runner
    # ========================================================================
    print("Step 4: Creating experiment runner...")
    
    runner = StandardExperimentRunner(
        metric_registry=metric_registry,
        intervention_hooks=interventions,
    )
    
    print("✓ Runner initialized\n")
    
    # ========================================================================
    # Step 5: Run Experiment
    # ========================================================================
    print("Step 5: Running experiment...")
    print(f"This will take approximately {config.num_trials * 20} seconds")
    print("(assuming ~20 seconds per trial)\n")
    
    print("NOTE: This is a DEMO - actual execution requires:")
    print("  1. Implementing full agent.act() logic")
    print("  2. Connecting tools to marketplace")
    print("  3. Handling perception mode switching")
    print("\nFor now, we'll show what the logs would look like:\n")
    
    # ========================================================================
    # Step 6: Demo - What the Logs Look Like
    # ========================================================================
    print("="*80)
    print("SAMPLE TRAJECTORY LOG (JSONL format)")
    print("="*80 + "\n")
    
    # Example trajectory step
    sample_step = {
        "step_number": 1,
        "timestamp": 1706544000.123,
        "step_type": "observation",
        "content": {
            "modality": "visual",
            "has_image": True,
            "num_products": 8,
        },
        "input_modality": "visual",
        "environment_state": {
            "query": "mousepad",
            "num_products": 8,
            "mode": "offline"
        },
        "metadata": {
            "initial": True
        }
    }
    
    print("Step 1: Initial Observation")
    print(json.dumps(sample_step, indent=2))
    
    sample_thought = {
        "step_number": 2,
        "timestamp": 1706544001.456,
        "step_type": "thought",
        "content": "I see 8 mousepads. Let me analyze prices and ratings...",
        "input_modality": "visual",  # Thought triggered by visual input
        "agent_state": {
            "step_count": 1,
            "num_messages": 2
        },
        "metadata": {
            "reasoning_length": 58
        }
    }
    
    print("\nStep 2: Agent Reasoning")
    print(json.dumps(sample_thought, indent=2))
    
    sample_action = {
        "step_number": 3,
        "timestamp": 1706544002.789,
        "step_type": "action",
        "content": {
            "tool_name": "add_to_cart",
            "parameters": {
                "product_id": "mousepad_3",
                "product_title": "SteelSeries QcK Gaming Mouse Pad"
            }
        },
        "output_modality": "action",
        "metadata": {
            "tool": "add_to_cart"
        }
    }
    
    print("\nStep 3: Agent Action")
    print(json.dumps(sample_action, indent=2))
    
    print("\n" + "="*80)
    print("These logs enable analysis like:")
    print("  - Filter by input_modality='visual' vs 'verbal'")
    print("  - Measure time between steps (timestamps)")
    print("  - Extract reasoning patterns (thought content)")
    print("  - Analyze tool usage by perception mode")
    print("="*80 + "\n")
    
    # ========================================================================
    # Step 7: Analysis Preview
    # ========================================================================
    print("Step 7: Post-Experiment Analysis\n")
    
    print("After 100 trials complete, you can analyze:")
    print()
    print("1. Load trajectories:")
    print("   >>> import pandas as pd")
    print("   >>> df = pd.read_json('trajectories.jsonl', lines=True)")
    print()
    print("2. Compare groups:")
    print("   >>> visual_trials = df[df['trial_number'] < 50]")
    print("   >>> verbal_trials = df[df['trial_number'] >= 50]")
    print()
    print("3. Test hypothesis:")
    print("   >>> from scipy import stats")
    print("   >>> stats.ttest_ind(visual_times, verbal_times)")
    print()
    print("4. Visualize:")
    print("   >>> import seaborn as sns")
    print("   >>> sns.boxplot(data=df, x='perception_mode', y='decision_time')")
    print()
    
    print("="*80)
    print("✓ A/B Test Example Complete")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_visual_vs_verbal_test()
    
    print("\nNext steps:")
    print("  1. Review the generated logs in experiment_results/")
    print("  2. Implement full agent-environment loop in runner.py")
    print("  3. Run real experiments with actual LLM calls")
    print("  4. Analyze results using pandas/scipy")
    print()
