"""
Configuration Loader

Load experiments from YAML configuration files.
"""

from pathlib import Path
from typing import Any, Dict, List
import yaml
import logging

from aces.core.protocols import ExperimentConfig
from aces.agents.base_agent import ComposableAgent
from aces.llm_backends.factory import LLMBackendFactory
from aces.perception.factory import PerceptionFactory
from aces.tools.factory import ToolFactory


logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Loads and parses YAML configuration files.
    
    This is the "Dependency Injection Container" that assembles
    agents from config specifications.
    """
    
    @staticmethod
    def load_experiment(config_path: str) -> ExperimentConfig:
        """
        Load experiment configuration from YAML file.
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            ExperimentConfig object
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Parse and validate
        exp_config = ExperimentConfig(
            agent_config=config_dict.get("agent", {}),
            environment_config=config_dict.get("environment", {}),
            perception_mode=config_dict.get("perception_mode", "visual"),
            max_steps=config_dict.get("max_steps", 10),
            random_seed=config_dict.get("random_seed"),
            metadata=config_dict.get("metadata", {}),
        )
        
        logger.info(f"Loaded experiment config from {config_path}")
        
        return exp_config
    
    @staticmethod
    def instantiate_agent(
        config: Dict[str, Any],
        marketplace_api=None,
    ) -> ComposableAgent:
        """
        Instantiate an agent from configuration.
        
        Args:
            config: Agent configuration dict
            
        Returns:
            Configured ComposableAgent
        
        Example config:
            agent:
              llm:
                backend: openai
                model: gpt-4o
                temperature: 1.0
              perception:
                mode: visual
                detail_level: high
              tools:
                - search_products
                - add_to_cart
              system_prompt: "You are a helpful shopping assistant."
        """
        # Instantiate LLM backend
        llm_config = config.get("llm", {})
        llm = LLMBackendFactory.create(llm_config)
        
        # Instantiate perception mode
        perception_config = config.get("perception", {})
        perception = PerceptionFactory.create(perception_config)
        
        # Instantiate tools (simplified - would fetch from registry)
        tool_names = config.get("tools", [])
        if tool_names and marketplace_api is not None:
            tools = ToolFactory.create_many(tool_names, marketplace_api)
        else:
            tools = []
        
        # System prompt
        system_prompt = config.get("system_prompt", "You are a helpful assistant.")
        
        # Assemble agent
        agent = ComposableAgent(
            llm=llm,
            perception=perception,
            tools=tools,
            system_prompt=system_prompt,
        )
        
        logger.info("Instantiated agent from config")
        
        return agent
    
    @staticmethod
    def _create_llm(config: Dict[str, Any]):
        """Backward-compatible wrapper."""
        return LLMBackendFactory.create(config)
    
    @staticmethod
    def _create_perception(config: Dict[str, Any]):
        """Backward-compatible wrapper."""
        return PerceptionFactory.create(config)
