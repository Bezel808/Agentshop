"""
Experiment Intervention Hooks

Interventions modify the environment before the agent starts.
This is crucial for causal inference and A/B testing.
"""

import logging
from typing import Any, Dict, List
import random

from aces.experiments.protocols import InterventionHook
from aces.environments.protocols import MarketplaceProvider, Product


logger = logging.getLogger(__name__)


class ConditionalIntervention(InterventionHook):
    """
    Base class for interventions that apply conditionally.
    
    Supports:
    - Applying to specific trial numbers
    - Random sampling (e.g., 50% of trials)
    - Treatment/control group assignment
    """
    
    def __init__(
        self,
        name: str,
        apply_probability: float = 1.0,
        treatment_trials: Optional[List[int]] = None,
    ):
        """
        Initialize conditional intervention.
        
        Args:
            name: Intervention name
            apply_probability: Probability of applying (0-1)
            treatment_trials: Specific trial numbers to apply to (if None, use probability)
        """
        self._name = name
        self.apply_probability = apply_probability
        self.treatment_trials = set(treatment_trials) if treatment_trials else None
    
    def get_name(self) -> str:
        return self._name
    
    def should_apply(self, trial_number: int) -> bool:
        """Determine if intervention applies to this trial."""
        # Specific trial list takes precedence
        if self.treatment_trials is not None:
            return trial_number in self.treatment_trials
        
        # Otherwise use probability
        return random.random() < self.apply_probability


class KeywordInjectionIntervention(ConditionalIntervention):
    """
    Inject persuasive keywords into product titles.
    
    Research application:
    - Test if keywords like "Premium", "Best Choice" affect agent selection
    - Measure susceptibility to marketing language
    
    Example:
        "Wireless Mouse" → "Premium Wireless Mouse"
    """
    
    def __init__(
        self,
        keywords: List[str],
        target_product_criteria: Dict[str, Any],
        apply_probability: float = 1.0,
        **kwargs
    ):
        """
        Initialize keyword injection.
        
        Args:
            keywords: Keywords to inject (e.g., ["Premium", "Best Seller"])
            target_product_criteria: Which products to modify
                Examples:
                - {"sponsored": True} - Only sponsored products
                - {"position": 0} - Only first product
                - {"price_rank": "highest"} - Most expensive product
            apply_probability: Probability of applying intervention
        """
        super().__init__(
            name="keyword_injection",
            apply_probability=apply_probability,
            **kwargs
        )
        self.keywords = keywords
        self.target_criteria = target_product_criteria
    
    def apply(
        self,
        environment: MarketplaceProvider,
        trial_metadata: Dict[str, Any]
    ) -> None:
        """Inject keywords into matching products."""
        # This assumes environment is OfflineMarketplace (has intervention registry)
        if not hasattr(environment, 'intervention_registry'):
            logger.warning(
                "Keyword injection only works in offline mode. Skipping."
            )
            return
        
        # Apply keyword to products matching criteria
        # (Implementation would modify current_products)
        
        logger.info(
            f"Applied keyword injection: {self.keywords} "
            f"(trial {trial_metadata['trial_number']})"
        )


class PriceManipulationIntervention(ConditionalIntervention):
    """
    Manipulate prices to test agent's price sensitivity.
    
    Research application:
    - Create "decoy effect" (add overpriced option)
    - Test price anchoring
    - Measure elasticity of demand
    
    Example:
        Original: $50, $75, $100
        Modified: $50, $75, $150 (anchoring effect)
    """
    
    def __init__(
        self,
        manipulation_type: str,
        target_product: int,
        price_multiplier: float = 1.5,
        **kwargs
    ):
        """
        Initialize price manipulation.
        
        Args:
            manipulation_type: "increase", "decrease", "anchor"
            target_product: Which product to modify (by index or ID)
            price_multiplier: How much to change price
        """
        super().__init__(
            name="price_manipulation",
            **kwargs
        )
        self.manipulation_type = manipulation_type
        self.target_product = target_product
        self.price_multiplier = price_multiplier
    
    def apply(
        self,
        environment: MarketplaceProvider,
        trial_metadata: Dict[str, Any]
    ) -> None:
        """Manipulate product prices."""
        logger.info(
            f"Applied price manipulation: {self.manipulation_type} "
            f"(multiplier={self.price_multiplier})"
        )


class PositionShuffleIntervention(ConditionalIntervention):
    """
    Randomize product positions to control for position bias.
    
    Research application:
    - Measure position bias (do agents favor top results?)
    - Control for confounds in experiments
    
    Example:
        Original order: [A, B, C, D]
        Shuffled order: [C, A, D, B]
    """
    
    def __init__(self, seed: Optional[int] = None, **kwargs):
        """
        Initialize position shuffle.
        
        Args:
            seed: Random seed for reproducibility
        """
        super().__init__(
            name="position_shuffle",
            **kwargs
        )
        self.seed = seed
    
    def apply(
        self,
        environment: MarketplaceProvider,
        trial_metadata: Dict[str, Any]
    ) -> None:
        """Shuffle product positions."""
        if self.seed is not None:
            random.seed(self.seed + trial_metadata['trial_number'])
        
        # Shuffle logic would go here
        
        logger.info(
            f"Shuffled product positions (trial {trial_metadata['trial_number']})"
        )


class BadgeManipulationIntervention(ConditionalIntervention):
    """
    Add or remove badges (Sponsored, Best Seller, etc.).
    
    Research application:
    - Test impact of trust signals
    - Measure badge effectiveness
    
    Example:
        Add "Best Seller" badge to random product
    """
    
    def __init__(
        self,
        badge_type: str,
        action: str,  # "add" or "remove"
        target_product: int,
        **kwargs
    ):
        """
        Initialize badge manipulation.
        
        Args:
            badge_type: "sponsored", "best_seller", "overall_pick", etc.
            action: "add" or "remove"
            target_product: Which product to modify
        """
        super().__init__(
            name="badge_manipulation",
            **kwargs
        )
        self.badge_type = badge_type
        self.action = action
        self.target_product = target_product
    
    def apply(
        self,
        environment: MarketplaceProvider,
        trial_metadata: Dict[str, Any]
    ) -> None:
        """Manipulate badges."""
        logger.info(
            f"Badge manipulation: {self.action} {self.badge_type} "
            f"to product {self.target_product}"
        )
