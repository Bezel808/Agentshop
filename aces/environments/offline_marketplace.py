"""
Offline Marketplace Provider

Implements a high-fidelity simulation using local ACES datasets.
Supports marketing interventions for research.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import random

from aces.environments.protocols import (
    MarketplaceProvider,
    MarketplaceMode,
    Product,
    SearchResult,
    PageState,
    InterventionRegistry,
)
from aces.config.settings import resolve_datasets_dir, resolve_screenshots_dir
from aces.environments.product_utils import product_from_dict


logger = logging.getLogger(__name__)


class OfflineMarketplace(MarketplaceProvider):
    """
    Offline marketplace using ACES datasets.
    
    Data sources:
    - Product JSON files (from ACES datasets/)
    - Pre-rendered screenshots
    - Experiment configurations
    
    Features:
    - Deterministic behavior (reproducibility)
    - Support for marketing interventions
    - Experimental control variables (price, label, title overrides)
    - Fast execution (no network I/O)
    """
    
    def __init__(self):
        """Initialize offline marketplace."""
        self.datasets_dir: Optional[Path] = None
        self.screenshots_dir: Optional[Path] = None
        self.current_query: Optional[str] = None
        self.current_products: List[Product] = []
        self.cart: List[Dict[str, Any]] = []
        
        # Marketing intervention system
        self.intervention_registry = InterventionRegistry()
        
        # Experimental control variables (new)
        self._active_condition = None  # ExperimentCondition
        
        # Cache for loaded datasets
        self._dataset_cache: Dict[str, List[Dict]] = {}
        
        logger.info("Initialized OfflineMarketplace")
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize with configuration.
        
        Config format:
            datasets_dir: Path to ACES datasets
            screenshots_dir: Path to pre-rendered screenshots
            default_query: Optional default search query
            interventions: List of intervention configs
        """
        self.datasets_dir = resolve_datasets_dir(config.get("datasets_dir"))
        self.screenshots_dir = resolve_screenshots_dir(config.get("screenshots_dir"))
        
        # Load interventions if specified
        if "interventions" in config:
            self._load_interventions(config["interventions"])
        
        logger.info(
            f"Initialized offline marketplace: "
            f"datasets={self.datasets_dir}, "
            f"interventions={self.intervention_registry.get_active_interventions()}"
        )
    
    def search_products(
        self,
        query: str,
        sort_by: str = "relevance",
        limit: int = 10,
        **kwargs
    ) -> SearchResult:
        """
        Search products in local datasets.
        
        Loads products from JSON files named by query
        (e.g., "mousepad.json", "laptop.json").
        """
        self.current_query = query
        
        # Load products from file
        products_data = self._load_products_for_query(query)
        
        # Convert to Product objects
        products = [
            self._dict_to_product(data, index)
            for index, data in enumerate(products_data)
        ]
        
        # Apply marketing interventions (legacy system)
        products = [
            self.intervention_registry.apply_all(p)
            for p in products
        ]
        
        # Apply experimental control variables (new system)
        if self._active_condition is not None:
            products = self._active_condition.apply(products)
        
        # Sort
        products = self._sort_products(products, sort_by)
        
        # Limit
        products = products[:limit]
        
        # Store current products
        self.current_products = products
        
        logger.info(
            f"Search '{query}' returned {len(products)} products "
            f"(sorted by {sort_by})"
        )
        
        return SearchResult(
            query=query,
            products=products,
            total_count=len(products_data),
            page=1,
            metadata={
                "mode": "offline",
                "interventions": self.intervention_registry.get_active_interventions(),
            }
        )
    
    def get_product_details(self, product_id: str) -> Product:
        """
        Get product details.
        
        In offline mode, we already have all details from JSON.
        """
        # Find product in current products
        for product in self.current_products:
            if product.id == product_id:
                return product
        
        raise ValueError(f"Product {product_id} not found in current results")
    
    def get_page_state(self) -> PageState:
        """
        Get current page state.
        
        Returns:
            PageState with products and optional screenshot
        """
        # Try to load pre-rendered screenshot if available
        screenshot = None
        if self.current_query and self.screenshots_dir:
            screenshot_path = (
                self.screenshots_dir / f"{self.current_query}.png"
            )
            if screenshot_path.exists():
                with open(screenshot_path, "rb") as f:
                    screenshot = f.read()
        
        return PageState(
            products=self.current_products,
            query=self.current_query,
            screenshot=screenshot,
            metadata={
                "mode": "offline",
                "source": "aces_dataset",
            }
        )
    
    def add_to_cart(self, product_id: str, quantity: int = 1) -> Dict[str, Any]:
        """
        Add product to cart.
        
        In offline mode, this is just state tracking.
        """
        product = self.get_product_details(product_id)
        
        self.cart.append({
            "product_id": product_id,
            "product_title": product.title,
            "quantity": quantity,
            "price": product.price,
        })
        
        cart_total = sum(item["price"] * item["quantity"] for item in self.cart)
        
        logger.info(f"Added {quantity}x '{product.title}' to cart")
        
        return {
            "success": True,
            "cart": self.cart,
            "cart_total": cart_total,
        }
    
    def reset(self) -> PageState:
        """Reset marketplace state."""
        self.current_query = None
        self.current_products = []
        self.cart = []
        
        logger.info("Reset offline marketplace")
        
        return PageState(products=[], metadata={"mode": "offline"})
    
    def get_mode(self) -> MarketplaceMode:
        """Return marketplace mode."""
        return MarketplaceMode.OFFLINE
    
    def close(self) -> None:
        """Cleanup (nothing to clean in offline mode)."""
        logger.info("Closed offline marketplace")
    
    # ========================================================================
    # Private Helper Methods
    # ========================================================================
    
    def _load_products_for_query(self, query: str) -> List[Dict]:
        """Load products from JSON file."""
        # Check cache
        if query in self._dataset_cache:
            return self._dataset_cache[query]
        
        # Try different filename formats
        filename_options = [
            f"{query}.json",
            f"{query.lower().replace(' ', '_')}.json",
            f"{query.lower().replace(' ', '+')}.json",
        ]
        
        for filename in filename_options:
            filepath = self.datasets_dir / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._dataset_cache[query] = data
                    return data
        
        logger.warning(f"No dataset found for query '{query}'")
        return []
    
    def _dict_to_product(self, data: Dict, index: int) -> Product:
        """Convert JSON dict to Product object."""
        return product_from_dict(
            data,
            index=index,
            source="offline",
            default_id_prefix="product",
        )
    
    def _sort_products(self, products: List[Product], sort_by: str) -> List[Product]:
        """Sort products."""
        if sort_by == "price_asc":
            return sorted(products, key=lambda p: p.price)
        elif sort_by == "price_desc":
            return sorted(products, key=lambda p: p.price, reverse=True)
        elif sort_by == "rating":
            return sorted(
                products,
                key=lambda p: p.rating if p.rating else 0,
                reverse=True
            )
        else:  # relevance (keep original order)
            return products
    
    def set_condition(self, condition) -> None:
        """
        Set the active experimental condition.
        
        Args:
            condition: ExperimentCondition instance, or None to clear.
        """
        self._active_condition = condition
        name = condition.name if condition else "None"
        logger.info(f"Active experiment condition set to: {name}")

    def clear_condition(self) -> None:
        """Remove active experimental condition."""
        self._active_condition = None
        logger.info("Cleared active experiment condition")

    def _load_interventions(self, intervention_configs: List[Dict]) -> None:
        """Load and register marketing interventions."""
        # This would instantiate intervention classes based on config
        # For now, placeholder
        logger.info(f"Would load {len(intervention_configs)} interventions")


# ============================================================================
# Example Marketing Interventions
# ============================================================================

class TitleEnhancementIntervention:
    """
    Add persuasive language to product titles.
    
    Example research question: Does adding "Premium" or "Best Seller"
    to titles affect agent choice?
    """
    
    def __init__(self, keywords: List[str]):
        """
        Initialize intervention.
        
        Args:
            keywords: Keywords to add (e.g., ["Premium", "Best Choice"])
        """
        self.keywords = keywords
    
    def apply(self, product: Product) -> Product:
        """Add keyword to title."""
        # Only apply to certain products (e.g., sponsored ones)
        if product.sponsored and self.keywords:
            keyword = random.choice(self.keywords)
            if keyword not in product.title:
                product.title = f"{keyword} {product.title}"
        return product
    
    def get_name(self) -> str:
        return "title_enhancement"
    
    def get_description(self) -> str:
        return f"Add keywords {self.keywords} to titles"


class PriceAnchoringIntervention:
    """
    Add fake "original price" to create anchoring effect.
    
    Example: Show "$99.99 $49.99" instead of just "$49.99"
    """
    
    def __init__(self, discount_pct: float = 0.3):
        """
        Initialize intervention.
        
        Args:
            discount_pct: Fake discount percentage (0.3 = 30% off)
        """
        self.discount_pct = discount_pct
    
    def apply(self, product: Product) -> Product:
        """Add fake original price."""
        if not product.raw_data:
            product.raw_data = {}
        
        original_price = product.price / (1 - self.discount_pct)
        product.raw_data["original_price"] = round(original_price, 2)
        product.raw_data["discount_pct"] = self.discount_pct
        
        return product
    
    def get_name(self) -> str:
        return "price_anchoring"
    
    def get_description(self) -> str:
        return f"Add {self.discount_pct*100}% fake discount"
