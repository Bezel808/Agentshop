"""
Environment Layer Protocols

Defines the abstract interfaces for marketplace providers.
These are MCP Server abstractions that hide the complexity of
data sources (offline files vs online scraping).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class MarketplaceMode(Enum):
    """Marketplace operation modes."""
    OFFLINE = "offline"  # Controlled sandbox using ACES datasets
    ONLINE = "online"    # Live web scraping


@dataclass
class Product:
    """
    Unified product representation.
    
    Both Offline and Online modes must convert their data to this format.
    """
    id: str
    title: str
    price: float
    rating: Optional[float] = None
    rating_count: Optional[int] = None
    image_url: Optional[str] = None
    description: Optional[str] = None
    
    # Platform-specific attributes
    sponsored: bool = False
    best_seller: bool = False
    overall_pick: bool = False
    low_stock: bool = False
    
    # Metadata
    position: Optional[int] = None  # Position in search results
    source: Optional[str] = None    # "offline" or "online:{platform}"
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """Result of a product search."""
    query: str
    products: List[Product]
    total_count: int
    page: int = 1
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PageState:
    """
    Current state of the marketplace page.
    
    Can be rendered as image (visual) or text (verbal).
    """
    products: List[Product]
    query: Optional[str] = None
    html: Optional[str] = None      # For offline mode
    screenshot: Optional[bytes] = None  # Pre-rendered or captured
    metadata: Dict[str, Any] = None


# ============================================================================
# The Marketplace Provider Interface (MCP Server Abstraction)
# ============================================================================

class MarketplaceProvider(ABC):
    """
    Abstract base class for marketplace data providers.
    
    This is the "MCP Server" abstraction - it provides a unified interface
    for accessing product data, regardless of whether the source is:
    - Offline: Local ACES datasets
    - Online: Live web scraping
    
    Agents interact with this via Tools, never directly.
    """
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the provider with configuration.
        
        Args:
            config: Provider-specific configuration
        """
        pass
    
    @abstractmethod
    def search_products(
        self,
        query: str,
        sort_by: str = "relevance",
        limit: int = 10,
        **kwargs
    ) -> SearchResult:
        """
        Search for products.
        
        Args:
            query: Search query
            sort_by: Sort order ("relevance", "price_asc", "price_desc", "rating")
            limit: Maximum number of results
            **kwargs: Provider-specific parameters
            
        Returns:
            Search results
        """
        pass
    
    @abstractmethod
    def get_product_details(self, product_id: str) -> Product:
        """
        Get detailed information about a product.
        
        Args:
            product_id: Unique product identifier
            
        Returns:
            Detailed product information
        """
        pass
    
    @abstractmethod
    def get_page_state(self) -> PageState:
        """
        Get current page state for rendering.
        
        Returns:
            Current page state (for visual/verbal perception)
        """
        pass
    
    @abstractmethod
    def add_to_cart(self, product_id: str, quantity: int = 1) -> Dict[str, Any]:
        """
        Add product to shopping cart.
        
        Args:
            product_id: Product to add
            quantity: Quantity to add
            
        Returns:
            Cart state after addition
        """
        pass
    
    @abstractmethod
    def reset(self) -> PageState:
        """
        Reset marketplace to initial state.
        
        Returns:
            Initial page state
        """
        pass
    
    @abstractmethod
    def get_mode(self) -> MarketplaceMode:
        """Return the marketplace mode."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Cleanup resources (e.g., close browser for online mode)."""
        pass


# ============================================================================
# Marketing Intervention Interface (For Research)
# ============================================================================

class MarketingIntervention(ABC):
    """
    Abstract base class for marketing interventions.
    
    Interventions modify product data to test agent bias and behavior.
    Only applicable in OFFLINE mode (can't modify live websites).
    """
    
    @abstractmethod
    def apply(self, product: Product) -> Product:
        """
        Apply intervention to a product.
        
        Args:
            product: Original product
            
        Returns:
            Modified product
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return intervention name."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Return intervention description."""
        pass


class InterventionRegistry:
    """
    Registry for marketing interventions.
    
    Allows composing multiple interventions and applying them
    to products in the offline marketplace.
    """
    
    def __init__(self):
        self._interventions: List[MarketingIntervention] = []
    
    def register(self, intervention: MarketingIntervention) -> None:
        """Register an intervention."""
        self._interventions.append(intervention)
    
    def apply_all(self, product: Product) -> Product:
        """Apply all registered interventions to a product."""
        modified = product
        for intervention in self._interventions:
            modified = intervention.apply(modified)
        return modified
    
    def clear(self) -> None:
        """Clear all interventions."""
        self._interventions.clear()
    
    def get_active_interventions(self) -> List[str]:
        """Get names of active interventions."""
        return [i.get_name() for i in self._interventions]
