"""
Marketplace Router and Factory

Provides a unified interface for creating marketplace providers
based on configuration. This is the "entry point" that decides
whether to use offline or online mode.
"""

import logging
from typing import Any, Dict, Optional
from enum import Enum

from aces.environments.protocols import (
    MarketplaceProvider,
    MarketplaceMode,
)
from aces.environments.offline_marketplace import OfflineMarketplace
from aces.environments.online_marketplace import OnlineMarketplace
from aces.environments.llamaindex_marketplace import LlamaIndexMarketplace


logger = logging.getLogger(__name__)


class MarketplaceFactory:
    """
    Factory for creating marketplace providers.
    
    This is the "Router" that decides which implementation to use
    based on configuration.
    
    Usage:
        # Create from config
        config = {
            "mode": "offline",  # or "online"
            "offline": {
                "datasets_dir": "datasets/",
                "interventions": [...]
            },
            "online": {
                "platform": "amazon",
                "headless": True
            }
        }
        
        marketplace = MarketplaceFactory.create(config)
    """
    
    @staticmethod
    def create(config: Dict[str, Any]) -> MarketplaceProvider:
        """
        Create a marketplace provider based on configuration.
        
        Args:
            config: Configuration dict with 'mode' key
            
        Returns:
            Initialized MarketplaceProvider
            
        Raises:
            ValueError: If mode is invalid
        """
        mode = config.get("mode", "offline").lower()
        
        if mode == "offline":
            return MarketplaceFactory._create_offline(
                config.get("offline", {})
            )
        elif mode == "online":
            return MarketplaceFactory._create_online(
                config.get("online", {})
            )
        elif mode == "llamaindex":
            return MarketplaceFactory._create_llamaindex(
                config.get("llamaindex", {})
            )
        else:
            raise ValueError(
                f"Invalid marketplace mode: {mode}. "
                f"Must be 'offline', 'online', or 'llamaindex'"
            )
    
    @staticmethod
    def _create_offline(config: Dict[str, Any]) -> OfflineMarketplace:
        """Create and initialize offline marketplace."""
        marketplace = OfflineMarketplace()
        marketplace.initialize(config)
        
        logger.info("Created offline marketplace")
        return marketplace
    
    @staticmethod
    def _create_online(config: Dict[str, Any]) -> OnlineMarketplace:
        """Create and initialize online marketplace."""
        marketplace = OnlineMarketplace()
        marketplace.initialize(config)
        
        logger.info("Created online marketplace")
        return marketplace
    
    @staticmethod
    def _create_llamaindex(config: Dict[str, Any]) -> LlamaIndexMarketplace:
        """Create and initialize LlamaIndex marketplace (学术级)."""
        marketplace = LlamaIndexMarketplace()
        marketplace.initialize(config)
        
        logger.info("Created LlamaIndex marketplace (Academic-grade RAG)")
        return marketplace
    
    @staticmethod
    def create_from_env() -> MarketplaceProvider:
        """
        Create marketplace from environment variables.
        
        Convenience method for simple setups.
        
        Environment variables:
            ENV_MODE: "offline" or "online"
            ACES_DATASETS_DIR / DATASETS_DIR: datasets path (offline mode)
            MARKETPLACE_PLATFORM: "amazon" or "taobao" (online mode)
        """
        import os
        
        env_mode = os.getenv("ENV_MODE", "offline").lower()
        
        if env_mode == "offline":
            config = {
                "mode": "offline",
                "offline": {
                    "datasets_dir": os.getenv("ACES_DATASETS_DIR")
                    or os.getenv("DATASETS_DIR", "datasets_unified"),
                }
            }
        else:
            config = {
                "mode": "online",
                "online": {
                    "platform": os.getenv("MARKETPLACE_PLATFORM", "amazon"),
                    "headless": os.getenv("HEADLESS", "true").lower() == "true",
                }
            }
        
        return MarketplaceFactory.create(config)


class MarketplaceAdapter:
    """
    Adapter that wraps a marketplace provider and provides
    a consistent interface for agents.
    
    This is what the Tools interact with - they don't need to know
    if they're talking to offline or online marketplace.
    """
    
    def __init__(self, provider: MarketplaceProvider):
        """
        Initialize adapter with a provider.
        
        Args:
            provider: Marketplace provider (offline or online)
        """
        self.provider = provider
        self._mode = provider.get_mode()
        
        logger.info(f"Initialized marketplace adapter in {self._mode.value} mode")
    
    def search(self, query: str, **kwargs):
        """Search products (backwards-compatible alias)."""
        return self.provider.search_products(query, **kwargs)
    
    def search_products(self, query: str, **kwargs):
        """Search products (preferred API)."""
        return self.provider.search_products(query, **kwargs)
    
    def get_product_details(self, product_id: str):
        """Get product details (delegates to provider)."""
        return self.provider.get_product_details(product_id)
    
    def get_page_state(self):
        """Get page state (delegates to provider)."""
        return self.provider.get_page_state()
    
    def add_to_cart(self, product_id: str, quantity: int = 1):
        """Add to cart (delegates to provider)."""
        return self.provider.add_to_cart(product_id, quantity)
    
    def reset(self):
        """Reset marketplace (delegates to provider)."""
        return self.provider.reset()
    
    def close(self):
        """Close marketplace (delegates to provider)."""
        self.provider.close()
    
    @property
    def mode(self) -> MarketplaceMode:
        """Get marketplace mode."""
        return self._mode
    
    def is_offline(self) -> bool:
        """Check if in offline mode."""
        return self._mode == MarketplaceMode.OFFLINE
    
    def is_online(self) -> bool:
        """Check if in online mode."""
        return self._mode == MarketplaceMode.ONLINE
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on exit."""
        self.close()
