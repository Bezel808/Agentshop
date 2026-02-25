"""
ACES Environment Layer

The "Twin Worlds" architecture for marketplace providers.
"""

from aces.environments.protocols import (
    MarketplaceProvider,
    MarketplaceMode,
    Product,
    SearchResult,
    PageState,
    MarketingIntervention,
    InterventionRegistry,
)

from aces.environments.offline_marketplace import OfflineMarketplace
from aces.environments.online_marketplace import OnlineMarketplace
from aces.environments.llamaindex_marketplace import LlamaIndexMarketplace
from aces.environments.web_renderer_marketplace import WebRendererMarketplace
from aces.environments.router import (
    MarketplaceFactory,
    MarketplaceAdapter,
)


__all__ = [
    # Protocols
    "MarketplaceProvider",
    "MarketplaceMode",
    "Product",
    "SearchResult",
    "PageState",
    "MarketingIntervention",
    "InterventionRegistry",
    
    # Implementations
    "OfflineMarketplace",
    "OnlineMarketplace",
    "LlamaIndexMarketplace",  # 学术级 RAG 搜索
    "WebRendererMarketplace",  # 带网页渲染的离线环境
    
    # Factory and Adapter
    "MarketplaceFactory",
    "MarketplaceAdapter",
]
