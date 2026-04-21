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
from aces.environments.web_renderer_marketplace import WebRendererMarketplace
from aces.environments.router import (
    MarketplaceFactory,
    MarketplaceAdapter,
)
from aces.environments.browser_shopping_env import BrowserShoppingEnv
from aces.environments.api_shopping_env import APIShoppingEnv
from aces.environments.mcp_shopping_env import MCPShoppingEnv

try:
    # Optional dependency: llama_index + embedding stack (HF/transformers)
    from aces.environments.llamaindex_marketplace import LlamaIndexMarketplace  # type: ignore
except Exception:  # pragma: no cover
    LlamaIndexMarketplace = None  # type: ignore[assignment]

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
    "WebRendererMarketplace",  # 带网页渲染的离线环境
    
    # Factory and Adapter
    "MarketplaceFactory",
    "MarketplaceAdapter",

    # Shopping env (Tool-First refactor)
    "BrowserShoppingEnv",
    "APIShoppingEnv",
    "MCPShoppingEnv",
]

if LlamaIndexMarketplace is not None:
    __all__.append("LlamaIndexMarketplace")  # 学术级 RAG 搜索
