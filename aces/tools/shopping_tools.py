"""
Example Shopping Tools

Concrete implementations of tools for e-commerce scenarios.
"""

from typing import Any, Dict
import logging

from aces.tools.base_tool import BaseTool
from aces.environments.product_utils import (
    products_to_summaries,
    product_to_detail,
)


logger = logging.getLogger(__name__)


class SearchTool(BaseTool):
    """
    搜索商品工具。
    
    如果 marketplace 支持截图，可以选择返回截图数据。
    """
    """
    Tool for searching products in the marketplace.
    
    Example of a read-only tool (doesn't modify environment).
    """
    
    def __init__(self, marketplace_api):
        """
        Initialize search tool.
        
        Args:
            marketplace_api: Reference to marketplace to search
        """
        super().__init__(
            name="search_products",
            description="Search for products matching a query",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'laptop under $1000')",
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["price_asc", "price_desc", "rating", "relevance"],
                        "description": "How to sort results",
                        "default": "relevance",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50,
                    }
                },
                "required": ["query"],
            }
        )
        self.marketplace = marketplace_api
    
    def _execute_impl(self, parameters: Dict[str, Any]) -> Any:
        """Search marketplace for products."""
        query = parameters["query"]
        sort_by = parameters.get("sort_by", "relevance")
        limit = parameters.get("limit", 10)
        
        # Call marketplace API (MarketplaceAdapter method)
        results = self.marketplace.search_products(
            query=query,
            sort_by=sort_by,
            limit=limit,
        )
        
        # Convert SearchResult to dict for tool return
        if hasattr(results, 'products'):
            products_data = products_to_summaries(results.products)
            return {
                "query": results.query,
                "products": products_data,
                "count": len(products_data)
            }
        
        logger.info(f"Search '{query}' returned {len(results.products) if hasattr(results, 'products') else 0} results")
        
        return results


class AddToCartTool(BaseTool):
    """
    Tool for adding a product to cart.
    
    Example of a tool that modifies environment state.
    """
    
    def __init__(self, marketplace_api):
        """Initialize add-to-cart tool."""
        super().__init__(
            name="add_to_cart",
            description="Add a product to the shopping cart",
            input_schema={
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Unique product identifier",
                    },
                    "product_title": {
                        "type": "string",
                        "description": "Product title (for verification)",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Quantity to add",
                        "default": 1,
                        "minimum": 1,
                    }
                },
                "required": ["product_id", "product_title"],
            }
        )
        self.marketplace = marketplace_api
    
    def _execute_impl(self, parameters: Dict[str, Any]) -> Any:
        """Add product to cart."""
        product_id = parameters["product_id"]
        product_title = parameters["product_title"]
        quantity = parameters.get("quantity", 1)
        
        # Add to cart (MarketplaceAdapter method)
        result = self.marketplace.add_to_cart(
            product_id=product_id,
            quantity=quantity,
        )
        
        logger.info(f"Added {quantity}x '{product_title}' to cart")
        
        # Ensure result is dict
        if isinstance(result, dict):
            return {
                "success": result.get("success", True),
                "product_id": product_id,
                "product_title": product_title,
                "quantity": quantity,
                "cart_total": result.get("cart_total", 0),
            }
        
        return {
            "success": True,
            "product_id": product_id,
            "product_title": product_title,
            "quantity": quantity,
        }


class ViewProductDetailsTool(BaseTool):
    """
    Tool for viewing detailed product information.
    
    Example of a tool that provides deeper information.
    """
    
    def __init__(self, marketplace_api):
        """Initialize view details tool."""
        super().__init__(
            name="view_product_details",
            description="Get detailed information about a specific product",
            input_schema={
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Unique product identifier",
                    }
                },
                "required": ["product_id"],
            }
        )
        self.marketplace = marketplace_api
    
    def _execute_impl(self, parameters: Dict[str, Any]) -> Any:
        """Get product details."""
        product_id = parameters["product_id"]
        
        # Fetch details (MarketplaceAdapter method)
        details = self.marketplace.get_product_details(product_id)
        
        logger.info(f"Viewed details for product {product_id}")
        
        # Convert Product to dict
        if hasattr(details, '__dict__'):
            return product_to_detail(details)
        
        return details
