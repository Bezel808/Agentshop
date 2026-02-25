"""
Tool factory for centralized creation logic.
"""

from __future__ import annotations

from typing import Iterable, List

from aces.tools.shopping_tools import SearchTool, AddToCartTool, ViewProductDetailsTool


_TOOL_MAP = {
    "search_products": SearchTool,
    "add_to_cart": AddToCartTool,
    "view_product_details": ViewProductDetailsTool,
}


class ToolFactory:
    @staticmethod
    def create(tool_name: str, marketplace_api):
        tool_key = tool_name.strip().lower()
        if tool_key not in _TOOL_MAP:
            raise ValueError(f"Unsupported tool: {tool_name}")
        return _TOOL_MAP[tool_key](marketplace_api)

    @staticmethod
    def create_many(tool_names: Iterable[str], marketplace_api) -> List:
        return [ToolFactory.create(name, marketplace_api) for name in tool_names]
