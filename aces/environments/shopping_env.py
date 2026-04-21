"""
Shopping Environment Protocol

Defines the interface for browser/API shopping environments used by
the Tool-First agent. Both BrowserShoppingEnv and APIShoppingEnv implement this.
"""

from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from aces.core.protocols import Observation


class ShoppingEnv(Protocol):
    """
    Protocol for shopping environments.
    
    Environments maintain state (context, page, filters, viewed products)
    and provide observations (screenshot or formatted text) for the agent.
    Tools call env methods to perform actions.
    """

    def reset(self, keywords: str) -> Observation:
        """
        Initialize environment with search keywords.
        Navigates to search page or fetches initial results.
        
        Returns:
            Initial observation (screenshot or product list text)
        """
        ...

    def get_observation(self) -> Observation:
        """
        Get current page state as observation.
        
        Returns:
            Observation (visual: screenshot, verbal: formatted product/detail text)
        """
        ...

    def get_available_actions(self) -> Dict[str, Any]:
        """
        Return metadata about available actions for current context.
        Used to build system prompt.
        
        Returns:
            Dict with context, num_products, has_next, has_prev, viewed_ids, etc.
        """
        ...

    def execute_action(self, tool_name: str, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Execute a tool action against the environment.

        Returns:
            (success, error_message, data)
        """
        ...

    def get_state(self) -> Dict[str, Any]:
        """Return serializable state for tracing/debug."""
        ...

    # ---- Actions (called by Tools) ----

    def select_product(self, index: int) -> Tuple[bool, Optional[str]]:
        """
        Select product by 1-based index to view details.
        
        Returns:
            (success, error_message)
        """
        ...

    def next_page(self) -> Tuple[bool, Optional[str]]:
        """Go to next search results page. Returns (success, error)."""
        ...

    def prev_page(self) -> Tuple[bool, Optional[str]]:
        """Go to previous search results page. Returns (success, error)."""
        ...

    def filter_price(self, min_val: float, max_val: float) -> Tuple[bool, Optional[str]]:
        """Apply price filter. Returns (success, error)."""
        ...

    def filter_rating(self, min_val: float) -> Tuple[bool, Optional[str]]:
        """Apply rating filter. Returns (success, error)."""
        ...

    def back(self) -> Tuple[bool, Optional[str]]:
        """Return from detail page to search page. Returns (success, error)."""
        ...

    def get_viewed_for_recommend(self) -> List[Dict[str, Any]]:
        """
        Get list of viewed products for recommend tool.
        Each item: {"id": str, "title": str, "index": int}
        """
        ...

    def get_current_product(self) -> Optional[Dict[str, Any]]:
        """
        When on detail page, get current product info.
        Returns {"id": str, "title": str} or None.
        """
        ...

    def close(self) -> None:
        """Release resources (browser, etc.)."""
        ...
