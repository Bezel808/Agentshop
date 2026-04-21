"""
MCP Shopping Environment

MCP-driven shopping environment for visual mode that uses browser tools
through SimpleBrowserController. Falls back to API search metadata for
pagination/filtering and recommendation bookkeeping.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import time
import base64

from aces.core.protocols import Observation
from aces.environments.api_shopping_env import APIShoppingEnv
from aces.tools.browser_tools import SimpleBrowserController


class MCPShoppingEnv:
    """Shopping env using MCP browser tools for navigation/click/snapshot."""

    def __init__(
        self,
        mcp_caller: Any,
        web_server_url: str = "http://localhost:5000",
        user_query: str = "",
        condition_name: Optional[str] = None,
        page_num: int = 1,
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
        rating_min: Optional[float] = None,
        log_fn: Optional[Callable[[str, str], None]] = None,
        push_screenshot_fn: Optional[Callable[[bytes, str], None]] = None,
        log_path_fn: Optional[Callable[[str], None]] = None,
    ):
        if mcp_caller is None:
            raise RuntimeError("MCP backend selected but mcp_caller is None")

        self.web_server_url = web_server_url.rstrip("/")
        self.user_query = user_query
        self.condition_name = condition_name
        self.page_num = page_num
        self.price_min = price_min
        self.price_max = price_max
        self.rating_min = rating_min
        self.log_fn = log_fn or (lambda level, msg: None)
        self.push_screenshot_fn = push_screenshot_fn or (lambda b, u: None)
        self.log_path_fn = log_path_fn or (lambda s: None)

        self.controller = SimpleBrowserController(mcp_caller, base_url=self.web_server_url)
        self.api_env = APIShoppingEnv(
            web_server_url=web_server_url,
            user_query=user_query,
            condition_name=condition_name,
            page_num=page_num,
            price_min=price_min,
            price_max=price_max,
            rating_min=rating_min,
            log_fn=log_fn,
            log_path_fn=log_path_fn,
        )
        self.search_keywords = ""

    def _snapshot_to_observation(self, prompt: str) -> Observation:
        snap = self.controller.get_current_state()
        data = snap.data if snap and snap.success else {}
        screenshot_data = None
        if isinstance(data, dict):
            screenshot_data = data.get("screenshot") or data.get("image")

        if isinstance(screenshot_data, str) and screenshot_data.startswith("data:image/"):
            data_url = screenshot_data
            try:
                raw = base64.b64decode(data_url.split(",", 1)[1])
                self.push_screenshot_fn(raw, self.web_server_url)
            except Exception:
                pass
        else:
            # MCP snapshot may not include raw image; fallback to text-like dump.
            text_payload = str(data)[:8000]
            return Observation(
                data=text_payload,
                modality="verbal",
                timestamp=time.time(),
                metadata={"prompt": prompt, "backend": "mcp"},
            )

        return Observation(
            data=data_url,
            modality="visual",
            timestamp=time.time(),
            metadata={"prompt": prompt, "backend": "mcp"},
        )

    def reset(self, keywords: str) -> Observation:
        self.search_keywords = keywords
        self.api_env.reset(keywords)
        self.controller.goto_search_page(query=keywords)
        return self.get_observation()

    def get_observation(self) -> Observation:
        if self.api_env.context == "search":
            actions = self.api_env.get_available_actions()
            prompt = (
                f"User wants: {self.user_query}. Context=search. "
                f"Page {actions.get('page_num')}/{actions.get('total_pages')}. "
                "Use tools: select_product, next_page, prev_page, filter_price, filter_rating, recommend."
            )
        else:
            prompt = (
                f"User wants: {self.user_query}. Context=detail. "
                "Use recommend when confident, otherwise back."
            )
        return self._snapshot_to_observation(prompt)

    def get_available_actions(self) -> Dict[str, Any]:
        actions = self.api_env.get_available_actions()
        actions["backend"] = "mcp"
        return actions

    def get_state(self) -> Dict[str, Any]:
        state = self.api_env.get_state()
        state["backend"] = "mcp"
        return state

    def select_product(self, index: int) -> Tuple[bool, Optional[str]]:
        click_res = self.controller.click_product(index)
        ok, err = self.api_env.select_product(index)
        if not ok:
            return ok, err
        if hasattr(click_res, "success") and not click_res.success:
            self.log_fn("error", f"MCP click failed: {click_res.error}")
        return True, None

    def next_page(self) -> Tuple[bool, Optional[str]]:
        ok, err = self.api_env.next_page()
        if not ok:
            return ok, err
        self.controller.goto_search_page(query=self.search_keywords)
        return True, None

    def prev_page(self) -> Tuple[bool, Optional[str]]:
        ok, err = self.api_env.prev_page()
        if not ok:
            return ok, err
        self.controller.goto_search_page(query=self.search_keywords)
        return True, None

    def filter_price(self, min_val: float, max_val: float) -> Tuple[bool, Optional[str]]:
        return self.api_env.filter_price(min_val, max_val)

    def filter_rating(self, min_val: float) -> Tuple[bool, Optional[str]]:
        return self.api_env.filter_rating(min_val)

    def back(self) -> Tuple[bool, Optional[str]]:
        return self.api_env.back()

    def get_viewed_for_recommend(self) -> List[Dict[str, Any]]:
        return self.api_env.get_viewed_for_recommend()

    def get_current_product(self) -> Optional[Dict[str, Any]]:
        return self.api_env.get_current_product()

    def execute_action(self, tool_name: str, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        return self.api_env.execute_action(tool_name, parameters)

    def close(self) -> None:
        return None
