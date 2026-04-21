"""
API Shopping Environment

HTTP API-based environment for verbal mode. Fetches product data
via REST and provides formatted text observations.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

from aces.core.protocols import Observation


class APIShoppingEnv:
    """
    Shopping environment using HTTP API (no browser).
    Used in verbal mode - provides formatted product text observations.
    """

    def __init__(
        self,
        web_server_url: str = "http://localhost:5000",
        user_query: str = "",
        condition_name: Optional[str] = None,
        page_num: int = 1,
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
        rating_min: Optional[float] = None,
        log_fn: Optional[Callable[[str, str], None]] = None,
        log_path_fn: Optional[Callable[[str], None]] = None,
    ):
        self.web_server_url = web_server_url.rstrip("/")
        self.user_query = user_query
        self.condition_name = condition_name
        self.page_num = page_num
        self.price_min = price_min
        self.price_max = price_max
        self.rating_min = rating_min
        self.log_fn = log_fn or (lambda level, msg: None)
        self.log_path_fn = log_path_fn or (lambda s: None)

        self.search_keywords = ""
        self.context = "search"
        self.products: List[dict] = []
        self.total_pages = 1
        self.current_detail: Optional[dict] = None
        self.current_product_id: Optional[str] = None
        self.viewed: List[Dict[str, Any]] = []

    def _api_search(
        self,
        keywords: str,
        limit: int = 8,
        page: int = 1,
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
        rating_min: Optional[float] = None,
    ) -> dict:
        url = f"{self.web_server_url}/api/search?q={quote(keywords)}&page_size={limit}&page={page}"
        if self.condition_name:
            url += f"&condition_name={quote(self.condition_name)}"
        if price_min is not None and price_min > 0:
            url += f"&price_min={price_min}"
        if price_max is not None and price_max > 0:
            url += f"&price_max={price_max}"
        if rating_min is not None and rating_min > 0:
            url += f"&rating_min={rating_min}"
        try:
            resp = requests.get(url, timeout=15)
            data = resp.json()
            return {
                "products": data.get("products", []),
                "page": data.get("page", 1),
                "total_pages": data.get("total_pages", 1),
            }
        except Exception as e:
            self.log_fn("error", f"API search request failed: {e}")
            return {"products": [], "page": 1, "total_pages": 1}

    def _api_product_detail(self, product_id: str) -> Optional[dict]:
        url = f"{self.web_server_url}/api/product/{quote(product_id)}"
        if self.condition_name:
            url += f"?condition_name={quote(self.condition_name)}"
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
            return data.get("product")
        except Exception as e:
            self.log_fn("error", f"API product detail request failed: {e}")
            return None

    @staticmethod
    def _format_product_list(products: List[dict]) -> str:
        lines = []
        for i, p in enumerate(products, 1):
            badges = []
            if p.get("sponsored"):
                badges.append("Sponsored")
            if p.get("best_seller"):
                badges.append("Best Seller")
            if p.get("overall_pick"):
                badges.append("Overall Pick")
            badge_str = f"  [{', '.join(badges)}]" if badges else ""
            pid = p.get("id", "?")
            lines.append(
                f"[{i}] ID:{pid} | {p['title']}\n"
                f"    Price: ${p['price']:.2f} | "
                f"Rating: {p.get('rating', 0):.1f}/5 ({p.get('rating_count', 0)} reviews)"
                f"{badge_str}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_product_detail(p: dict) -> str:
        parts = [
            f"Product ID: {p.get('id', 'N/A')}",
            f"Title: {p['title']}",
            f"Price: ${p['price']:.2f}",
            f"Rating: {p.get('rating', 0):.1f}/5 ({p.get('rating_count', 0)} reviews)",
        ]
        if p.get("description"):
            parts.append(f"\nDescription:\n{p['description']}")
        reviews = p.get("reviews") or []
        if reviews:
            parts.append("\nCustomer Reviews:")
            for i, r in enumerate(reviews[:10], 1):
                text = r.get("text") or r.get("content") or str(r)
                rating = r.get("rating")
                rating_str = f" ({rating}★)" if rating else ""
                parts.append(f"  [{i}]{rating_str} {text[:200]}{'...' if len(text) > 200 else ''}")
        return "\n".join(parts)

    def reset(self, keywords: str) -> Observation:
        self.search_keywords = keywords
        self.context = "search"
        self.current_detail = None
        self.current_product_id = None

        self.log_fn(
            "action",
            f'[Verbal] Fetching products via API: "{keywords}" '
            f"(page={self.page_num}, price={self.price_min}-{self.price_max}, rating>={self.rating_min})",
        )
        data = self._api_search(
            keywords,
            page=self.page_num,
            price_min=self.price_min,
            price_max=self.price_max,
            rating_min=self.rating_min,
        )
        self.products = data["products"]
        self.total_pages = data["total_pages"]

        if not self.products:
            self.log_fn("error", "No products found.")
            text = "No products found."
        else:
            self.log_fn("action", f"[Verbal] Retrieved {len(self.products)} products (page {self.page_num}/{self.total_pages})")
            product_text = self._format_product_list(self.products)
            has_next = self.page_num < self.total_pages
            has_prev = self.page_num > 1
            page_hint = (
                f"Page {self.page_num}/{self.total_pages}. "
                + ("Say next_page for next page. " if has_next else "(No more pages.) ")
                + ("Say prev_page to go back. " if has_prev else "")
            )
            text = (
                f"You are helping user find: \"{self.user_query}\"\n\n"
                f"Search results ({page_hint}):\n\n{product_text}\n\n"
                "Use tools: select_product(index), next_page, prev_page, filter_price(min,max), "
                "filter_rating(min), or recommend when ready."
            )

        return Observation(
            data=text,
            modality="verbal",
            timestamp=time.time(),
            metadata={"prompt": text},
        )

    def get_observation(self) -> Observation:
        if self.context == "search":
            product_text = self._format_product_list(self.products)
            has_next = self.page_num < self.total_pages
            has_prev = self.page_num > 1
            page_hint = (
                f"Page {self.page_num}/{self.total_pages}. "
                + ("next_page for next. " if has_next else "")
                + ("prev_page for prev. " if has_prev else "")
            )
            text = (
                f"User wants: \"{self.user_query}\"\n\n"
                f"Results ({page_hint}):\n\n{product_text}\n\n"
                "Use tools: select_product, next_page, prev_page, filter_price, filter_rating, recommend."
            )
        else:
            if self.current_detail:
                text = self._format_product_detail(self.current_detail)
            else:
                text = "No product detail available."
            viewed_n = len(self.viewed)
            viewed_pages = {int(v.get("page")) for v in self.viewed if isinstance(v, dict) and v.get("page") is not None}
            if viewed_n < 2:
                conf_hint = (
                    f"You have only viewed {viewed_n} product(s). Consider using back to compare more "
                    "before recommend, unless you are highly confident (>85%) this is the best match."
                )
            else:
                conf_hint = f"You have viewed {viewed_n} products. Use recommend if this is best, else back to compare more."
            if self.total_pages > 1 and len(viewed_pages) < 2:
                conf_hint += (
                    " Hard rule: before recommend, compare across at least 2 result pages. "
                    "Do back first, then next_page, inspect products there, and only then recommend."
                )
            text = (
                f"User wants: \"{self.user_query}\"\n\n"
                f"Product detail:\n\n{text}\n\n"
                f"{conf_hint} Use recommend to confirm, or back to return to search."
            )

        return Observation(
            data=text,
            modality="verbal",
            timestamp=time.time(),
            metadata={"prompt": text},
        )

    def get_available_actions(self) -> Dict[str, Any]:
        return {
            "context": self.context,
            "num_products": len(self.products),
            "page_num": self.page_num,
            "total_pages": self.total_pages,
            "has_next": self.page_num < self.total_pages,
            "has_prev": self.page_num > 1,
            "viewed_count": len(self.viewed),
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            "backend": "api",
            "context": self.context,
            "page_num": self.page_num,
            "total_pages": self.total_pages,
            "viewed_count": len(self.viewed),
            "current_product_id": self.current_product_id,
            "price_min": self.price_min,
            "price_max": self.price_max,
            "rating_min": self.rating_min,
        }

    def execute_action(self, tool_name: str, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        if tool_name == "select_product":
            ok, err = self.select_product(int(parameters.get("index", 0)))
            return ok, err, {}
        if tool_name == "next_page":
            ok, err = self.next_page()
            return ok, err, {}
        if tool_name == "prev_page":
            ok, err = self.prev_page()
            return ok, err, {}
        if tool_name == "filter_price":
            ok, err = self.filter_price(float(parameters.get("min", 0)), float(parameters.get("max", 0)))
            return ok, err, {}
        if tool_name == "filter_rating":
            ok, err = self.filter_rating(float(parameters.get("min", 0)))
            return ok, err, {}
        if tool_name == "back":
            ok, err = self.back()
            return ok, err, {}
        if tool_name == "recommend":
            current = self.get_current_product()
            if current:
                return True, None, {"recommended": True, "product_id": current["id"], "product_title": current["title"]}
            viewed = self.get_viewed_for_recommend()
            if viewed:
                v = viewed[-1]
                return True, None, {"recommended": True, "product_id": v["id"], "product_title": v.get("title", v["id"])}
            return False, "No product to recommend", {}
        return False, f"Unknown action: {tool_name}", {}

    def select_product(self, index: int) -> Tuple[bool, Optional[str]]:
        if self.context != "search":
            return False, "Not on search page"
        if index < 1 or index > len(self.products):
            return False, f"Invalid index {index}"
        selected = self.products[index - 1]
        product_id = selected["id"]
        self.log_fn("action", f"[Verbal] Fetching product detail: {product_id}")
        detail = self._api_product_detail(product_id)
        if not detail:
            detail = selected
        self.current_detail = detail
        self.current_product_id = product_id
        self.context = "detail"
        self.viewed.append(
            {
                "id": product_id,
                "title": detail.get("title", product_id),
                "index": len(self.viewed) + 1,
                "page": self.page_num,
            }
        )
        self.log_path_fn(f"p{self.page_num}:choose#{index}")
        return True, None

    def next_page(self) -> Tuple[bool, Optional[str]]:
        if self.context != "search":
            return False, "Not on search page"
        if self.page_num >= self.total_pages:
            return False, "Already on last page"
        self.page_num += 1
        self.log_path_fn(f"page{self.page_num}")
        self.log_fn("action", f"[Verbal] Moved to page {self.page_num}")
        data = self._api_search(
            self.search_keywords,
            page=self.page_num,
            price_min=self.price_min,
            price_max=self.price_max,
            rating_min=self.rating_min,
        )
        self.products = data["products"]
        return True, None

    def prev_page(self) -> Tuple[bool, Optional[str]]:
        if self.context != "search":
            return False, "Not on search page"
        if self.page_num <= 1:
            return False, "Already on first page"
        self.page_num -= 1
        self.log_path_fn(f"page{self.page_num}")
        self.log_fn("action", f"[Verbal] Moved back to page {self.page_num}")
        data = self._api_search(
            self.search_keywords,
            page=self.page_num,
            price_min=self.price_min,
            price_max=self.price_max,
            rating_min=self.rating_min,
        )
        self.products = data["products"]
        return True, None

    def filter_price(self, min_val: float, max_val: float) -> Tuple[bool, Optional[str]]:
        self.price_min = min_val
        self.price_max = max_val
        self.page_num = 1
        self.log_path_fn(f"filter_price:{min_val}-{max_val}")
        self.log_fn("action", f"[Verbal] Applied price filter ${min_val}-${max_val}")
        data = self._api_search(
            self.search_keywords,
            page=self.page_num,
            price_min=self.price_min,
            price_max=self.price_max,
            rating_min=self.rating_min,
        )
        self.products = data["products"]
        self.total_pages = data["total_pages"]
        return True, None

    def filter_rating(self, min_val: float) -> Tuple[bool, Optional[str]]:
        self.rating_min = min_val
        self.page_num = 1
        self.log_path_fn(f"filter_rating>={min_val}")
        self.log_fn("action", f"[Verbal] Applied rating filter >= {min_val}")
        data = self._api_search(
            self.search_keywords,
            page=self.page_num,
            price_min=self.price_min,
            price_max=self.price_max,
            rating_min=self.rating_min,
        )
        self.products = data["products"]
        self.total_pages = data["total_pages"]
        return True, None

    def back(self) -> Tuple[bool, Optional[str]]:
        if self.context != "detail":
            return False, "Not on detail page"
        self.context = "search"
        self.current_detail = None
        self.current_product_id = None
        data = self._api_search(
            self.search_keywords,
            page=self.page_num,
            price_min=self.price_min,
            price_max=self.price_max,
            rating_min=self.rating_min,
        )
        self.products = data["products"]
        self.log_path_fn("back")
        return True, None

    def get_viewed_for_recommend(self) -> List[Dict[str, Any]]:
        return list(self.viewed)

    def get_current_product(self) -> Optional[Dict[str, Any]]:
        if self.context != "detail" or not self.current_product_id:
            return None
        title = self.current_product_id
        if self.current_detail:
            title = self.current_detail.get("title", title)
        return {"id": self.current_product_id, "title": title}

    def close(self) -> None:
        pass
