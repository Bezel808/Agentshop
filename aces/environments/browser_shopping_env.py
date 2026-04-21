"""
Browser Shopping Environment

Playwright-based environment for visual mode. Captures screenshots
and provides DOM parsing for product links and descriptions.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from aces.core.protocols import Observation


class BrowserShoppingEnv:
    """
    Shopping environment using Playwright for browser interaction.
    Used in visual mode - provides screenshot observations.
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
        push_screenshot_fn: Optional[Callable[[bytes, str], None]] = None,
        log_path_fn: Optional[Callable[[str], None]] = None,
        api_search_fn: Optional[Callable[..., Dict]] = None,
    ):
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
        self._api_search_fn = api_search_fn

        self.playwright = None
        self.browser = None
        self.page = None
        self.search_keywords = ""

        self.context = "search"
        self.search_url = ""
        self.product_links: List[Tuple[str, str]] = []
        self.viewed: List[Dict[str, Any]] = []
        self.pages_seen: List[int] = []
        self.total_pages = 1
        self.current_product_id: Optional[str] = None
        self.current_detail_url: Optional[str] = None
        self.detail_steps = 0

    def _build_search_url(
        self,
        keywords: str,
        page: int = 1,
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
        rating_min: Optional[float] = None,
    ) -> str:
        from urllib.parse import quote
        url = f"{self.web_server_url}/search?q={quote(keywords)}&page={page}&page_size=8"
        if self.condition_name:
            url += f"&condition_name={quote(self.condition_name)}"
        if price_min is not None and price_min > 0:
            url += f"&price_min={price_min}"
        if price_max is not None and price_max > 0:
            url += f"&price_max={price_max}"
        if rating_min is not None and rating_min > 0:
            url += f"&rating_min={rating_min}"
        return url

    def init_browser(self) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise ImportError("pip install playwright && playwright install")
        self.log_fn("action", "Initializing browser...")
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox"],
        )
        self.page = self.browser.new_page(viewport={"width": 1280, "height": 800})
        self.log_fn("action", "Browser started.")

    def _navigate_and_capture(self, url: str, full_page: bool = True) -> bytes:
        self.log_fn("action", f"Navigating to: {url}")
        self.page.goto(url, wait_until="networkidle")
        time.sleep(1.5)
        screenshot_bytes = self.page.screenshot(type="png", full_page=full_page)
        self.log_fn("action", f"Screenshot captured ({len(screenshot_bytes)/1024:.1f} KB)")
        self.push_screenshot_fn(screenshot_bytes, url)
        return screenshot_bytes

    def _get_product_detail_links(self) -> List[Tuple[str, str]]:
        try:
            hrefs = self.page.evaluate("""
                () => Array.from(document.querySelectorAll('a[href^="/product/"]'))
                    .map(a => a.getAttribute('href'))
                    .filter(Boolean)
            """)
        except Exception as e:
            self.log_fn("error", f"Failed to parse product links: {e}")
            return []
        seen = set()
        result = []
        for href in (hrefs or []):
            path = href.split("?")[0]
            pid = path.rstrip("/").split("/")[-1]
            if pid and pid not in seen:
                seen.add(pid)
                result.append((pid, self.web_server_url + href))
        return result

    def _get_description_from_detail_page(self) -> str:
        try:
            loc = self.page.locator(".detail-description .text").first
            if loc.count() == 0:
                return "(Description section not found on page)"
            text = loc.inner_text(timeout=2000).strip()
            return text or "No description available."
        except Exception as e:
            self.log_fn("error", f"Failed to extract description: {e}")
            return "(Description extraction failed)"

    def reset(self, keywords: str) -> Observation:
        self.search_keywords = keywords
        self.context = "search"
        self.detail_steps = 0
        self.current_product_id = None
        self.current_detail_url = None

        if self.browser is None:
            self.init_browser()

        self.search_url = self._build_search_url(
            keywords,
            page=self.page_num,
            price_min=self.price_min,
            price_max=self.price_max,
            rating_min=self.rating_min,
        )
        self.log_fn("thinking", "Preparing to open the search results page...")
        self._navigate_and_capture(self.search_url)
        self.product_links = self._get_product_detail_links()

        if self._api_search_fn:
            meta = self._api_search_fn(
                keywords, limit=8, page=1,
                price_min=self.price_min,
                price_max=self.price_max,
                rating_min=self.rating_min,
            )
            self.total_pages = meta.get("total_pages", 1)
        else:
            self.total_pages = 1

        screenshot = self.page.screenshot(type="png", full_page=True)
        self.push_screenshot_fn(screenshot, self.page.url)

        import base64
        base64_img = base64.b64encode(screenshot).decode("utf-8")
        data_url = f"data:image/png;base64,{base64_img}"

        prompt = self._build_search_prompt()
        return Observation(
            data=data_url,
            modality="visual",
            timestamp=time.time(),
            metadata={"prompt": prompt},
        )

    def _build_search_prompt(self) -> str:
        total_p = self.total_pages
        num_products = len(self.product_links)
        pages_hint = (
            f"\n[Important] Earlier result pages are usually more relevant to the search intent. "
            f"Current page: {self.page_num}/{total_p}."
        )
        if self.pages_seen:
            pages_hint += f" Pages already visited: {', '.join(str(p) for p in self.pages_seen)}."
        has_prev = self.page_num > 1
        prev_hint = "\n- If this page is clearly irrelevant, call prev_page to go back to earlier pages." if has_prev else ""
        filter_hint = ""
        if self.price_min is not None or self.price_max is not None:
            lo = self.price_min if self.price_min is not None else 0
            hi = self.price_max if self.price_max is not None else 999
            filter_hint = f" Active filter: price ${lo}-${hi}."
        viewed_ids = {v["id"] for v in self.viewed}
        pos_to_id = ", ".join(f"{i}={pid}" for i, (pid, _) in enumerate(self.product_links, 1))
        viewed_hint = ""
        if viewed_ids:
            viewed_hint = (
                f"\n[Already viewed product IDs] {', '.join(sorted(viewed_ids))} - do not click them again."
                f"\n[Current page position -> ID] {pos_to_id}"
            )
        return (
            f"[User need] {self.user_query}\n\n"
            f"[Current: Search results page {self.page_num}/{total_p}] {num_products} products on this page."
            f"{pages_hint}{prev_hint}{filter_hint}{viewed_hint}\n\n"
            "Judge products by thumbnail/image in the screenshot and click only visually relevant items.\n"
            "[Output constraint] Structured tool calls only. Do not output natural-language action descriptions.\n"
            "[Tool names must match exactly] select_product, next_page, prev_page, filter_price, filter_rating, back, recommend.\n"
            "[Suggested actions] select_product(1-{n}) to inspect, next_page/prev_page to browse pages, "
            "filter_price/filter_rating to filter, recommend for the final choice."
        ).format(n=num_products or 1)

    def _build_detail_prompt(self) -> str:
        desc = self._get_description_from_detail_page()
        desc_short = desc[:500] + "..." if len(desc) > 500 else desc
        viewed_n = len(self.viewed)
        viewed_pages = {int(v.get("page")) for v in self.viewed if isinstance(v, dict) and v.get("page") is not None}
        confidence_hint = ""
        if viewed_n < 2:
            confidence_hint = (
                f"\n[Confidence hint] You have only viewed {viewed_n} product(s). Use back to compare more "
                "before recommend, unless you are highly confident (>85%) this is the best match."
            )
        else:
            confidence_hint = (
                f"\n[Viewed {viewed_n} products] If this is best, use recommend. "
                "Otherwise use back to continue comparison."
            )
        if self.total_pages > 1 and len(viewed_pages) < 2:
            confidence_hint += (
                "\n[Hard rule] You have only covered one result page. Before recommend, compare across at least "
                "2 pages: back -> next_page -> inspect products there -> then recommend."
            )
        return (
            f"[User need] {self.user_query}\n\n"
            "[Current: Product detail page] Description: " + desc_short + "\n\n"
            "[Output constraint] Structured tool calls only. Do not output natural-language action descriptions.\n"
            f"Available tools: recommend (only with high confidence), back (return to search).{confidence_hint}"
        )

    def get_observation(self) -> Observation:
        screenshot = self.page.screenshot(type="png", full_page=True)
        self.push_screenshot_fn(screenshot, self.page.url)
        import base64
        base64_img = base64.b64encode(screenshot).decode("utf-8")
        data_url = f"data:image/png;base64,{base64_img}"

        if self.context == "search":
            prompt = self._build_search_prompt()
        else:
            prompt = self._build_detail_prompt()

        return Observation(
            data=data_url,
            modality="visual",
            timestamp=time.time(),
            metadata={"prompt": prompt},
        )

    def get_available_actions(self) -> Dict[str, Any]:
        return {
            "context": self.context,
            "num_products": len(self.product_links),
            "page_num": self.page_num,
            "total_pages": self.total_pages,
            "has_next": self.page_num < self.total_pages,
            "has_prev": self.page_num > 1,
            "viewed_ids": [v["id"] for v in self.viewed],
            "viewed_count": len(self.viewed),
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            "backend": "playwright",
            "context": self.context,
            "page_num": self.page_num,
            "total_pages": self.total_pages,
            "viewed_count": len(self.viewed),
            "current_product_id": self.current_product_id,
            "current_url": self.page.url if self.page else None,
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
        if index < 1 or index > len(self.product_links):
            return False, f"Invalid index {index}, max {len(self.product_links)}"
        product_id, detail_url = self.product_links[index - 1]
        viewed_ids = {v["id"] for v in self.viewed}
        if product_id in viewed_ids:
            return False, f"Product {product_id} already viewed"
        if self.page_num not in self.pages_seen:
            self.pages_seen.append(self.page_num)
        self.log_fn("action", f"Opening product detail #{index}: {product_id}")
        self._navigate_and_capture(detail_url)
        self.context = "detail"
        self.current_product_id = product_id
        self.current_detail_url = detail_url
        self.detail_steps = 0
        self.log_path_fn(f"p{self.page_num}:view#{product_id}")
        self.viewed.append(
            {
                "id": product_id,
                "title": product_id,
                "index": len(self.viewed) + 1,
                "page": self.page_num,
            }
        )
        return True, None

    def next_page(self) -> Tuple[bool, Optional[str]]:
        if self.context != "search":
            return False, "Not on search page"
        if self.page_num >= self.total_pages:
            return False, "Already on last page"
        if self.page_num not in self.pages_seen:
            self.pages_seen.append(self.page_num)
        self.page_num += 1
        self.log_path_fn(f"page{self.page_num}")
        self.log_fn("action", f"Moved to page {self.page_num}")
        self.search_url = self._build_search_url(
            self.search_keywords,
            page=self.page_num,
            price_min=self.price_min,
            price_max=self.price_max,
            rating_min=self.rating_min,
        )
        self._navigate_and_capture(self.search_url)
        self.product_links = self._get_product_detail_links()
        return True, None

    def prev_page(self) -> Tuple[bool, Optional[str]]:
        if self.context != "search":
            return False, "Not on search page"
        if self.page_num <= 1:
            return False, "Already on first page"
        self.page_num -= 1
        self.log_path_fn(f"page{self.page_num}")
        self.log_fn("action", f"Moved back to page {self.page_num}")
        self.search_url = self._build_search_url(
            self.search_keywords,
            page=self.page_num,
            price_min=self.price_min,
            price_max=self.price_max,
            rating_min=self.rating_min,
        )
        self._navigate_and_capture(self.search_url)
        self.product_links = self._get_product_detail_links()
        return True, None

    def filter_price(self, min_val: float, max_val: float) -> Tuple[bool, Optional[str]]:
        self.price_min = min_val
        self.price_max = max_val
        self.page_num = 1
        self.log_path_fn(f"filter_price:{min_val}-{max_val}")
        self.log_fn("action", f"Applied price filter ${min_val}-${max_val}")
        self.search_url = self._build_search_url(
            self.search_keywords,
            page=self.page_num,
            price_min=self.price_min,
            price_max=self.price_max,
            rating_min=self.rating_min,
        )
        self._navigate_and_capture(self.search_url)
        self.product_links = self._get_product_detail_links()
        return True, None

    def filter_rating(self, min_val: float) -> Tuple[bool, Optional[str]]:
        self.rating_min = min_val
        self.page_num = 1
        self.log_path_fn(f"filter_rating>={min_val}")
        self.log_fn("action", f"Applied rating filter >= {min_val}")
        self.search_url = self._build_search_url(
            self.search_keywords,
            page=self.page_num,
            price_min=self.price_min,
            price_max=self.price_max,
            rating_min=self.rating_min,
        )
        self._navigate_and_capture(self.search_url)
        self.product_links = self._get_product_detail_links()
        return True, None

    def back(self) -> Tuple[bool, Optional[str]]:
        if self.context != "detail":
            return False, "Not on detail page"
        self.log_path_fn("back")
        self.log_fn("action", "Returning to search results page")
        self._navigate_and_capture(self.search_url)
        self.context = "search"
        self.current_product_id = None
        self.current_detail_url = None
        self.detail_steps = 0
        return True, None

    def get_viewed_for_recommend(self) -> List[Dict[str, Any]]:
        return list(self.viewed)

    def get_current_product(self) -> Optional[Dict[str, Any]]:
        if self.context != "detail" or not self.current_product_id:
            return None
        return {"id": self.current_product_id, "title": self.current_product_id}

    def close(self) -> None:
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
        self.log_fn("action", "Browser closed")
