"""
Shopping Browser Tools (Tool-First refactor)

Tools for the unified shopping agent. Each tool delegates to ShoppingEnv.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from aces.tools.base_tool import BaseTool
from aces.core.protocols import ToolResult


def _execute_with_search_page_recovery(
    env: Any,
    action: Callable[[], Tuple[bool, Optional[str]]],
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Execute a search-page action with lightweight context auto-recovery.

    Recovery strategy:
    1) If the first error is "Not on search page", retry once directly (handles stale state).
    2) If still failing, call back() once, then retry the original action.
    """
    ok, err = action()
    if ok or not (isinstance(err, str) and "Not on search page" in err):
        return ok, err, None

    # Fast retry for transient context mismatch.
    ok_retry, err_retry = action()
    if ok_retry:
        return True, None, "direct_retry"

    back = getattr(env, "back", None)
    if callable(back):
        try:
            back_ok, back_err = back()
        except Exception as exc:  # noqa: BLE001
            back_ok, back_err = False, f"{type(exc).__name__}: {exc}"
        if back_ok:
            ok_back_retry, err_back_retry = action()
            if ok_back_retry:
                return True, None, "back_then_retry"
            return False, err_back_retry, "back_then_retry_failed"
        # Keep original root cause, append recovery hint for easier debugging.
        if isinstance(back_err, str) and back_err:
            return False, f"{err_retry}; auto back failed: {back_err}", "back_failed"

    return False, err_retry, "not_recovered"


class SelectProductTool(BaseTool):
    """Select product by 1-based index to view details."""

    def __init__(self, env: Any):
        super().__init__(
            name="select_product",
            description="Select a product by its index (1-based) to view details. Use when you see a product that matches the user's need in the search results.",
            input_schema={
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Product index (1-based, e.g. 1 for first product)",
                    },
                },
                "required": ["index"],
                "additionalProperties": False,
            },
        )
        self._env = env

    def _coerce_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        p = super()._coerce_parameters(parameters or {})
        if "index" not in p:
            for alias in ("product_index", "productIndex", "number"):
                if alias in p:
                    p["index"] = p.pop(alias)
                    break
        return p

    def _execute_impl(self, parameters: Dict[str, Any]) -> Any:
        index = parameters.get("index")
        if index is None:
            return {"success": False, "error": "Missing required parameter: index"}
        success, err, recovered = _execute_with_search_page_recovery(
            self._env,
            lambda: self._env.select_product(index),
        )
        if not success:
            return {"success": False, "error": err}
        payload = {"success": True, "message": f"Opened product {index}"}
        if recovered:
            payload["auto_recovered"] = recovered
        return payload


class NextPageTool(BaseTool):
    """Go to next search results page."""

    def __init__(self, env: Any):
        super().__init__(
            name="next_page",
            description="Go to the next page of search results. Use when current page has no matching products.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )
        self._env = env

    def _execute_impl(self, parameters: Dict[str, Any]) -> Any:
        success, err, recovered = _execute_with_search_page_recovery(
            self._env,
            self._env.next_page,
        )
        if not success:
            return {"success": False, "error": err}
        payload = {"success": True, "message": "Navigated to next page"}
        if recovered:
            payload["auto_recovered"] = recovered
        return payload


class PrevPageTool(BaseTool):
    """Go to previous search results page."""

    def __init__(self, env: Any):
        super().__init__(
            name="prev_page",
            description="Go back to the previous page of search results. Use when you need to see earlier results.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )
        self._env = env

    def _execute_impl(self, parameters: Dict[str, Any]) -> Any:
        success, err, recovered = _execute_with_search_page_recovery(
            self._env,
            self._env.prev_page,
        )
        if not success:
            return {"success": False, "error": err}
        payload = {"success": True, "message": "Navigated to previous page"}
        if recovered:
            payload["auto_recovered"] = recovered
        return payload


class FilterPriceTool(BaseTool):
    """Apply price filter (min and max in USD)."""

    def __init__(self, env: Any):
        super().__init__(
            name="filter_price",
            description="Filter products by price range. Use when user mentions budget (e.g. 'under $50', 'between $20-$100').",
            input_schema={
                "type": "object",
                "properties": {
                    "min": {
                        "type": "number",
                        "description": "Minimum price in USD",
                    },
                    "max": {
                        "type": "number",
                        "description": "Maximum price in USD",
                    },
                },
                "additionalProperties": False,
            },
        )
        self._env = env

    def _execute_impl(self, parameters: Dict[str, Any]) -> Any:
        min_val = parameters.get("min")
        max_val = parameters.get("max")
        if min_val is None and max_val is None:
            return {"success": False, "error": "At least one of min/max is required"}
        # Keep one-sided filter semantics stable: do not inherit stale bounds.
        if min_val is None:
            min_val = 0.0
        if max_val is None:
            max_val = 999999.0
        min_val = float(min_val)
        max_val = float(max_val)
        if min_val > max_val:
            min_val, max_val = max_val, min_val
        success, err = self._env.filter_price(min_val, max_val)
        if not success:
            return {"success": False, "error": err}
        return {"success": True, "message": f"Filtered price ${min_val}-${max_val}"}


class FilterRatingTool(BaseTool):
    """Apply minimum rating filter."""

    def __init__(self, env: Any):
        super().__init__(
            name="filter_rating",
            description="Filter products by minimum star rating (e.g. 4 for 4+ stars).",
            input_schema={
                "type": "object",
                "properties": {
                    "min": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 5,
                        "description": "Minimum star rating",
                    },
                },
                "required": ["min"],
                "additionalProperties": False,
            },
        )
        self._env = env

    def _coerce_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        p = super()._coerce_parameters(parameters or {})
        if "min" in p:
            try:
                p["min"] = max(0.0, min(5.0, float(p["min"])))
            except Exception:
                pass
        return p

    def _execute_impl(self, parameters: Dict[str, Any]) -> Any:
        min_val = float(parameters["min"])
        success, err = self._env.filter_rating(min_val)
        if not success:
            return {"success": False, "error": err}
        return {"success": True, "message": f"Filtered rating >={min_val}"}


class BackTool(BaseTool):
    """Return from product detail page to search results."""

    def __init__(self, env: Any):
        super().__init__(
            name="back",
            description="Go back from product detail page to search results. Use when current product does not match user's need.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )
        self._env = env

    def _execute_impl(self, parameters: Dict[str, Any]) -> Any:
        success, err = self._env.back()
        if not success:
            # Make back() idempotent for agent robustness:
            # repeated back calls while already on search page should not be treated as hard tool failures.
            if (err or "").strip().lower() == "not on detail page":
                return {"success": True, "message": "Already on search results"}
            return {"success": False, "error": err}
        return {"success": True, "message": "Returned to search results"}


class RecommendTool(BaseTool):
    """
    Recommend a product and finish the task.
    Use product_id when on detail page, or viewed_index (1-based) to recommend from previously viewed list.
    """

    def __init__(self, env: Any):
        super().__init__(
            name="recommend",
            description="Recommend a product and finish. Only use when confident (>85%) it's the best match. "
            "Prefer browsing 2+ products before recommending. Return an ordered Top-3 slate (primary + 2 backups). "
            "Use product_ids (ordered list) when available. Legacy fields product_id/viewed_index are still accepted.",
            input_schema={
                "type": "object",
                "properties": {
                    "product_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 3,
                        "description": "Ordered Top-N product IDs. Target length is exactly 3.",
                    },
                    "product_id": {
                        "type": "string",
                        "description": "Product ID to recommend (e.g. smart_watch_3), NOT the product title. Use the ID shown in search results like 'ID:smart_watch_3'.",
                    },
                    "viewed_index": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "1-based index into viewed products list (when on search page)",
                    },
                },
                "additionalProperties": False,
            },
        )
        self._env = env

    def _candidate_pool(self) -> List[Dict[str, Any]]:
        pool: List[Dict[str, Any]] = []
        seen = set()

        def _push(pid: Any, title: Any = None) -> None:
            pid_s = str(pid or "").strip()
            if not pid_s or pid_s in seen:
                return
            seen.add(pid_s)
            title_s = str(title or "").strip() or pid_s
            pool.append({"id": pid_s, "title": title_s})

        current = self._env.get_current_product()
        if current and current.get("id"):
            _push(current.get("id"), current.get("title"))
        # Newest viewed first (recent comparisons usually carry stronger context).
        for item in reversed(self._env.get_viewed_for_recommend() or []):
            _push(item.get("id"), item.get("title"))

        # Also include products visible on current search page (if exposed by env),
        # so recommend(product_ids=[...]) can reference observed-but-not-opened items.
        visible_products = getattr(self._env, "products", None)
        if isinstance(visible_products, list):
            for item in visible_products:
                if isinstance(item, dict):
                    _push(item.get("id"), item.get("title"))

        # Playwright env uses `product_links` as [(product_id, detail_url), ...].
        visible_links = getattr(self._env, "product_links", None)
        if isinstance(visible_links, list):
            for item in visible_links:
                if isinstance(item, (list, tuple)) and item:
                    _push(item[0], item[0])

        # MCP env wraps API env and keeps visible products there.
        api_env = getattr(self._env, "api_env", None)
        api_visible = getattr(api_env, "products", None)
        if isinstance(api_visible, list):
            for item in api_visible:
                if isinstance(item, dict):
                    _push(item.get("id"), item.get("title"))
        return pool

    @staticmethod
    def _int_or_default(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _target_recommendation_count(self) -> int:
        # Keep backward compatibility: default top-3 unless runtime overrides via env.
        env_val = self._int_or_default(os.getenv("ACES_RECOMMENDATION_COUNT", "3"), 3)
        return max(1, min(3, env_val))

    def _exploration_snapshot(self) -> Dict[str, Any]:
        viewed = self._env.get_viewed_for_recommend() or []
        seen_ids = set()
        pages = set()
        has_page_info = False
        for item in viewed:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("id") or "").strip()
            if pid:
                seen_ids.add(pid)
            if "page" in item and item.get("page") is not None:
                has_page_info = True
                try:
                    pages.add(int(item.get("page")))
                except Exception:
                    pass

        state = {}
        if hasattr(self._env, "get_state"):
            try:
                state = self._env.get_state() or {}
            except Exception:
                state = {}
        total_pages = max(1, self._int_or_default(state.get("total_pages", 1), 1))
        return {
            "viewed_unique": len(seen_ids),
            "viewed_pages": len(pages),
            "has_page_info": has_page_info,
            "total_pages": total_pages,
        }

    def _attempt_auto_cross_page_exploration(self) -> str:
        """
        Best-effort recovery when recommend is called before cross-page exploration.
        This reduces repeated dead-loop errors by nudging env to the next page.
        """
        notes: List[str] = []
        state = {}
        if hasattr(self._env, "get_state"):
            try:
                state = self._env.get_state() or {}
            except Exception:
                state = {}

        context = str(state.get("context") or "")
        page_num = self._int_or_default(state.get("page_num", 1), 1)
        total_pages = max(1, self._int_or_default(state.get("total_pages", 1), 1))

        if context == "detail" and hasattr(self._env, "back"):
            try:
                ok, err = self._env.back()
                if ok:
                    notes.append("auto back")
                elif err:
                    notes.append(f"back_failed={err}")
            except Exception as exc:  # noqa: BLE001
                notes.append(f"back_failed={type(exc).__name__}")

        # Refresh paging state after back().
        if hasattr(self._env, "get_state"):
            try:
                state = self._env.get_state() or {}
                page_num = self._int_or_default(state.get("page_num", page_num), page_num)
                total_pages = max(1, self._int_or_default(state.get("total_pages", total_pages), total_pages))
            except Exception:
                pass

        if page_num < total_pages and hasattr(self._env, "next_page"):
            try:
                ok, err = self._env.next_page()
                if ok:
                    notes.append("auto next_page")
                elif err:
                    notes.append(f"next_page_failed={err}")
            except Exception as exc:  # noqa: BLE001
                notes.append(f"next_page_failed={type(exc).__name__}")

        return ", ".join(notes)

    def _resolve_product_id(self, val: str, pool: List[Dict[str, Any]]) -> Optional[tuple]:
        """
        Resolve LLM input to (product_id, product_title). LLM may pass ID (smart_watch_3)
        or title (Garmin Forerunner 245...). Returns (id, title) or None.
        Reject ambiguous matches: short strings like "5"", "4"", "6'" (screen sizes) that
        would wrongly match numbers in titles (e.g. "5 ATM").
        """
        import re
        if not pool:
            return None
        val = (val or "").strip()
        if not val:
            return None
        # Exact ID match always ok
        for v in pool:
            if str(v["id"]) == val:
                return (str(v["id"]), v.get("title", str(v["id"])))
        # Reject ambiguous: looks like screen size (e.g. 5", 4", 6') or single digit
        if re.match(r'^\d+["\']?$', val) or len(val) <= 3:
            return None
        # Title match only for longer, ID-like strings (e.g. smart_watch_3) or clear titles
        for v in pool:
            title = str(v.get("title") or "")
            if title and (val in title.lower() or title.lower() in val.lower()):
                return (str(v["id"]), title)
        return None

    def _execute_impl(self, parameters: Dict[str, Any]) -> Any:
        target_n = self._target_recommendation_count()
        snapshot = self._exploration_snapshot()

        pool = self._candidate_pool()
        if not pool:
            return {"success": False, "error": "No product to recommend"}

        product_id_param = parameters.get("product_id")
        viewed_index = parameters.get("viewed_index")
        product_ids = parameters.get("product_ids")

        def _add_unique(target: List[tuple], pid: str, title: str):
            if not any(existing_pid == pid for existing_pid, _ in target):
                target.append((pid, title))

        ranked: List[tuple] = []

        if isinstance(product_ids, list):
            for item in product_ids:
                resolved = self._resolve_product_id(str(item), pool)
                if resolved:
                    _add_unique(ranked, resolved[0], resolved[1])
                else:
                    return {"success": False, "error": f"Unknown product_id in product_ids: {item}"}

        if product_id_param:
            resolved = self._resolve_product_id(str(product_id_param), pool)
            if resolved:
                _add_unique(ranked, resolved[0], resolved[1])

        if viewed_index is not None:
            try:
                idx = int(viewed_index)
            except Exception:
                return {"success": False, "error": f"Invalid viewed_index {viewed_index}"}
            viewed = self._env.get_viewed_for_recommend() or []
            if idx < 1 or idx > len(viewed):
                return {"success": False, "error": f"Invalid viewed_index {viewed_index}"}
            v = viewed[idx - 1]
            _add_unique(ranked, str(v.get("id")), str(v.get("title") or v.get("id")))

        # Fill remaining slots from candidate pool.
        for item in pool:
            if len(ranked) >= 3:
                break
            _add_unique(ranked, str(item["id"]), str(item.get("title") or item["id"]))

        if len(ranked) < target_n:
            return {
                "success": False,
                "error": f"Need at least {target_n} distinct candidates before final recommend",
            }

        recommendations = [
            {"rank": i + 1, "product_id": pid, "product_title": title}
            for i, (pid, title) in enumerate(ranked[:target_n])
        ]
        top1 = recommendations[0]
        return {
            "success": True,
            "recommended": True,
            # Backward-compatible top-1 fields:
            "product_id": top1["product_id"],
            "product_title": top1["product_title"],
            # New ordered slate payload:
            "recommendations": recommendations,
        }


def create_shopping_browser_tools(env: Any) -> List[BaseTool]:
    """Create all shopping browser tools with env injected."""
    return [
        SelectProductTool(env),
        NextPageTool(env),
        PrevPageTool(env),
        FilterPriceTool(env),
        FilterRatingTool(env),
        BackTool(env),
        RecommendTool(env),
    ]
