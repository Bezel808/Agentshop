#!/usr/bin/env python3
"""
ACES-v2 unified agent entrypoint (Tool-First).

Flow:
1) Parse user intent with VLM/LLM
2) Run in verbal or visual perception mode
3) Browse multi-page marketplace via structured tool calling
4) Output final recommendation(s)
"""

import os
import sys
import time
import base64
import requests
import json
import ast
import re
import random
from pathlib import Path
from urllib.parse import quote
from typing import List, Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

from aces.agents import ComposableAgent
from aces.llm_backends import OpenAIBackend, QwenBackend, KimiBackend
from aces.perception import VisualPerception, VerbalPerception
from aces.core.protocols import Message
from aces.core.protocols import Action, ToolResult, Observation
from aces.environments import BrowserShoppingEnv, APIShoppingEnv, MCPShoppingEnv
from aces.tools import ToolFactory, HTTPMCPCaller
from aces.orchestration import AgentOrchestrator, OrchestratorEvent, OrchestratorResult
from aces.config.prompt_profiles import get_prompt_profile, PROMPT_PROFILES


def _normalize_recommendation_count(value: Optional[int], default: int = 1) -> int:
    try:
        n = int(value if value is not None else default)
    except Exception:
        n = int(default)
    return max(1, min(3, n))


def _build_system_prompt(
    profile: Dict[str, Any],
    recommendation_count: int,
    *,
    budget_hint: str = "",
) -> str:
    threshold = profile["confidence_threshold"]
    min_viewed = profile["min_viewed_before_recommend"]
    rec_count = _normalize_recommendation_count(recommendation_count, default=1)
    if rec_count <= 1:
        final_rule = (
            "Final output rule: recommend a single best product (Top-1 primary). "
            "Do not force backup candidates in the final action."
        )
    else:
        final_rule = (
            f"Final output rule: recommend should return an ordered Top-{rec_count} slate "
            "(primary + backups). "
            f"Pass product_ids as an ordered list of {rec_count} product IDs whenever possible."
        )
    return (
        "You are a shopping assistant in STRICT tool-calling mode. "
        "For every turn, output tool calls only. Do NOT output natural-language actions, explanations, or chat. "
        "Use exact tool names only: select_product, next_page, prev_page, filter_price, filter_rating, back, recommend. "
        "Never output aliases like 'next', 'prev', 'click', or 'reply'. "
        "If reasoning text is present in your tool call payload, keep it in English only. "
        "In search results: use select_product to open a product, next_page/prev_page to browse pages, "
        "filter_price/filter_rating to refine results. "
        "On product detail page: use recommend to confirm or back to return and compare more. "
        "Before calling recommend, gather enough concrete evidence (price, rating, feature fit) "
        "so the final recommendation rationale can be detailed and well-supported. "
        f"Confidence rule: call recommend only when confidence > {threshold}%. "
        f"If viewed products < {min_viewed}, prefer back/filters unless evidence is overwhelming. "
        + final_rule
        + " "
        "If needed, use viewed_index to anchor the primary candidate and let the tool fill backups. "
        "For tools without parameters, call with an empty object. "
        + (budget_hint + " " if budget_hint else "")
        + profile.get("system_suffix", "")
    )


class LiveBrowserAgent:
    """
    Live browser shopping agent (Tool-First).
    Supports both visual and verbal modes with structured tool-calling.
    """

    def __init__(
        self,
        llm_api_key: str,
        llm_backend: str = "qwen",
        perception_mode: str = "visual",
        web_server_url: str = "http://localhost:5000",
        user_query: str = "mousepad",
        condition_name: str = None,
        stay_open: bool = True,
        page: int = 1,
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
        rating_min: Optional[float] = None,
        temperature: float = 0.7,
        env_backend: Optional[str] = None,
        legacy_fallback: bool = False,
        trace_output: Optional[str] = None,
        mcp_endpoint: Optional[str] = None,
        prompt_profile: str = "robust_compare",
        max_steps: int = 40,
        max_repeated_actions: int = 6,
        max_errors: int = 8,
        max_history: int = 12,
        verbal_use_vlm: bool = False,
        confidence_threshold: Optional[int] = None,
        min_viewed_before_recommend: Optional[int] = None,
        recommendation_count: Optional[int] = None,
    ):
        self.web_server_url = web_server_url
        self.user_query = user_query
        self.condition_name = condition_name
        self.perception_mode = perception_mode
        self.stay_open = stay_open
        self.search_keywords: Optional[str] = None
        self.page_num = page
        self.price_min = price_min
        self.price_max = price_max
        self.rating_min = rating_min
        self.env_backend = env_backend
        self.trace_output = Path(trace_output) if trace_output else None
        self.trace_events: List[Dict[str, Any]] = []
        self.prompt_profile = get_prompt_profile(prompt_profile).copy()
        if confidence_threshold is not None:
            self.prompt_profile["confidence_threshold"] = int(confidence_threshold)
        if min_viewed_before_recommend is not None:
            self.prompt_profile["min_viewed_before_recommend"] = int(min_viewed_before_recommend)
        self.last_recommendation: Optional[Dict[str, str]] = None
        self.last_recommendations: List[Dict[str, Any]] = []
        self.last_recommendation_reason: str = ""
        self.max_steps = int(max_steps)
        self.max_repeated_actions = int(max_repeated_actions)
        self.max_errors = int(max_errors)
        self.max_history = int(max_history)
        self.verbal_use_vlm = bool(verbal_use_vlm)
        self.viewer_token = (os.getenv("ACES_VIEWER_TOKEN") or "").strip()
        env_recommendation_count = os.getenv("ACES_RECOMMENDATION_COUNT", "1")
        self.recommendation_count = _normalize_recommendation_count(
            recommendation_count if recommendation_count is not None else env_recommendation_count,
            default=1,
        )
        # Keep tool-level recommendation behavior aligned with runtime target count.
        os.environ["ACES_RECOMMENDATION_COUNT"] = str(self.recommendation_count)

        explicit_price_min = price_min
        explicit_price_max = price_max

        # Fallback budget intent from regex if CLI bounds are not explicitly set.
        budget = self._extract_budget_constraints(user_query)
        if price_min is None and budget.get("price_min") is not None:
            self.price_min = budget["price_min"]
        if price_max is None and budget.get("price_max") is not None:
            self.price_max = budget["price_max"]

        if llm_backend == "qwen":
            if perception_mode == "visual" or (perception_mode == "verbal" and self.verbal_use_vlm):
                llm = QwenBackend(model="qwen-vl-plus", api_key=llm_api_key, temperature=temperature)
            else:
                llm = QwenBackend(model="qwen-plus", api_key=llm_api_key, temperature=temperature)
        elif llm_backend == "openai":
            llm = OpenAIBackend(model="gpt-4o", api_key=llm_api_key, temperature=temperature)
        elif llm_backend == "kimi":
            if perception_mode == "visual" or (perception_mode == "verbal" and self.verbal_use_vlm):
                llm = KimiBackend(model="moonshot-v1-8k-vision-preview", api_key=llm_api_key, temperature=temperature)
            else:
                llm = KimiBackend(model="moonshot-v1-8k", api_key=llm_api_key, temperature=temperature)
        else:
            llm = OpenAIBackend(model="gpt-4o", api_key=llm_api_key, temperature=temperature)

        # Primary path: let LLM separate search intent and budget in a structured way.
        self.query_intent = self._extract_intent_with_llm(llm, self.user_query)
        llm_min = self.query_intent.get("price_min")
        llm_max = self.query_intent.get("price_max")
        if explicit_price_min is None and llm_min is not None:
            self.price_min = llm_min
        if explicit_price_max is None and llm_max is not None:
            self.price_max = llm_max

        self.has_budget_constraint = self.price_min is not None or self.price_max is not None
        budget_hint = ""
        if self.has_budget_constraint:
            min_text = "none" if self.price_min is None else f"{self.price_min:g}"
            max_text = "none" if self.price_max is None else f"{self.price_max:g}"
            budget_hint = (
                "Budget-first rule: the user has explicit budget constraints. "
                f"Prioritize applying filter_price early on search page with min={min_text}, max={max_text} "
                "before broad exploration, then continue comparing qualified products."
            )

        perception = VisualPerception() if perception_mode == "visual" else VerbalPerception()

        # Create env based on perception mode
        backend = env_backend or ("playwright" if perception_mode == "visual" else "api")
        self.env_backend = backend

        if backend == "playwright":
            self.env = BrowserShoppingEnv(
                web_server_url=web_server_url,
                user_query=user_query,
                condition_name=condition_name,
                page_num=page,
                price_min=self.price_min,
                price_max=self.price_max,
                rating_min=self.rating_min,
                log_fn=self.log,
                push_screenshot_fn=self.push_screenshot,
                log_path_fn=self._log_path,
                api_search_fn=self._api_search,
            )
        elif backend == "api":
            self.env = APIShoppingEnv(
                web_server_url=web_server_url,
                user_query=user_query,
                condition_name=condition_name,
                page_num=page,
                price_min=self.price_min,
                price_max=self.price_max,
                rating_min=self.rating_min,
                log_fn=self.log,
                log_path_fn=self._log_path,
            )
        elif backend == "mcp":
            # Keep MCP optional; user can inject caller at runtime by extending this script.
            # Here we support env shape but fail fast if caller is unavailable.
            mcp_caller = HTTPMCPCaller(mcp_endpoint) if mcp_endpoint else None
            self.env = MCPShoppingEnv(
                mcp_caller=mcp_caller,
                web_server_url=web_server_url,
                user_query=user_query,
                condition_name=condition_name,
                page_num=page,
                price_min=self.price_min,
                price_max=self.price_max,
                rating_min=self.rating_min,
                log_fn=self.log,
                push_screenshot_fn=self.push_screenshot,
                log_path_fn=self._log_path,
            )
        else:
            raise ValueError(f"Unsupported env backend: {backend}")

        tools = ToolFactory.create_shopping_browser_tools(self.env)
        self.agent = ComposableAgent(
            llm=llm,
            perception=perception,
            tools=tools,
            system_prompt=_build_system_prompt(
                self.prompt_profile,
                self.recommendation_count,
                budget_hint=budget_hint,
            ),
            max_history=self.max_history,
            legacy_fallback=legacy_fallback,
        )

        self.decision_path: List[str] = []

    def _publish_no_budget_match(self, source: str = "budget_filter") -> None:
        message = "No products match the current budget constraints. Please widen the price range and try again."
        self.last_recommendation_reason = message
        self._log_path("no_match_budget")
        self.log("action", f"[Decision path] {self._summary_path()}")
        self.push_to_viewer("decision_path", {"summary": self._summary_path(), "path": self.decision_path})
        self.push_to_viewer("metric", {"name": "step", "value": "No budget match"})
        self.log("action", message)
        fallback_url = f"/search?q={quote(self.search_keywords or self.user_query)}&page=1&page_size=8"
        if self.price_min is not None and self.price_min > 0:
            fallback_url += f"&price_min={self.price_min}"
        if self.price_max is not None and self.price_max > 0:
            fallback_url += f"&price_max={self.price_max}"
        self.push_to_viewer(
            "recommend",
            {
                "product_id": "NO_MATCH",
                "product_title": message,
                "product_url": fallback_url,
                "reason": message,
                "source": source,
                "recommendations": [
                    {
                        "rank": 1,
                        "product_id": "NO_MATCH",
                        "product_title": message,
                        "product_url": fallback_url,
                        "price": None,
                        "rating": None,
                        "rating_count": None,
                        "image_url": "",
                    }
                ],
            },
        )
        self.last_recommendation = {
            "product_id": "NO_MATCH",
            "product_title": message,
            "product_url": fallback_url,
            "reason": message,
        }
        self.last_recommendations = [
            {
                "rank": 1,
                "product_id": "NO_MATCH",
                "product_title": message,
                "product_url": fallback_url,
                "price": None,
                "rating": None,
                "rating_count": None,
                "image_url": "",
            }
        ]

    @staticmethod
    def _extract_budget_constraints(query: str) -> Dict[str, Optional[float]]:
        q = (query or "").lower()
        price_min: Optional[float] = None
        price_max: Optional[float] = None

        m_between = re.search(r"(?:between|from)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:-|to|and)\s*\$?\s*(\d+(?:\.\d+)?)", q)
        if m_between:
            a = float(m_between.group(1))
            b = float(m_between.group(2))
            price_min, price_max = (a, b) if a <= b else (b, a)

        m_under = re.search(r"(?:under|below|less than|<=?)\s*\$?\s*(\d+(?:\.\d+)?)", q)
        if m_under:
            price_max = float(m_under.group(1))

        m_above = re.search(r"(?:over|above|more than|>=?)\s*\$?\s*(\d+(?:\.\d+)?)", q)
        if m_above:
            price_min = float(m_above.group(1))

        return {"price_min": price_min, "price_max": price_max}

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        raw = (text or "").strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        return raw

    @classmethod
    def _parse_intent_json(cls, raw: str) -> Dict[str, Any]:
        txt = cls._strip_code_fences(raw)
        parsed: Dict[str, Any] = {}
        try:
            loaded = json.loads(txt)
            if isinstance(loaded, dict):
                parsed = loaded
        except Exception:
            try:
                loaded = ast.literal_eval(txt)
                if isinstance(loaded, dict):
                    parsed = loaded
            except Exception:
                parsed = {}

        def _to_float_or_none(v: Any) -> Optional[float]:
            if v is None:
                return None
            try:
                return float(v)
            except Exception:
                return None

        keywords = str(parsed.get("keywords", "")).strip()
        price_min = _to_float_or_none(parsed.get("price_min"))
        price_max = _to_float_or_none(parsed.get("price_max"))
        if price_min is not None and price_max is not None and price_min > price_max:
            price_min, price_max = price_max, price_min
        if keywords:
            keywords = cls._strip_budget_from_keywords(keywords)
        return {"keywords": keywords, "price_min": price_min, "price_max": price_max}

    def _extract_intent_with_llm(self, llm: Any, query: str) -> Dict[str, Any]:
        prompt = (
            "You are parsing a shopping request into structured fields.\n"
            "Return ONLY one JSON object with exactly these keys:\n"
            "{\"keywords\": string, \"price_min\": number|null, \"price_max\": number|null}\n\n"
            "Rules:\n"
            "- `keywords` must describe product intent and key attributes only.\n"
            "- Never include budget/price words or numbers in `keywords`.\n"
            "- Put budget constraints only into `price_min` / `price_max`.\n"
            "- If user says 'under X', set price_max=X.\n"
            "- If user says 'over X', set price_min=X.\n"
            "- If budget is absent, set both to null.\n\n"
            f"User request: {query}\n"
        )
        try:
            messages = [
                Message(role="system", content="You output strict JSON only."),
                Message(role="user", content=prompt),
            ]
            resp = llm.generate(messages=messages, tools=None)
            raw = resp.content if isinstance(resp.content, str) else str(resp.content)
            parsed = self._parse_intent_json(raw)
            if parsed.get("keywords"):
                return parsed
        except Exception:
            pass

        # Fallback to deterministic parse when LLM extraction fails.
        budget = self._extract_budget_constraints(query)
        keywords = self._strip_budget_from_keywords(query)
        return {
            "keywords": keywords,
            "price_min": budget.get("price_min"),
            "price_max": budget.get("price_max"),
        }

    @staticmethod
    def _strip_budget_from_keywords(text: str) -> str:
        s = (text or "").strip()
        if not s:
            return s
        # Remove common budget phrases so price constraints are handled by filter_price only.
        patterns = [
            r"\b\d+(?:\.\d+)?\s*(?:-|to|~|–|—)\s*\d+(?:\.\d+)?\b",
            r"\b(?:under|below|less than|up to|no more than)\s*\$?\s*\d+(?:\.\d+)?\b",
            r"\b(?:over|above|more than|at least)\s*\$?\s*\d+(?:\.\d+)?\b",
            r"\b(?:between|from)\s*\$?\s*\d+(?:\.\d+)?\s*(?:-|to|and)\s*\$?\s*\d+(?:\.\d+)?\b",
            r"\$\s*\d+(?:\.\d+)?\b",
            r"\b\d+(?:\.\d+)?\s*(?:dollars?|usd|rmb|yuan)\b",
            r"\b(?:dollars?|usd|rmb|yuan)\b",
            r"\b(?:budget|price|cost)\b",
        ]
        out = s
        for pat in patterns:
            out = re.sub(pat, " ", out, flags=re.IGNORECASE)
        out = re.sub(r"[,\.;:]+", " ", out)
        out = re.sub(r"\s+", " ", out).strip()
        return out

    def _log_path(self, step: str):
        self.decision_path.append(step)
        self.push_to_viewer("decision_path", {"step": step, "path": self.decision_path})

    def _summary_path(self) -> str:
        return " -> ".join(self.decision_path) if self.decision_path else "(none)"

    def push_to_viewer(self, data_type: str, data: dict):
        try:
            headers = {}
            if self.viewer_token:
                headers["X-ACES-Token"] = self.viewer_token
            requests.post(
                f"{self.web_server_url}/api/push",
                json={"type": data_type, **data},
                headers=headers,
                timeout=1,
            )
        except Exception:
            pass

    def log(self, level: str, message: str):
        print(f"[{level.upper()}] {message}")
        self.push_to_viewer("log", {"level": level, "message": message})

    def push_screenshot(self, screenshot_bytes: bytes, url: str):
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        screenshot_data = f"data:image/png;base64,{screenshot_base64}"
        self.push_to_viewer("screenshot", {"screenshot": screenshot_data, "url": url})

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
            self.log("error", f"API search request failed: {e}")
            return {"products": [], "page": 1, "total_pages": 1}

    def _api_product_detail(self, product_id: str) -> Optional[dict]:
        url = f"{self.web_server_url}/api/product/{quote(str(product_id), safe='')}"
        if self.condition_name:
            url += f"?condition_name={quote(self.condition_name)}"
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
            return data.get("product")
        except Exception as e:
            self.log("error", f"API product detail request failed ({product_id}): {e}")
            return None

    def _build_recommendation_cards(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cards: List[Dict[str, Any]] = []
        for idx, item in enumerate(recommendations, 1):
            product_id = str(item.get("product_id", "")).strip()
            if not product_id:
                continue
            product_title = str(item.get("product_title", "")).strip() or product_id
            detail = self._api_product_detail(product_id)
            product_url = f"/product/{quote(product_id, safe='')}"
            if detail:
                product_title = str(detail.get("title") or product_title)
                cards.append(
                    {
                        "rank": int(item.get("rank") or idx),
                        "product_id": product_id,
                        "product_title": product_title,
                        "product_url": product_url,
                        "price": detail.get("price"),
                        "rating": detail.get("rating"),
                        "rating_count": detail.get("rating_count"),
                        "image_url": detail.get("image_url"),
                    }
                )
            else:
                cards.append(
                    {
                        "rank": int(item.get("rank") or idx),
                        "product_id": product_id,
                        "product_title": product_title,
                        "product_url": product_url,
                        "price": None,
                        "rating": None,
                        "rating_count": None,
                        "image_url": "",
                    }
                )
        return cards

    def _normalize_recommendations_from_payload(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_recs = data.get("recommendations") if isinstance(data, dict) else None
        recommendations: List[Dict[str, Any]] = []
        if isinstance(raw_recs, list):
            for i, rec in enumerate(raw_recs, 1):
                if not isinstance(rec, dict):
                    continue
                pid = str(rec.get("product_id", "")).strip()
                if not pid:
                    continue
                title = str(rec.get("product_title", "")).strip() or pid
                recommendations.append(
                    {"rank": int(rec.get("rank") or i), "product_id": pid, "product_title": title}
                )
        if not recommendations:
            product_id = str(data.get("product_id", "unknown"))
            product_title = str(data.get("product_title", product_id))
            recommendations = [{"rank": 1, "product_id": product_id, "product_title": product_title}]
        return recommendations

    def _publish_recommendations(self, recommendations: List[Dict[str, Any]], source: str = "tool_call") -> None:
        if not recommendations:
            return
        rec_count = _normalize_recommendation_count(self.recommendation_count, default=1)
        recommendations = recommendations[:rec_count]
        top1 = recommendations[0]
        product_id = str(top1.get("product_id") or "unknown")
        product_title = str(top1.get("product_title") or product_id)

        self._log_path(f"recommend:{product_id}")
        self.log("action", f"[Decision path] {self._summary_path()}")
        self.push_to_viewer("decision_path", {"summary": self._summary_path(), "path": self.decision_path})
        reason_text = self._do_final_recommend(product_title, product_id)
        self.push_to_viewer("metric", {"name": "step", "value": "Recommendation complete"})

        cards = self._build_recommendation_cards(recommendations)
        product_url = f"/product/{quote(str(product_id), safe='')}"
        self.push_to_viewer(
            "recommend",
            {
                "product_id": product_id,
                "product_title": product_title,
                "product_url": product_url,
                "reason": reason_text,
                "recommendations": cards,
                "source": source,
            },
        )
        self.last_recommendation = {
            "product_id": product_id,
            "product_title": product_title,
            "product_url": product_url,
            "reason": reason_text,
        }
        self.last_recommendations = cards

    def _try_finalize_recommendation_after_termination(self, run_reason: str) -> bool:
        """
        Best-effort finalization when orchestrator exits without explicit recommend.
        Useful for long-loop terminations (max_steps / repeat_threshold).
        """
        if self.last_recommendation:
            return True
        recommend_tool = self.agent.tools.get("recommend")
        if not recommend_tool:
            return False

        viewed = []
        if hasattr(self.env, "get_viewed_for_recommend"):
            try:
                viewed = self.env.get_viewed_for_recommend() or []
            except Exception:
                viewed = []

        ordered_ids: List[str] = []
        seen = set()
        for item in reversed(viewed):
            if not isinstance(item, dict):
                continue
            pid = str(item.get("id") or "").strip()
            if pid and pid not in seen:
                seen.add(pid)
                ordered_ids.append(pid)
            if len(ordered_ids) >= 3:
                break

        rec_count = _normalize_recommendation_count(self.recommendation_count, default=1)
        params: Dict[str, Any]
        if rec_count <= 1:
            params = {"product_id": ordered_ids[0]} if ordered_ids else {}
        elif len(ordered_ids) >= rec_count:
            params = {"product_ids": ordered_ids[:rec_count]}
        else:
            # Let RecommendTool fill from env candidate pool if enough evidence exists.
            params = {}

        result = recommend_tool.execute(params)
        if result.success and result.data:
            self.log("action", f"Auto-finalization triggered after termination reason={run_reason}.")
            recommendations = self._normalize_recommendations_from_payload(result.data)
            self._publish_recommendations(recommendations, source="auto_finalize")
            return True

        self.log(
            "error",
            "Auto-final recommendation failed after run end: reason={reason}, detail={detail}".format(
                reason=run_reason,
                detail=result.error or "unknown",
            ),
        )
        return False

    def extract_search_keywords(self) -> str:
        self.log("thinking", f'Understanding user need: "{self.user_query}"')
        intent_keywords = str((self.query_intent or {}).get("keywords", "")).strip()
        if intent_keywords:
            keywords = intent_keywords
            self.search_keywords = keywords
            self.log("action", f'Extracted search keywords: "{keywords}"')
            self._log_path(f'query="{self.user_query[:40]}{"..." if len(self.user_query) > 40 else ""}"')
            self._log_path(f'keywords="{keywords}"')
            return keywords

        prompt = (
            "You are a shopping search assistant. The user has a shopping need described below.\n"
            "Your job is to extract concise English search keywords suitable for an e-commerce search box.\n\n"
            "Rules:\n"
            "- Output ONLY the search keywords, nothing else.\n"
            "- Use 2-5 words, like what a real user would type into Amazon search.\n"
            "- Translate to English if the input is in another language.\n"
            "- Focus on the product type and key attributes (e.g. material, style, use case).\n"
            "- Do NOT include budget/price constraints in keywords.\n"
            "- Budget constraints will be handled via filter_price tool, not via search keywords.\n\n"
            f"User need: {self.user_query}\n\n"
            "Search keywords:"
        )
        try:
            messages = [
                Message(role="system", content="You extract e-commerce search keywords. Reply with ONLY the keywords."),
                Message(role="user", content=prompt),
            ]
            response = self.agent.llm.generate(messages=messages, tools=None)
            raw = response.content if isinstance(response.content, str) else str(response.content)
            keywords = raw.strip().strip('"').strip("'").split("\n")[0].strip()
            if self.has_budget_constraint:
                stripped = self._strip_budget_from_keywords(keywords)
                if stripped:
                    keywords = stripped
                else:
                    fallback_stripped = self._strip_budget_from_keywords(self.user_query)
                    if fallback_stripped:
                        keywords = fallback_stripped
            if not keywords or len(keywords) > 100:
                fallback = self._strip_budget_from_keywords(self.user_query) if self.has_budget_constraint else self.user_query
                keywords = fallback or self.user_query
            self.search_keywords = keywords
            self.log("action", f'Extracted search keywords: "{keywords}"')
            self._log_path(f'query="{self.user_query[:40]}{"..." if len(self.user_query) > 40 else ""}"')
            self._log_path(f'keywords="{keywords}"')
            return keywords
        except Exception as e:
            self.log("error", f"Keyword extraction failed, falling back to raw query: {e}")
            self.search_keywords = self.user_query
            return self.user_query

    def _do_final_recommend(self, product_title: str, product_id: str) -> str:
        self.log("action", f"Selected recommended product: {product_title} (ID: {product_id})")
        self.log("action", f"========== Recommended Product ID: {product_id} ==========")
        self.log("thinking", "Generating recommendation rationale...")
        try:
            detail = self._api_product_detail(product_id) or {}
            price = detail.get("price")
            rating = detail.get("rating")
            rating_count = detail.get("rating_count")
            description = str(detail.get("description") or "").strip()
            if len(description) > 700:
                description = description[:700].rstrip() + "..."

            product_facts: List[str] = []
            product_facts.append(f"Product ID: {product_id}")
            product_facts.append(f"Title: {detail.get('title') or product_title}")
            try:
                price_num = float(price) if price is not None else None
            except Exception:
                price_num = None
            try:
                rating_num = float(rating) if rating is not None else None
            except Exception:
                rating_num = None
            try:
                rating_count_num = int(rating_count) if rating_count is not None else None
            except Exception:
                rating_count_num = None

            if price_num is not None:
                product_facts.append(f"Price: ${price_num:.2f}")
            if rating_num is not None:
                if rating_count_num is not None:
                    product_facts.append(f"Rating: {rating_num:.1f}/5 from {rating_count_num} ratings")
                else:
                    product_facts.append(f"Rating: {rating_num:.1f}/5")
            if description:
                product_facts.append(f"Description excerpt: {description}")

            prompt = (
                f'User need: "{self.user_query}"\n'
                f"Final recommended product: {product_title}\n\n"
                "Product facts:\n"
                + "\n".join(f"- {line}" for line in product_facts)
                + "\n\n"
                "Write a detailed recommendation reason in English.\n"
                "Requirements:\n"
                "1) Explain how the product matches the user constraints and intent.\n"
                "2) Cite concrete evidence from price/rating/description when available.\n"
                "3) Focus on strengths and fit; do NOT mention trade-offs, drawbacks, or weaknesses.\n"
                "4) Add who should buy it.\n"
                "Length: 120-220 words."
            )
            messages = [
                Message(
                    role="system",
                    content=(
                        "You are a rigorous shopping analyst. "
                        "Write in clear English, grounded only in provided facts. "
                        "Do not speculate about missing facts, and do not mention drawbacks or trade-offs."
                    ),
                ),
                Message(role="user", content=prompt),
            ]
            resp = self.agent.llm.generate(messages=messages, tools=None)
            txt = resp.content if isinstance(resp.content, str) else str(resp.content)
            reason_text = txt.strip()
            if not reason_text:
                reason_text = (
                    "This recommendation is the best available match for the stated need, "
                    "but the model could not generate a detailed rationale in this run."
                )

            self.last_recommendation_reason = reason_text
            for line in reason_text.split("\n"):
                if line.strip():
                    self.log("thinking", line.strip())
            return reason_text
        except Exception as e:
            self.log("error", f"Failed to generate recommendation rationale: {e}")
            fallback = (
                "The selected product appears to best match the requested criteria based on the explored candidates, "
                "but detailed rationale generation failed in this run."
            )
            self.last_recommendation_reason = fallback
            self.log("thinking", fallback)
            return fallback

    def _event_to_dict(self, event: OrchestratorEvent) -> Dict[str, Any]:
        def _json_safe(value: Any) -> Any:
            if isinstance(value, Action):
                return {
                    "tool_name": value.tool_name,
                    "parameters": _json_safe(value.parameters),
                    "reasoning": value.reasoning,
                }
            if isinstance(value, ToolResult):
                return {
                    "success": value.success,
                    "data": _json_safe(value.data),
                    "error": value.error,
                    "metadata": _json_safe(value.metadata),
                }
            if isinstance(value, Observation):
                data = value.data
                if isinstance(data, str) and data.startswith("data:image/"):
                    data = "<image_data_url>"
                return {
                    "modality": value.modality,
                    "timestamp": value.timestamp,
                    "data": _json_safe(data),
                    "metadata": _json_safe(value.metadata),
                }
            if isinstance(value, dict):
                return {str(k): _json_safe(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_json_safe(v) for v in value]
            if isinstance(value, tuple):
                return [_json_safe(v) for v in value]
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            return str(value)

        payload = _json_safe(dict(event.payload))
        return {
            "event_type": event.event_type,
            "timestamp": event.timestamp,
            "step": event.step,
            "payload": payload,
        }

    def _run_unified(self, keywords: str):
        self.last_recommendation = None
        self.last_recommendations = []
        self.last_recommendation_reason = ""

        initial_observation = self.env.reset(keywords)
        if self.has_budget_constraint:
            try:
                actions = self.env.get_available_actions() if hasattr(self.env, "get_available_actions") else {}
            except Exception:
                actions = {}
            num_products = int(actions.get("num_products", 0) or 0)
            if num_products <= 0:
                self._publish_no_budget_match(source="initial_budget_filter")
                return OrchestratorResult(
                    success=True,
                    reason="budget_no_match",
                    steps=0,
                    actions_executed=0,
                    errors=0,
                    repeated_actions=0,
                    action_counts={},
                )

        def _on_event(event: OrchestratorEvent):
            d = self._event_to_dict(event)
            self.trace_events.append(d)
            if event.event_type == "action":
                action = event.payload.get("action")
                if isinstance(action, Action) and action.reasoning:
                    for line in action.reasoning.strip().split("\n"):
                        if line.strip():
                            self.log("thinking", line.strip())
            elif event.event_type == "tool_result":
                result = event.payload.get("result")
                action = event.payload.get("action")
                if isinstance(result, ToolResult) and not result.success:
                    err = str(result.error or "")
                    search_tool_names = {"select_product", "next_page", "prev_page", "filter_price", "filter_rating"}
                    if (
                        isinstance(action, Action)
                        and action.tool_name in search_tool_names
                        and "Not on search page" in err
                    ):
                        self.log("action", f"Page context mismatch detected; auto-recovery attempted: {err}")
                    else:
                        self.log("error", f"Tool execution failed: {result.error}")
                if isinstance(action, Action) and action.tool_name == "recommend" and isinstance(result, ToolResult):
                    if result.success and result.data:
                        recommendations = self._normalize_recommendations_from_payload(result.data)
                        self._publish_recommendations(recommendations, source="tool_call")

        def _execute_action(action: Action) -> ToolResult:
            if action.tool_name == "noop":
                # LLM parse fallback; treat as non-fatal no-op to avoid noisy "unknown tool" errors.
                return ToolResult(success=True, data={"noop": True}, error=None)
            tool = self.agent.tools.get(action.tool_name)
            if not tool:
                return ToolResult(success=False, data=None, error=f"Unknown tool: {action.tool_name}")
            result = tool.execute(action.parameters)
            # Auto-heal: when model repeatedly selects an already-viewed index on search page,
            # try the next available indices to keep exploration moving.
            if (
                action.tool_name == "select_product"
                and not result.success
                and isinstance(result.error, str)
                and "already viewed" in result.error.lower()
            ):
                try:
                    actions = self.env.get_available_actions() if hasattr(self.env, "get_available_actions") else {}
                except Exception:
                    actions = {}
                num_products = int(actions.get("num_products", 0) or 0)
                try:
                    start_idx = int((action.parameters or {}).get("index", 1)) + 1
                except Exception:
                    start_idx = 2
                for idx in range(max(1, start_idx), max(1, num_products) + 1):
                    retry = tool.execute({"index": idx})
                    if retry.success:
                        retry.metadata = {**(retry.metadata or {}), "auto_recovered": f"skip_viewed_to_index_{idx}"}
                        return retry
                # If all visible items are already viewed, auto-advance to next page (when available)
                # and continue selecting from there. This avoids visual-mode select loops on exhausted pages.
                has_next = bool(actions.get("has_next")) if isinstance(actions, dict) else False
                if has_next:
                    next_tool = self.agent.tools.get("next_page")
                    if next_tool:
                        next_result = next_tool.execute({})
                        if next_result.success:
                            try:
                                actions2 = self.env.get_available_actions() if hasattr(self.env, "get_available_actions") else {}
                            except Exception:
                                actions2 = {}
                            num_products2 = int(actions2.get("num_products", 0) or 0)
                            for idx2 in range(1, max(1, num_products2) + 1):
                                retry2 = tool.execute({"index": idx2})
                                if retry2.success:
                                    retry2.metadata = {
                                        **(retry2.metadata or {}),
                                        "auto_recovered": f"all_viewed_then_next_page_to_index_{idx2}",
                                    }
                                    return retry2
                            return ToolResult(
                                success=True,
                                data={
                                    "success": True,
                                    "message": "Current page exhausted; auto-moved to next page",
                                    "auto_next_page": True,
                                },
                                error=None,
                                metadata={"tool": action.tool_name, "auto_recovered": "all_viewed_then_next_page"},
                            )
                        return ToolResult(
                            success=False,
                            data=None,
                            error=f"{result.error}; auto next_page failed: {next_result.error}",
                            metadata={"tool": action.tool_name, "auto_recovered": "all_viewed_next_page_failed"},
                        )
                # If no next page, avoid counting as hard error.
                return ToolResult(
                    success=True,
                    data={
                        "success": True,
                        "message": "All products on current page are already viewed",
                        "all_viewed_on_page": True,
                    },
                    error=None,
                    metadata={"tool": action.tool_name, "auto_recovered": "all_viewed_on_page"},
                )
            if (
                self.has_budget_constraint
                and action.tool_name == "filter_price"
                and result.success
            ):
                try:
                    actions = self.env.get_available_actions() if hasattr(self.env, "get_available_actions") else {}
                except Exception:
                    actions = {}
                num_products = int(actions.get("num_products", 0) or 0)
                if num_products <= 0:
                    return ToolResult(
                        success=True,
                        data={
                            "no_match_budget": True,
                            "message": "No products found after budget filtering",
                        },
                        error=None,
                        metadata={"tool": action.tool_name},
                    )
            # Auto-heal common context mismatch:
            # model calls a search-page tool while still on detail page.
            if (
                not result.success
                and isinstance(result.error, str)
                and "Not on search page" in result.error
                and action.tool_name in {"select_product", "next_page", "prev_page", "filter_price", "filter_rating"}
            ):
                back_tool = self.agent.tools.get("back")
                if back_tool:
                    back_result = back_tool.execute({})
                    if back_result.success:
                        retry = tool.execute(action.parameters)
                        if retry.success:
                            retry.metadata = {**(retry.metadata or {}), "auto_recovered": "back_then_retry"}
                            return retry
            return result

        def _stop_condition(action: Action, result: ToolResult, _step: int) -> Optional[str]:
            if action.tool_name == "recommend" and result.success:
                return "recommend_done"
            if (
                action.tool_name == "filter_price"
                and result.success
                and isinstance(result.data, dict)
                and bool(result.data.get("no_match_budget"))
            ):
                self._publish_no_budget_match(source="post_filter_budget")
                return "budget_no_match"
            return None

        orchestrator = AgentOrchestrator(
            agent=self.agent,
            get_initial_observation=lambda: initial_observation,
            get_next_observation=self.env.get_observation,
            execute_action=_execute_action,
            should_stop=_stop_condition,
            on_event=_on_event,
            max_steps=self.max_steps,
            max_repeated_actions=self.max_repeated_actions,
            max_errors=self.max_errors,
        )
        run_result = orchestrator.run()
        self.push_to_viewer("metric", {"name": "run_reason", "value": run_result.reason})
        self.push_to_viewer("metric", {"name": "run_steps", "value": str(run_result.steps)})
        self.push_to_viewer("metric", {"name": "run_errors", "value": str(run_result.errors)})
        if not self.last_recommendation and run_result.reason in {"max_steps", "repeat_threshold", "empty_actions", "error_threshold"}:
            self._try_finalize_recommendation_after_termination(run_result.reason)
        if self.trace_output:
            self.trace_output.parent.mkdir(parents=True, exist_ok=True)
            with open(self.trace_output, "w", encoding="utf-8") as f:
                for evt in self.trace_events:
                    f.write(json.dumps(evt, ensure_ascii=False) + "\n")

        self.env.close()

        if self.perception_mode == "visual" and self.stay_open:
            print("\nScreenshot is now shown in viewer. Press Ctrl+C to exit...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        return run_result

    def run(self):
        mode_label = "Visual (Screenshot)" if self.perception_mode == "visual" else "Verbal (Text)"
        print("\n" + "=" * 80)
        print(f"Browser Agent started  [{mode_label}] (Tool-First)")
        print("=" * 80)
        viewer_url = f"{self.web_server_url}/viewer"
        if self.viewer_token:
            viewer_url += f"?token={quote(self.viewer_token, safe='')}"
        print(f"\nViewer: {viewer_url}")
        print(f"User need: {self.user_query}")
        print(f"Perception mode: {mode_label}")
        print(f"Prompt profile: {self.prompt_profile['key']}")
        print("\nRunning...\n")

        keywords = self.extract_search_keywords()
        result = self._run_unified(keywords)
        return {
            "profile": self.prompt_profile["key"],
            "query": self.user_query,
            "keywords": keywords,
            "run_result": {
                "success": result.success,
                "reason": result.reason,
                "steps": result.steps,
                "actions_executed": result.actions_executed,
                "errors": result.errors,
                "repeated_actions": result.repeated_actions,
                "action_counts": result.action_counts,
            },
            "recommendation": self.last_recommendation,
            "recommendations": self.last_recommendations,
            "recommendation_reason": self.last_recommendation_reason,
        }


def _load_eval_queries(suite_name: str, limit: int = 0, seed: int = 42) -> List[str]:
    cfg_path = Path(__file__).parent / "configs" / "queries.yaml"
    if not cfg_path.exists():
        return []
    try:
        import yaml
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    queries = data.get(suite_name)
    if isinstance(queries, str):
        queries = [queries]
    if not isinstance(queries, list):
        return []
    queries = [str(q).strip() for q in queries if str(q).strip()]
    rnd = random.Random(seed)
    rnd.shuffle(queries)
    if limit and limit > 0:
        queries = queries[:limit]
    return queries


def _run_eval_suite(args, api_key: str) -> int:
    profiles = [p.strip() for p in args.prompt_profile.split(",") if p.strip()]
    queries = _load_eval_queries(args.eval_suite, limit=args.eval_limit, seed=args.eval_seed)
    if not queries:
        print(f"Error: no available queries in eval suite '{args.eval_suite}'.")
        return 2

    output_dir = Path(args.eval_output_dir or "experiment_results/prompt_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: Dict[str, Any] = {
        "suite": args.eval_suite,
        "seed": args.eval_seed,
        "queries": queries,
        "profiles": profiles,
        "results": {},
    }

    for profile in profiles:
        all_results["results"][profile] = []
        for idx, query in enumerate(queries, 1):
            trace_file = output_dir / f"trace_{profile}_{idx:02d}.jsonl"
            print(f"\n[Eval] profile={profile} ({idx}/{len(queries)}) query={query}")
            try:
                agent = LiveBrowserAgent(
                    llm_api_key=api_key,
                    llm_backend=args.llm,
                    perception_mode=args.perception,
                    web_server_url=args.server,
                    user_query=query,
                    condition_name=args.condition_name,
                    stay_open=False,
                    page=args.page,
                    price_min=args.price_min,
                    price_max=args.price_max,
                    rating_min=args.rating_min,
                    temperature=args.temperature,
                    env_backend=None if args.env_backend == "auto" else args.env_backend,
                    legacy_fallback=args.legacy_fallback,
                    trace_output=str(trace_file),
                    mcp_endpoint=args.mcp_endpoint,
                    prompt_profile=profile,
                    max_steps=args.max_steps,
                    max_repeated_actions=args.max_repeated_actions,
                    max_errors=args.max_errors,
                    max_history=args.max_history,
                    verbal_use_vlm=args.verbal_use_vlm,
                    confidence_threshold=args.confidence_threshold,
                    min_viewed_before_recommend=args.min_viewed_before_recommend,
                )
                run_info = agent.run()
                all_results["results"][profile].append(run_info)
            except Exception as e:
                all_results["results"][profile].append(
                    {"profile": profile, "query": query, "error": str(e), "run_result": {"success": False}}
                )

    summary = {"suite": args.eval_suite, "profiles": {}}
    for profile in profiles:
        runs = all_results["results"][profile]
        total = len(runs)
        success_runs = [r for r in runs if r.get("run_result", {}).get("success")]
        avg_steps = (
            sum(float(r.get("run_result", {}).get("steps", 0)) for r in runs) / total
            if total else 0.0
        )
        avg_errors = (
            sum(float(r.get("run_result", {}).get("errors", 0)) for r in runs) / total
            if total else 0.0
        )
        summary["profiles"][profile] = {
            "num_runs": total,
            "success_rate": (len(success_runs) / total) if total else 0.0,
            "avg_steps": round(avg_steps, 3),
            "avg_errors": round(avg_errors, 3),
        }

    (output_dir / "prompt_eval_results.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "prompt_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md_lines = [f"# Prompt Eval Summary ({args.eval_suite})", ""]
    for profile, s in summary["profiles"].items():
        md_lines.append(
            f"- `{profile}`: success_rate={s['success_rate']:.2%}, avg_steps={s['avg_steps']}, avg_errors={s['avg_errors']} (n={s['num_runs']})"
        )
    (output_dir / "prompt_eval_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\nEval completed. Output: {output_dir}")
    return 0


def main():
    import argparse

    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        for line in _env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip("'\"")
                if k and v and k not in os.environ:
                    os.environ[k] = v

    parser = argparse.ArgumentParser(description="Browser Agent live demo (Tool-First + Orchestrator)")
    parser.add_argument("--api-key", default=None, help="API Key")
    parser.add_argument("--llm", choices=["openai", "qwen", "kimi"], default="qwen", help="LLM backend")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM sampling temperature")
    parser.add_argument("--max-steps", type=int, default=40, help="Maximum decision steps")
    parser.add_argument("--max-repeated-actions", type=int, default=6, help="Repeat-action fuse threshold")
    parser.add_argument("--max-errors", type=int, default=8, help="Error fuse threshold")
    parser.add_argument("--max-history", type=int, default=12, help="Max retained message history for agent")
    parser.add_argument(
        "--recommendation-count",
        type=int,
        default=None,
        help="Final recommendation count (1~3). Defaults to ACES_RECOMMENDATION_COUNT, or 1 if unset.",
    )
    parser.add_argument("--confidence-threshold", type=int, default=None, help="Override prompt-profile confidence threshold")
    parser.add_argument(
        "--min-viewed-before-recommend",
        type=int,
        default=None,
        help="Override prompt-profile minimum viewed products before recommend",
    )
    parser.add_argument(
        "--perception",
        choices=["visual", "verbal"],
        default="visual",
        help="Perception mode: visual=screenshot, verbal=text",
    )
    parser.add_argument(
        "--verbal-use-vlm",
        action="store_true",
        help="Use a vision model even when perception=verbal (changes model only, not verbal observation format)",
    )
    parser.add_argument(
        "--query",
        default="smart watch under 100 dollars with heart rate and sleep tracking",
        help="User shopping need",
    )
    parser.add_argument("--server", default="http://localhost:5000", help="Web server URL")
    parser.add_argument("--condition-name", default=None, help="Experiment condition name")
    parser.add_argument("--page", type=int, default=1, help="Initial page number")
    parser.add_argument("--price-min", type=float, default=None, help="Price lower bound")
    parser.add_argument("--price-max", type=float, default=None, help="Price upper bound")
    parser.add_argument("--rating-min", type=float, default=None, help="Minimum star rating")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--legacy-fallback", action="store_true", help="Enable legacy text-guess fallback")
    parser.add_argument("--trace-output", default=None, help="JSONL trace output path")
    parser.add_argument(
        "--prompt-profile",
        default="robust_compare",
        help=f"Prompt profile key(s), comma-separated. choices={','.join(PROMPT_PROFILES.keys())}",
    )
    parser.add_argument("--eval-suite", default=None, help="Run batch eval by query suite key from configs/queries.yaml")
    parser.add_argument("--eval-limit", type=int, default=0, help="Limit number of eval queries (0 means all)")
    parser.add_argument("--eval-seed", type=int, default=42, help="Random seed for eval query shuffle")
    parser.add_argument("--eval-output-dir", default=None, help="Output directory for eval reports")
    parser.add_argument(
        "--env-backend",
        choices=["auto", "api", "playwright", "mcp"],
        default="auto",
        help="Environment backend. Default auto: visual->playwright, verbal->api",
    )
    parser.add_argument("--mcp-endpoint", default=None, help="MCP HTTP endpoint, required for --env-backend mcp")

    args = parser.parse_args()

    api_key = args.api_key
    if not api_key:
        if args.llm == "qwen":
            api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        elif args.llm == "kimi":
            api_key = os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: API key required. Pass --api-key or set an environment variable.")
        sys.exit(1)
    else:
        key_tail = api_key[-4:] if len(api_key) >= 4 else "***"
        print(f"[INFO] Using {args.llm} API key: ****{key_tail}")

    if args.eval_suite:
        code = _run_eval_suite(args, api_key)
        sys.exit(code)

    env_backend = None if args.env_backend == "auto" else args.env_backend
    try:
        agent = LiveBrowserAgent(
            llm_api_key=api_key,
            llm_backend=args.llm,
            perception_mode=args.perception,
            web_server_url=args.server,
            user_query=args.query,
            condition_name=args.condition_name,
            stay_open=not args.once,
            page=args.page,
            price_min=args.price_min,
            price_max=args.price_max,
            rating_min=args.rating_min,
            temperature=args.temperature,
            env_backend=env_backend,
            legacy_fallback=args.legacy_fallback,
            trace_output=args.trace_output,
            mcp_endpoint=args.mcp_endpoint,
            prompt_profile=args.prompt_profile.split(",")[0].strip(),
            max_steps=args.max_steps,
            max_repeated_actions=args.max_repeated_actions,
            max_errors=args.max_errors,
            max_history=args.max_history,
            verbal_use_vlm=args.verbal_use_vlm,
            recommendation_count=args.recommendation_count,
            confidence_threshold=args.confidence_threshold,
            min_viewed_before_recommend=args.min_viewed_before_recommend,
        )
    except RuntimeError as e:
        print(f"Error: {e}")
        if env_backend == "mcp":
            print("Hint: no MCP caller injected. You can switch to --env-backend api or playwright.")
        sys.exit(2)

    agent.run()


if __name__ == "__main__":
    main()
