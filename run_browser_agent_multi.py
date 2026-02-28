#!/usr/bin/env python3
"""
多详情页浏览版 Browser Agent

与 run_browser_agent.py 的区别：
- Agent 可以浏览多个商品详情页，而非只看一个
- 每看完一个详情页后，LLM 自主决定是否继续查看其他商品
- 最终基于所有已浏览商品做出推荐
- 支持 --max-views 设置安全上限（默认 5）

支持 visual / verbal 两种感知模式，与原版一致。
"""

import os
import sys
import re
import json
import time
import base64
import requests
from pathlib import Path
from typing import List, Tuple, Optional, Dict

sys.path.insert(0, str(Path(__file__).parent))

from aces.agents import ComposableAgent
from aces.llm_backends import OpenAIBackend, QwenBackend
from aces.perception import VisualPerception
from aces.core.protocols import Message


class MultiBrowseAgent:
    """
    多详情页浏览 Agent

    核心循环：搜索 → 选品 → 看详情 → 决定继续/停止 → … → 最终推荐
    """

    def __init__(
        self,
        llm_api_key: str,
        llm_backend: str = "qwen",
        perception_mode: str = "visual",
        web_server_url: str = "http://localhost:5000",
        user_query: str = "mousepad",
        condition_name: str = None,
        max_views: int = 5,
        stay_open: bool = True,
    ):
        self.web_server_url = web_server_url
        self.user_query = user_query
        self.condition_name = condition_name
        self.perception_mode = perception_mode
        self.max_views = max_views
        self.stay_open = stay_open
        self.search_keywords: Optional[str] = None

        # Browsing state
        self.viewed_products: List[Dict] = []  # [{index, id, title, summary}, ...]
        self.viewed_indices: set = set()  # 1-based indices already visited

        if llm_backend == "qwen":
            if perception_mode == "visual":
                llm = QwenBackend(model="qwen-vl-plus", api_key=llm_api_key)
            else:
                llm = QwenBackend(model="qwen-plus", api_key=llm_api_key)
        elif llm_backend == "openai":
            llm = OpenAIBackend(model="gpt-4o", api_key=llm_api_key)
        else:
            llm = OpenAIBackend(model="gpt-4o", api_key=llm_api_key)

        self.agent = ComposableAgent(
            llm=llm,
            perception=VisualPerception(),
            tools=[],
        )

        self.playwright = None
        self.browser = None
        self.page = None

    # ==================================================================
    # Shared utilities
    # ==================================================================

    def push_to_viewer(self, data_type: str, data: dict):
        try:
            requests.post(
                f"{self.web_server_url}/api/push",
                json={"type": data_type, **data},
                timeout=1,
            )
        except:
            pass

    def log(self, level: str, message: str):
        print(f"[{level.upper()}] {message}")
        self.push_to_viewer("log", {"level": level, "message": message})

    def push_screenshot(self, screenshot_bytes: bytes, url: str):
        b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        self.push_to_viewer("screenshot", {
            "screenshot": f"data:image/png;base64,{b64}",
            "url": url,
        })

    def _build_search_url(self, keywords: str) -> str:
        from urllib.parse import quote
        url = f"{self.web_server_url}/search?q={quote(keywords)}"
        if self.condition_name:
            url += f"&condition_name={quote(self.condition_name)}"
        return url

    def extract_search_keywords(self) -> str:
        self.log("thinking", f"理解用户需求: \"{self.user_query}\"")
        prompt = (
            "You are a shopping search assistant. The user has a shopping need described below.\n"
            "Your job is to extract concise English search keywords suitable for an e-commerce search box.\n\n"
            "Rules:\n"
            "- Output ONLY the search keywords, nothing else.\n"
            "- Use 2-5 words, like what a real user would type into Amazon search.\n"
            "- Translate to English if the input is in another language.\n"
            "- Focus on the product type and key attributes (e.g. material, style, use case).\n\n"
            f"User need: {self.user_query}\n\n"
            "Search keywords:"
        )
        try:
            messages = [
                Message(role="system", content="You extract e-commerce search keywords. Reply with ONLY the keywords."),
                Message(role="user", content=prompt),
            ]
            resp = self.agent.llm.generate(messages=messages, tools=None)
            raw = resp.content if isinstance(resp.content, str) else str(resp.content)
            keywords = raw.strip().strip('"').strip("'").split("\n")[0].strip()
            if not keywords or len(keywords) > 100:
                keywords = self.user_query
            self.search_keywords = keywords
            self.log("action", f"✅ 提取搜索关键词: \"{keywords}\"")
            return keywords
        except Exception as e:
            self.log("error", f"关键词提取失败，回退使用原始 query: {e}")
            self.search_keywords = self.user_query
            return self.user_query

    def _llm_call(self, system: str, user: str) -> str:
        messages = [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]
        resp = self.agent.llm.generate(messages=messages, tools=None)
        return resp.content if isinstance(resp.content, str) else str(resp.content)

    def _llm_call_with_image(self, system: str, image_data_url: str, user_text: str) -> str:
        messages = [
            Message(role="system", content=system),
            Message(role="user", content=image_data_url),
            Message(role="user", content=user_text),
        ]
        resp = self.agent.llm.generate(messages=messages, tools=None)
        return resp.content if isinstance(resp.content, str) else str(resp.content)

    def _viewed_summary(self) -> str:
        if not self.viewed_products:
            return "(none yet)"
        lines = []
        for v in self.viewed_products:
            lines.append(f"  - [{v['index']}] {v['title']} → {v['summary'][:120]}")
        return "\n".join(lines)

    # ==================================================================
    # Verbal helpers
    # ==================================================================

    def _api_search(self, keywords: str, limit: int = 8) -> List[dict]:
        from urllib.parse import quote
        url = f"{self.web_server_url}/api/search?q={quote(keywords)}&limit={limit}"
        if self.condition_name:
            url += f"&condition_name={quote(self.condition_name)}"
        try:
            resp = requests.get(url, timeout=15)
            return resp.json().get("products", [])
        except Exception as e:
            self.log("error", f"API 搜索请求失败: {e}")
            return []

    def _api_product_detail(self, product_id: str) -> Optional[dict]:
        from urllib.parse import quote
        url = f"{self.web_server_url}/api/product/{quote(product_id)}"
        if self.condition_name:
            url += f"?condition_name={quote(self.condition_name)}"
        try:
            resp = requests.get(url, timeout=10)
            return resp.json().get("product")
        except Exception as e:
            self.log("error", f"API 商品详情请求失败: {e}")
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
            lines.append(
                f"[{i}] {p['title']}\n"
                f"    Price: ${p['price']:.2f} | "
                f"Rating: {p.get('rating', 0):.1f}/5 ({p.get('rating_count', 0)} reviews)"
                f"{badge_str}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_product_detail(p: dict) -> str:
        parts = [
            f"Title: {p['title']}",
            f"Price: ${p['price']:.2f}",
            f"Rating: {p.get('rating', 0):.1f}/5 ({p.get('rating_count', 0)} reviews)",
        ]
        if p.get("description"):
            parts.append(f"\nDescription:\n{p['description']}")
        return "\n".join(parts)

    # ==================================================================
    # Visual helpers
    # ==================================================================

    def init_browser(self):
        from playwright.sync_api import sync_playwright
        self.log("action", "初始化浏览器...")
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=True, args=["--no-sandbox"]
        )
        self.page = self.browser.new_page(viewport={"width": 1280, "height": 800})
        self.log("action", "✓ 浏览器已启动")

    def navigate_and_capture(self, url: str) -> bytes:
        self.log("action", f"导航到: {url}")
        self.page.goto(url, wait_until="networkidle")
        time.sleep(1)
        screenshot_bytes = self.page.screenshot(type="png")
        self.log("action", f"✓ 截图完成 ({len(screenshot_bytes)/1024:.1f} KB)")
        self.push_screenshot(screenshot_bytes, url)
        return screenshot_bytes

    def get_product_detail_links(self) -> List[Tuple[str, str]]:
        try:
            hrefs = self.page.evaluate("""
                () => Array.from(document.querySelectorAll('a[href^="/product/"]'))
                    .map(a => a.getAttribute('href'))
                    .filter(Boolean)
            """)
        except Exception as e:
            self.log("error", f"解析商品链接失败: {e}")
            return []
        base = self.web_server_url.rstrip("/")
        seen = set()
        result = []
        for href in (hrefs or []):
            path = href.split("?")[0]
            pid = path.rstrip("/").split("/")[-1]
            if pid and pid not in seen:
                seen.add(pid)
                result.append((pid, base + href))
        return result

    def get_description_from_detail_page(self) -> str:
        try:
            loc = self.page.locator(".detail-description .text").first
            if loc.count() == 0:
                return "(页面上未找到描述区域)"
            text = loc.inner_text(timeout=2000).strip()
            return text or "No description available."
        except Exception as e:
            self.log("error", f"提取 description 失败: {e}")
            return "(提取失败)"

    # ==================================================================
    # Decision: pick next product or stop
    # ==================================================================

    def _ask_continue_or_stop(self, num_products: int) -> str:
        """
        让 LLM 决定：继续看下一个商品（回复数字）还是停止（回复 STOP）。
        Returns: 'STOP' or a number string like '3'
        """
        available = [i for i in range(1, num_products + 1) if i not in self.viewed_indices]
        if not available:
            return "STOP"

        viewed_str = self._viewed_summary()
        available_str = ", ".join(str(i) for i in available)

        prompt = (
            f"You are a shopping assistant helping a user find: \"{self.user_query}\"\n\n"
            f"You have already viewed the following products:\n{viewed_str}\n\n"
            f"Products you have NOT yet viewed: [{available_str}]\n\n"
            f"You can view at most {self.max_views} products total "
            f"(viewed so far: {len(self.viewed_products)}).\n\n"
            "Decision:\n"
            "- If you have found a clearly good match, or you have enough information to make a recommendation, reply: STOP\n"
            "- If you want to examine another product for comparison, reply with its number.\n\n"
            "Reply with ONLY 'STOP' or a single number."
        )
        try:
            raw = self._llm_call(
                "You are a shopping decision agent. Reply with ONLY 'STOP' or a number.",
                prompt,
            )
            decision = raw.strip().split("\n")[0].strip().upper()
            self.log("thinking", f"Agent 决策: {decision}")

            if "STOP" in decision:
                return "STOP"
            match = re.search(r"\b([1-9]\d*)\b", decision)
            if match:
                idx = int(match.group(1))
                if idx in available:
                    return str(idx)
            return "STOP"
        except Exception as e:
            self.log("error", f"决策调用失败: {e}")
            return "STOP"

    # ==================================================================
    # Final recommendation (shared)
    # ==================================================================

    def _final_recommendation(self, product_list_text: str) -> str:
        viewed_str = self._viewed_summary()
        prompt = (
            f"You are a shopping assistant. The user wants: \"{self.user_query}\"\n\n"
            f"Here are all the search results:\n{product_list_text}\n\n"
            f"You examined the following products in detail:\n{viewed_str}\n\n"
            "Based on everything you've seen, which ONE product do you recommend the user buy?\n"
            "Give your final recommendation with:\n"
            "1. The product number and name\n"
            "2. Key reasons for your choice\n"
            "3. Any caveats or alternatives worth mentioning"
        )
        return self._llm_call(
            "You are a shopping assistant giving a final purchase recommendation.",
            prompt,
        )

    # ==================================================================
    # Main entry
    # ==================================================================

    def run(self):
        mode_label = "Visual (截图)" if self.perception_mode == "visual" else "Verbal (文本)"
        print("\n" + "=" * 80)
        print(f"🤖 Multi-Browse Agent  [{mode_label}]")
        print("=" * 80)
        print(f"\n📺 Viewer: {self.web_server_url}/viewer")
        print(f"🛒 用户需求: {self.user_query}")
        print(f"👁 感知模式: {mode_label}")
        print(f"🔄 最多浏览详情页: {self.max_views}")
        print("\n开始执行...\n")

        keywords = self.extract_search_keywords()

        if self.perception_mode == "verbal":
            self._run_verbal(keywords)
        else:
            self._run_visual(keywords)

    # ------------------------------------------------------------------
    # Verbal multi-browse loop
    # ------------------------------------------------------------------

    def _run_verbal(self, keywords: str):
        self.log("action", f"[Verbal] 通过 API 检索商品: \"{keywords}\"")
        products = self._api_search(keywords)
        if not products:
            self.log("error", "未检索到任何商品")
            return
        num = len(products)
        self.log("action", f"[Verbal] 获取到 {num} 个商品")
        product_text = self._format_product_list(products)
        self.log("thinking", f"候选商品列表:\n{product_text}")

        # --- Initial pick ---
        self.log("thinking", "[Verbal] LLM 正在选择第一个要查看的商品...")
        raw = self._llm_call(
            "You are a shopping assistant. Reply with ONLY a number.",
            f"User wants: \"{self.user_query}\"\n\n{product_text}\n\n"
            f"Which product (1-{num}) do you want to examine first? Reply with ONLY a number.",
        )
        chosen = self._parse_choice(raw, num)
        self.log("action", f"✅ 选择第 {chosen} 个商品")

        # --- Browse loop ---
        while len(self.viewed_products) < self.max_views:
            self.viewed_indices.add(chosen)
            selected = products[chosen - 1]
            pid = selected["id"]
            self.log("action", f"[Verbal] 查看第 {chosen} 个商品详情: {pid} ({len(self.viewed_products)+1}/{self.max_views})")

            detail = self._api_product_detail(pid) or selected
            detail_text = self._format_product_detail(detail)
            self.log("thinking", f"商品详情:\n{detail_text[:500]}{'...' if len(detail_text) > 500 else ''}")

            summary_raw = self._llm_call(
                "You are a shopping assistant. Briefly summarize this product in 1-2 sentences.",
                f"User wants: \"{self.user_query}\"\n\n{detail_text}\n\nBrief summary:",
            )
            summary = summary_raw.strip().split("\n")[0][:200]
            self.log("thinking", f"摘要: {summary}")

            self.viewed_products.append({
                "index": chosen,
                "id": pid,
                "title": selected["title"],
                "summary": summary,
            })

            # --- Continue or stop? ---
            decision = self._ask_continue_or_stop(num)
            if decision == "STOP":
                self.log("action", f"🛑 Agent 决定停止浏览 (已看 {len(self.viewed_products)} 个商品)")
                break
            chosen = int(decision)
            self.log("action", f"🔄 Agent 决定继续查看第 {chosen} 个商品")

        # --- Final recommendation ---
        self.log("thinking", "LLM 正在做最终推荐...")
        final = self._final_recommendation(product_text)
        for line in final.strip().split("\n"):
            if line.strip():
                self.log("thinking", line.strip())
                time.sleep(0.2)
        self.log("action", f"✅ [Verbal] 最终推荐完成 (浏览了 {len(self.viewed_products)} 个详情页)")
        self.push_to_viewer("metric", {
            "name": "multi_browse_done",
            "value": json.dumps({
                "mode": "verbal",
                "viewed_count": len(self.viewed_products),
                "viewed_ids": [v["id"] for v in self.viewed_products],
            }),
        })

    # ------------------------------------------------------------------
    # Visual multi-browse loop
    # ------------------------------------------------------------------

    def _run_visual(self, keywords: str):
        search_url = self._build_search_url(keywords)
        try:
            self.init_browser()
            self.log("thinking", "准备访问商品搜索结果页...")
            search_screenshot = self.navigate_and_capture(search_url)

            product_links = self.get_product_detail_links()
            num = len(product_links)
            self.log("action", f"页面上共 {num} 个商品可点进详情")
            if num == 0:
                return

            # VLM: initial pick from search screenshot
            self.log("thinking", "VLM 正在分析搜索结果截图...")
            obs = self.agent.perception.encode(search_screenshot)

            prompt = (
                "请仔细查看这个商品搜索结果页的截图。\n"
                f"页面有 {num} 个商品。你想先看哪个商品的详情？\n"
                "请只回复一个数字（1 表示第一个，以此类推）。"
            )
            raw = self._llm_call_with_image(
                "你是一个购物助手。只回复一个数字。",
                obs.data, prompt,
            )
            chosen = self._parse_choice(raw, num)
            self.log("action", f"✅ 选择查看第 {chosen} 个商品")

            # --- Browse loop ---
            while len(self.viewed_products) < self.max_views:
                self.viewed_indices.add(chosen)
                pid, detail_url = product_links[chosen - 1]
                self.log("action", f"正在打开第 {chosen} 个商品详情页: {pid} ({len(self.viewed_products)+1}/{self.max_views})")

                detail_screenshot = self.navigate_and_capture(detail_url)
                description = self.get_description_from_detail_page()
                self.log("action", f"已提取商品描述（{len(description)} 字）")

                obs_detail = self.agent.perception.encode(detail_screenshot)
                summary_raw = self._llm_call_with_image(
                    "你是一个购物助手。请用 1-2 句话简要总结这个商品的特点。",
                    obs_detail.data,
                    f"商品描述:\n{description[:800]}\n\n请用 1-2 句话总结。",
                )
                summary = summary_raw.strip().split("\n")[0][:200]
                self.log("thinking", f"摘要: {summary}")

                self.viewed_products.append({
                    "index": chosen,
                    "id": pid,
                    "title": pid,
                    "summary": summary,
                })

                # --- Continue or stop? ---
                decision = self._ask_continue_or_stop(num)
                if decision == "STOP":
                    self.log("action", f"🛑 Agent 决定停止浏览 (已看 {len(self.viewed_products)} 个商品)")
                    break
                chosen = int(decision)
                self.log("action", f"🔄 Agent 决定继续查看第 {chosen} 个商品")

            # --- Back to search page for context, then final recommendation ---
            self.log("thinking", "返回搜索结果页做最终推荐...")
            self.navigate_and_capture(search_url)
            obs_final = self.agent.perception.encode(
                self.page.screenshot(type="png")
            )

            viewed_str = self._viewed_summary()
            prompt_final = (
                f"用户需求: \"{self.user_query}\"\n\n"
                f"你已经查看了以下商品的详情:\n{viewed_str}\n\n"
                "请根据搜索结果页截图和你查看过的详情，推荐一个最佳商品，并说明理由。"
            )
            final = self._llm_call_with_image(
                "你是一个购物助手，给出最终购买推荐。",
                obs_final.data, prompt_final,
            )
            for line in final.strip().split("\n"):
                if line.strip():
                    self.log("thinking", line.strip())
                    time.sleep(0.2)

            self.log("action", f"✅ [Visual] 最终推荐完成 (浏览了 {len(self.viewed_products)} 个详情页)")
            self.push_to_viewer("metric", {
                "name": "multi_browse_done",
                "value": json.dumps({
                    "mode": "visual",
                    "viewed_count": len(self.viewed_products),
                    "viewed_ids": [v["id"] for v in self.viewed_products],
                }),
            })

            if self.stay_open:
                print("\n按 Ctrl+C 退出...")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass

        finally:
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
            self.log("action", "浏览器已关闭")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_choice(self, raw: str, max_n: int) -> int:
        for line in raw.strip().split("\n"):
            if line.strip():
                self.log("thinking", line.strip())
        match = re.search(r"\b([1-9]\d*)\b", raw)
        if match:
            return max(1, min(int(match.group(1)), max_n))
        return 1


# ======================================================================
# CLI
# ======================================================================

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

    parser = argparse.ArgumentParser(
        description="Multi-Browse Agent: 浏览多个详情页后做推荐"
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--llm", choices=["openai", "qwen"], default="qwen")
    parser.add_argument(
        "--perception", choices=["visual", "verbal"], default="visual",
        help="感知模式: visual=截图给VLM, verbal=结构化文本给LLM",
    )
    parser.add_argument("--query", default="mousepad", help="用户购物需求（自然语言）")
    parser.add_argument("--server", default="http://localhost:5000")
    parser.add_argument("--condition-name", default=None)
    parser.add_argument(
        "--max-views", type=int, default=5,
        help="最多浏览几个详情页（安全上限，LLM 可提前 STOP）",
    )
    parser.add_argument("--once", action="store_true", help="完成后立即退出")

    args = parser.parse_args()

    api_key = args.api_key
    if not api_key:
        if args.llm == "qwen":
            api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("错误: 需要 API Key。请传 --api-key 或设置环境变量")
        sys.exit(1)

    agent = MultiBrowseAgent(
        llm_api_key=api_key,
        llm_backend=args.llm,
        perception_mode=args.perception,
        web_server_url=args.server,
        user_query=args.query,
        condition_name=args.condition_name,
        max_views=args.max_views,
        stay_open=not args.once,
    )

    agent.run()


if __name__ == "__main__":
    main()
