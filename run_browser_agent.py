#!/usr/bin/env python3
"""
在服务器上运行 Browser Agent，实时推送到 MacBook

Agent 使用 Playwright 操作网页，实时截图和日志推送到 MacBook 浏览器。
"""

import os
import sys
import re
import time
import base64
import requests
from pathlib import Path
from typing import List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent))

from aces.agents import ComposableAgent
from aces.llm_backends import OpenAIBackend, QwenBackend
from aces.perception import VisualPerception
from aces.core.protocols import Message


class LiveBrowserAgent:
    """
    实时浏览器 Agent
    
    在服务器上运行，通过 WebSocket 推送到 MacBook 查看。
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
    ):
        self.web_server_url = web_server_url
        self.user_query = user_query
        self.condition_name = condition_name
        self.perception_mode = perception_mode
        self.stay_open = stay_open
        self.search_keywords: Optional[str] = None
        
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
        
        # Playwright browser (only used in visual mode)
        self.playwright = None
        self.browser = None
        self.page = None
    
    def push_to_viewer(self, data_type: str, data: dict):
        """推送数据到 MacBook viewer"""
        try:
            requests.post(
                f"{self.web_server_url}/api/push",
                json={"type": data_type, **data},
                timeout=1
            )
        except:
            pass  # 不影响主流程
    
    def log(self, level: str, message: str):
        """记录日志并推送"""
        print(f"[{level.upper()}] {message}")
        self.push_to_viewer("log", {"level": level, "message": message})
    
    def push_screenshot(self, screenshot_bytes: bytes, url: str):
        """推送截图"""
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
        screenshot_data = f"data:image/png;base64,{screenshot_base64}"
        
        self.push_to_viewer("screenshot", {
            "screenshot": screenshot_data,
            "url": url
        })
    
    def _build_search_url(self, keywords: str) -> str:
        """根据关键词构造搜索 URL"""
        from urllib.parse import quote
        url = f"{self.web_server_url}/search?q={quote(keywords)}"
        if self.condition_name:
            url += f"&condition_name={quote(self.condition_name)}"
        return url

    def extract_search_keywords(self) -> str:
        """
        Step 0: Query Understanding
        LLM 理解用户的自然语言需求，提取适合电商搜索框的英文关键词。
        """
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
            response = self.agent.llm.generate(messages=messages, tools=None)
            raw = response.content if isinstance(response.content, str) else str(response.content)
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

    # ==================================================================
    # Verbal mode helpers
    # ==================================================================

    def _api_search(self, keywords: str, limit: int = 8) -> List[dict]:
        """调用 /api/search 获取结构化商品列表（与 visual 模式用同一检索管线）"""
        from urllib.parse import quote
        url = f"{self.web_server_url}/api/search?q={quote(keywords)}&limit={limit}"
        if self.condition_name:
            url += f"&condition_name={quote(self.condition_name)}"
        try:
            resp = requests.get(url, timeout=15)
            data = resp.json()
            return data.get("products", [])
        except Exception as e:
            self.log("error", f"API 搜索请求失败: {e}")
            return []

    def _api_product_detail(self, product_id: str) -> Optional[dict]:
        """调用 /api/product/{id} 获取单个商品完整详情"""
        from urllib.parse import quote
        url = f"{self.web_server_url}/api/product/{quote(product_id)}"
        if self.condition_name:
            url += f"?condition_name={quote(self.condition_name)}"
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
            return data.get("product")
        except Exception as e:
            self.log("error", f"API 商品详情请求失败: {e}")
            return None

    @staticmethod
    def _format_product_list(products: List[dict]) -> str:
        """将商品列表格式化为 LLM 可读的文本"""
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
        """将单个商品详情格式化为 LLM 可读的文本"""
        parts = [
            f"Title: {p['title']}",
            f"Price: ${p['price']:.2f}",
            f"Rating: {p.get('rating', 0):.1f}/5 ({p.get('rating_count', 0)} reviews)",
        ]
        if p.get("description"):
            parts.append(f"\nDescription:\n{p['description']}")
        return "\n".join(parts)

    def run_verbal(self, keywords: str):
        """Verbal 模式：用结构化文本（非截图）让 LLM 选品并推荐"""
        # Step 1: 获取 top-8 商品（同一检索管线）
        self.log("action", f"[Verbal] 通过 API 检索商品: \"{keywords}\"")
        products = self._api_search(keywords)
        if not products:
            self.log("error", "未检索到任何商品")
            return
        self.log("action", f"[Verbal] 获取到 {len(products)} 个商品")

        product_text = self._format_product_list(products)
        self.log("thinking", f"候选商品列表:\n{product_text}")

        # Step 2: LLM 从文本选品
        self.log("thinking", "[Verbal] LLM 正在分析商品列表...")
        prompt_select = (
            f"You are a shopping assistant helping a user find: \"{self.user_query}\"\n\n"
            f"Here are the top {len(products)} search results:\n\n"
            f"{product_text}\n\n"
            "Based on the user's need, which ONE product would you like to examine in detail?\n"
            "Reply with ONLY the number (1-{n}).".format(n=len(products))
        )
        try:
            messages = [
                Message(role="system", content="You are a shopping assistant. Reply with ONLY a number."),
                Message(role="user", content=prompt_select),
            ]
            resp = self.agent.llm.generate(messages=messages, tools=None)
            raw = resp.content if isinstance(resp.content, str) else str(resp.content)
            for line in raw.strip().split("\n"):
                if line.strip():
                    self.log("thinking", line.strip())

            chosen = 1
            match = re.search(r"\b([1-9]\d*)\b", raw)
            if match:
                chosen = max(1, min(int(match.group(1)), len(products)))
            self.log("action", f"✅ [Verbal] 选择第 {chosen} 个商品")
        except Exception as e:
            self.log("error", f"LLM 选品失败: {e}")
            chosen = 1

        # Step 3: 获取选中商品的完整详情
        selected = products[chosen - 1]
        product_id = selected["id"]
        self.log("action", f"[Verbal] 获取商品详情: {product_id}")
        detail = self._api_product_detail(product_id)
        if not detail:
            detail = selected

        detail_text = self._format_product_detail(detail)
        self.log("thinking", f"[Verbal] 商品详情:\n{detail_text[:500]}{'...' if len(detail_text) > 500 else ''}")

        # Step 4: LLM 最终推荐判断
        self.log("thinking", "[Verbal] LLM 正在做最终购买判断...")
        try:
            prompt_final = (
                f"The user wants: \"{self.user_query}\"\n\n"
                f"You selected the following product for detailed review:\n\n"
                f"{detail_text}\n\n"
                "Please:\n"
                "1. Summarize the key features of this product.\n"
                "2. Give your recommendation: should the user buy it? Why or why not?"
            )
            messages_final = [
                Message(role="system", content="You are a shopping assistant. Give a purchase recommendation."),
                Message(role="user", content=prompt_final),
            ]
            resp2 = self.agent.llm.generate(messages=messages_final, tools=None)
            final = resp2.content if isinstance(resp2.content, str) else str(resp2.content)
            for line in final.strip().split("\n"):
                if line.strip():
                    self.log("thinking", line.strip())
                    time.sleep(0.2)
            self.log("action", "✅ [Verbal] 最终分析完成")
            self.push_to_viewer("metric", {"name": "step", "value": "verbal 分析完成"})
        except Exception as e:
            self.log("error", f"LLM 最终判断失败: {e}")

    # ==================================================================
    # Visual mode helpers (browser)
    # ==================================================================

    def init_browser(self):
        """初始化 Playwright 浏览器"""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise ImportError("需要安装 playwright: pip install playwright && playwright install")
        
        self.log("action", "初始化浏览器...")
        
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=True,  # 服务器上无头模式
            args=['--no-sandbox']
        )
        self.page = self.browser.new_page(
            viewport={"width": 1280, "height": 800}
        )
        
        self.log("action", "✓ 浏览器已启动")
    
    def navigate_and_capture(self, url: str):
        """导航并截图"""
        self.log("action", f"导航到: {url}")
        
        self.page.goto(url, wait_until="networkidle")
        time.sleep(1)  # 等待渲染完成
        
        screenshot_bytes = self.page.screenshot(type="png")
        
        self.log("action", f"✓ 截图完成 ({len(screenshot_bytes)/1024:.1f} KB)")
        self.push_screenshot(screenshot_bytes, url)
        
        return screenshot_bytes

    def get_product_detail_links_from_search(self) -> List[Tuple[str, str]]:
        """从当前搜索结果页解析商品详情链接，返回 [(product_id, full_url), ...] 按页面顺序"""
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
        """从当前商品详情页提取 Product Description 文本"""
        try:
            loc = self.page.locator(".detail-description .text").first
            if loc.count() == 0:
                return "(页面上未找到描述区域)"
            text = loc.inner_text(timeout=2000).strip()
            return text or "No description available."
        except Exception as e:
            self.log("error", f"提取 description 失败: {e}")
            return "(提取失败)"
    
    def run(self):
        """运行 Agent（自动根据 perception_mode 走 visual 或 verbal 分支）"""
        mode_label = "Visual (截图)" if self.perception_mode == "visual" else "Verbal (文本)"
        print("\n" + "="*80)
        print(f"🤖 Browser Agent 开始运行  [{mode_label}]")
        print("="*80)
        print(f"\n📺 Viewer: {self.web_server_url}/viewer")
        print(f"🛒 用户需求: {self.user_query}")
        print(f"👁 感知模式: {mode_label}")
        print("\n开始执行...\n")

        # Step 1 (共享): Query Understanding
        keywords = self.extract_search_keywords()

        if self.perception_mode == "verbal":
            self.run_verbal(keywords)
        else:
            self.run_visual(keywords)

    def run_visual(self, keywords: str):
        """Visual 模式：浏览器截图 → VLM 决策"""
        search_url = self._build_search_url(keywords)
        try:
            self.init_browser()

            self.log("thinking", "准备访问商品搜索结果页...")
            screenshot = self.navigate_and_capture(search_url)

            product_links = self.get_product_detail_links_from_search()
            num_products = len(product_links)
            self.log("action", f"页面上共 {num_products} 个商品可点进详情")

            # VLM 分析搜索结果截图 → 选品
            self.log("thinking", "VLM 正在分析搜索结果截图...")
            observation = self.agent.perception.encode(screenshot)
            screenshot_data_url = observation.data

            prompt_search = (
                "请仔细查看这个商品搜索结果页的截图，分析：\n"
                "1. 有哪些商品？每个商品的价格和评分。\n"
                "2. 你更想进一步查看哪一个商品的详情（例如看描述、规格）？\n"
                "请只回复一个数字，表示你想查看第几个商品（1 表示第一个，2 表示第二个，以此类推）。"
            )
            if num_products > 0:
                prompt_search += f"\n当前页面有 {num_products} 个商品，请回复 1 到 {num_products} 之间的一个数字。"

            try:
                messages_search = [
                    Message(role="system", content="你是一个购物助手，擅长分析商品页面。只回复一个数字表示要查看第几个商品。"),
                    Message(role="user", content=screenshot_data_url),
                    Message(role="user", content=prompt_search),
                ]
                response = self.agent.llm.generate(messages=messages_search, tools=None)
                analysis = response.content if isinstance(response.content, str) else str(response.content)
                for line in analysis.strip().split("\n"):
                    if line.strip():
                        self.log("thinking", line.strip())
                        time.sleep(0.2)

                chosen = 1
                match = re.search(r"\b([1-9]\d*)\b", analysis)
                if match:
                    chosen = max(1, min(int(match.group(1)), num_products or 1))
                if num_products == 0:
                    chosen = 0

                self.log("action", f"✅ 搜索结果分析完成，选择查看第 {chosen} 个商品")
            except Exception as e:
                self.log("error", f"VLM 分析失败: {str(e)}")
                import traceback
                traceback.print_exc()
                chosen = 0

            # 点进详情页 → VLM 最终判断
            if chosen >= 1 and product_links and chosen <= len(product_links):
                product_id, detail_url = product_links[chosen - 1]
                self.log("action", f"正在打开第 {chosen} 个商品详情页: {product_id}")
                detail_screenshot = self.navigate_and_capture(detail_url)
                description_text = self.get_description_from_detail_page()
                self.log("action", f"已提取商品描述（{len(description_text)} 字）")
                self.log("thinking", f"[Description] {description_text[:300]}{'...' if len(description_text) > 300 else ''}")

                self.log("thinking", "VLM 正在结合详情页与描述做最终判断...")
                try:
                    obs_detail = self.agent.perception.encode(detail_screenshot)
                    detail_data_url = obs_detail.data
                    content_user = (
                        "下面是该商品详情页的截图，以及从页面上提取的 Product Description 文本。\n\n"
                        "【Product Description 文本】\n" + description_text + "\n\n"
                        "请根据详情页截图和上述描述，简要总结该商品特点，并给出你是否推荐购买及理由。"
                    )
                    messages_detail = [
                        Message(role="system", content="你是一个购物助手，根据商品详情页和描述给出购买建议。"),
                        Message(role="user", content=detail_data_url),
                        Message(role="user", content=content_user),
                    ]
                    response2 = self.agent.llm.generate(messages=messages_detail, tools=None)
                    final = response2.content if isinstance(response2.content, str) else str(response2.content)
                    for line in final.strip().split("\n"):
                        if line.strip():
                            self.log("thinking", line.strip())
                            time.sleep(0.2)
                    self.log("action", "✅ [Visual] 基于详情与 description 的最终分析完成")
                    self.push_to_viewer("metric", {"name": "step", "value": "visual 分析完成"})
                except Exception as e2:
                    self.log("error", f"详情页 VLM 分析失败: {str(e2)}")
            else:
                self.push_to_viewer("metric", {"name": "step", "value": "分析完成（未打开详情页）"})

            if self.stay_open:
                print("\n截图已显示在 MacBook 浏览器上。")
                print("按 Ctrl+C 退出...")
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


def main():
    import argparse
    
    # 加载 .env（若存在）到环境变量
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        for line in _env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip("'\"")
                if k and v and k not in os.environ:
                    os.environ[k] = v
    
    parser = argparse.ArgumentParser(description='Browser Agent 实时演示')
    parser.add_argument('--api-key', default=None, help='API Key（不传则从环境变量读取：qwen 用 QWEN_API_KEY/DASHSCOPE_API_KEY，openai 用 OPENAI_API_KEY）')
    parser.add_argument('--llm', choices=['openai', 'qwen'], default='qwen', help='LLM backend')
    parser.add_argument('--perception', choices=['visual', 'verbal'], default='visual',
                        help='感知模式: visual=截图给VLM, verbal=结构化文本给LLM')
    parser.add_argument('--query', default='mousepad', help='用户购物需求（自然语言，Agent 会自动提取搜索关键词）')
    parser.add_argument('--server', default='http://localhost:5000', help='Web 服务器 URL')
    parser.add_argument('--condition-name', default=None, help='实验条件名（需服务端 --condition-file 配合）')
    parser.add_argument('--once', action='store_true', help='完成一次分析后退出')
    
    args = parser.parse_args()
    
    api_key = args.api_key
    if not api_key:
        if args.llm == 'qwen':
            api_key = os.environ.get('QWEN_API_KEY') or os.environ.get('DASHSCOPE_API_KEY')
        else:
            api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print('错误: 需要 API Key。请传 --api-key 或设置环境变量：')
        print('  Qwen: export QWEN_API_KEY=... 或 DASHSCOPE_API_KEY=...')
        print('  OpenAI: export OPENAI_API_KEY=...')
        sys.exit(1)
    
    agent = LiveBrowserAgent(
        llm_api_key=api_key,
        llm_backend=args.llm,
        perception_mode=args.perception,
        web_server_url=args.server,
        user_query=args.query,
        condition_name=args.condition_name,
        stay_open=not args.once,
    )
    
    agent.run()


if __name__ == "__main__":
    main()
