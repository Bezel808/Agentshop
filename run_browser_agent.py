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
        web_server_url: str = "http://localhost:5000",
        target_url: str = "http://localhost:5000/search?q=mousepad",
        stay_open: bool = True,
    ):
        self.web_server_url = web_server_url
        self.target_url = target_url
        self.stay_open = stay_open
        
        # 创建 LLM
        if llm_backend == "qwen":
            llm = QwenBackend(model="qwen-vl-plus", api_key=llm_api_key)  # 使用VL模型支持图像
        elif llm_backend == "openai":
            llm = OpenAIBackend(model="gpt-4o", api_key=llm_api_key)
        else:
            llm = OpenAIBackend(model="gpt-4o", api_key=llm_api_key)
        
        # 创建 Agent
        self.agent = ComposableAgent(
            llm=llm,
            perception=VisualPerception(),
            tools=[],
        )
        
        # Playwright browser
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
        """运行 Agent"""
        print("\n" + "="*80)
        print("🤖 Browser Agent 开始运行")
        print("="*80)
        print(f"\n📺 MacBook 浏览器: {self.web_server_url}/viewer")
        print(f"🎯 目标页面: {self.target_url}")
        print("\n开始执行...\n")
        
        try:
            # 1. 初始化浏览器
            self.init_browser()
            
            # 2. 导航到搜索结果页
            self.log("thinking", "准备访问商品搜索结果页...")
            screenshot = self.navigate_and_capture(self.target_url)
            
            # 3. 解析当前页的商品详情链接（用于后续点进详情）
            product_links = self.get_product_detail_links_from_search()
            num_products = len(product_links)
            self.log("action", f"页面上共 {num_products} 个商品可点进详情")
            
            # 4. VLM 分析搜索结果截图，并让 VLM 选一个要看详情的商品
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
                
                # 解析 VLM 回复中的数字（1-based）
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
            
            # 5. 点进选中商品的详情页，截图并提取 description
            detail_screenshot: Optional[bytes] = None
            description_text = ""
            if chosen >= 1 and product_links and chosen <= len(product_links):
                product_id, detail_url = product_links[chosen - 1]
                self.log("action", f"正在打开第 {chosen} 个商品详情页: {product_id}")
                detail_screenshot = self.navigate_and_capture(detail_url)
                description_text = self.get_description_from_detail_page()
                self.log("action", f"已提取商品描述（{len(description_text)} 字）")
                self.log("thinking", f"[Description] {description_text[:300]}{'...' if len(description_text) > 300 else ''}")
                
                # 6. 第二次 VLM 调用：结合详情页截图与 description 文本
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
                    self.log("action", "✅ 基于详情与 description 的最终分析完成")
                    self.push_to_viewer("metric", {"name": "step", "value": "详情与描述分析完成"})
                except Exception as e2:
                    self.log("error", f"详情页 VLM 分析失败: {str(e2)}")
            else:
                self.push_to_viewer("metric", {"name": "step", "value": "分析完成（未打开详情页）"})
            
            if self.stay_open:
                # 保持截图显示
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
    parser.add_argument('--query', default='mousepad', help='搜索查询')
    parser.add_argument('--server', default='http://localhost:5000', help='Web 服务器 URL')
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
    
    # 构造目标 URL
    target_url = f"{args.server}/search?q={args.query}"
    
    agent = LiveBrowserAgent(
        llm_api_key=api_key,
        llm_backend=args.llm,
        web_server_url=args.server,
        target_url=target_url,
        stay_open=not args.once,
    )
    
    agent.run()


if __name__ == "__main__":
    main()
