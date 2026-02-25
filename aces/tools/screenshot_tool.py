"""
Screenshot Tool

MCP 标准的截图工具，用于捕获网页渲染结果。

架构设计:
- 截图功能属于 Tool（本模块）
- 网页渲染由 Environment 提供
- 截图结果通过 Perception Filter 传给 Agent
"""

from typing import Any, Dict
import logging
import time

from aces.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)


class ScreenshotTool(BaseTool):
    """
    截图工具 - 捕获当前商品页面的截图。
    
    特点:
    - 使用 Selenium/Playwright 截取网页
    - 支持多种浏览器（Chrome, Firefox, Edge）
    - 返回 base64 编码的图片数据
    - 可配置截图延迟（等待页面加载）
    
    Usage:
        tool = ScreenshotTool(
            server_url="http://localhost:5000",
            browser="chrome"
        )
        
        result = tool.execute({
            "query": "mousepad",
            "wait_time": 2.0
        })
        
        # result.data 包含 base64 编码的 PNG 图片
    """
    
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:5000",
        browser: str = "chrome",
        headless: bool = True,
    ):
        """
        初始化截图工具。
        
        Args:
            server_url: Web 服务器 URL
            browser: 浏览器类型 ("chrome", "firefox", "edge")
            headless: 是否无头模式
        """
        super().__init__(
            name="capture_screenshot",
            description="Capture a screenshot of the product search page",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to capture screenshot for",
                    },
                    "wait_time": {
                        "type": "number",
                        "description": "Seconds to wait for page load (default: 2)",
                        "default": 2.0,
                    },
                    "window_size": {
                        "type": "object",
                        "description": "Browser window size",
                        "properties": {
                            "width": {"type": "integer", "default": 1280},
                            "height": {"type": "integer", "default": 800},
                        }
                    }
                },
                "required": ["query"],
            }
        )
        
        self.server_url = server_url
        self.browser = browser
        self.headless = headless
        self.driver = None
        
        logger.info(f"Initialized ScreenshotTool (browser={browser}, headless={headless})")
    
    def _execute_impl(self, parameters: Dict[str, Any]) -> Any:
        """捕获截图。"""
        query = parameters["query"]
        wait_time = parameters.get("wait_time", 2.0)
        window_size = parameters.get("window_size", {"width": 1280, "height": 800})
        
        # 初始化浏览器（如果还没有）
        if not self.driver:
            self._init_driver(window_size)
        
        # 导航到搜索页面
        url = f"{self.server_url}/search?q={query}"
        logger.info(f"Navigating to: {url}")
        
        self.driver.get(url)
        
        # 等待页面加载
        time.sleep(wait_time)
        
        # 捕获截图
        screenshot_bytes = self.driver.get_screenshot_as_png()
        
        logger.info(f"Captured screenshot for '{query}' ({len(screenshot_bytes)} bytes)")
        
        # 返回 base64 编码的图片
        import base64
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
        
        return {
            "query": query,
            "screenshot_data": f"data:image/png;base64,{screenshot_base64}",
            "screenshot_bytes": screenshot_bytes,  # 原始字节（供保存文件）
            "format": "png",
            "size_bytes": len(screenshot_bytes),
        }
    
    def _init_driver(self, window_size: Dict[str, int]) -> None:
        """初始化浏览器驱动。"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options as ChromeOptions
            from selenium.webdriver.chrome.service import Service as ChromeService
        except ImportError:
            raise ImportError(
                "Screenshot tool requires 'selenium'. "
                "Install with: pip install selenium"
            )
        
        if self.browser.lower() == "chrome":
            options = ChromeOptions()
            
            if self.headless:
                options.add_argument("--headless")
                options.add_argument("--disable-gpu")
            
            # 常用选项
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument(f"--window-size={window_size['width']},{window_size['height']}")
            
            self.driver = webdriver.Chrome(options=options)
            
        elif self.browser.lower() == "firefox":
            from selenium.webdriver.firefox.options import Options as FirefoxOptions
            
            options = FirefoxOptions()
            if self.headless:
                options.add_argument("--headless")
            
            self.driver = webdriver.Firefox(options=options)
            
        else:
            raise ValueError(f"Unsupported browser: {self.browser}")
        
        logger.info(f"Browser driver initialized: {self.browser}")
    
    def close(self) -> None:
        """关闭浏览器。"""
        if self.driver:
            self.driver.quit()
            self.driver = None
            logger.info("Browser driver closed")
    
    def __del__(self):
        """析构时关闭浏览器。"""
        self.close()


class PlaywrightScreenshotTool(BaseTool):
    """
    使用 Playwright 的截图工具（更快、更稳定）。
    
    Playwright 相比 Selenium 的优势:
    - 更快的启动速度
    - 更好的稳定性
    - 自动等待元素加载
    - 更好的截图质量
    """
    
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:5000",
        browser: str = "chromium",
        headless: bool = True,
    ):
        """
        初始化 Playwright 截图工具。
        
        Args:
            server_url: Web 服务器 URL
            browser: 浏览器类型 ("chromium", "firefox", "webkit")
            headless: 是否无头模式
        """
        super().__init__(
            name="capture_screenshot_playwright",
            description="Capture a screenshot using Playwright (faster and more stable)",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to capture screenshot for",
                    },
                    "wait_time": {
                        "type": "number",
                        "description": "Seconds to wait after page load (default: 1)",
                        "default": 1.0,
                    },
                    "viewport": {
                        "type": "object",
                        "description": "Viewport size",
                        "properties": {
                            "width": {"type": "integer", "default": 1280},
                            "height": {"type": "integer", "default": 800},
                        }
                    }
                },
                "required": ["query"],
            }
        )
        
        self.server_url = server_url
        self.browser_type = browser
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        
        logger.info(f"Initialized PlaywrightScreenshotTool (browser={browser})")
    
    def _execute_impl(self, parameters: Dict[str, Any]) -> Any:
        """使用 Playwright 捕获截图。"""
        query = parameters["query"]
        wait_time = parameters.get("wait_time", 1.0)
        viewport = parameters.get("viewport", {"width": 1280, "height": 800})
        
        # 初始化 Playwright（如果还没有）
        if not self.page:
            self._init_playwright(viewport)
        
        # 导航到搜索页面
        url = f"{self.server_url}/search?q={query}"
        logger.info(f"Navigating to: {url}")
        
        self.page.goto(url, wait_until="networkidle")
        
        # 额外等待（可选）
        if wait_time > 0:
            time.sleep(wait_time)
        
        # 捕获截图
        screenshot_bytes = self.page.screenshot(type="png", full_page=False)
        
        logger.info(f"Captured screenshot for '{query}' ({len(screenshot_bytes)} bytes)")
        
        # 返回 base64 编码的图片
        import base64
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
        
        return {
            "query": query,
            "screenshot_data": f"data:image/png;base64,{screenshot_base64}",
            "screenshot_bytes": screenshot_bytes,
            "format": "png",
            "size_bytes": len(screenshot_bytes),
        }
    
    def _init_playwright(self, viewport: Dict[str, int]) -> None:
        """初始化 Playwright。"""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise ImportError(
                "Playwright screenshot tool requires 'playwright'. "
                "Install with: pip install playwright && playwright install"
            )
        
        self.playwright = sync_playwright().start()
        
        # 启动浏览器
        if self.browser_type == "chromium":
            self.browser = self.playwright.chromium.launch(headless=self.headless)
        elif self.browser_type == "firefox":
            self.browser = self.playwright.firefox.launch(headless=self.headless)
        elif self.browser_type == "webkit":
            self.browser = self.playwright.webkit.launch(headless=self.headless)
        else:
            raise ValueError(f"Unsupported browser: {self.browser_type}")
        
        # 创建上下文和页面
        self.context = self.browser.new_context(
            viewport=viewport,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        self.page = self.context.new_page()
        
        logger.info(f"Playwright initialized: {self.browser_type}")
    
    def close(self) -> None:
        """关闭 Playwright。"""
        if self.context:
            self.context.close()
            self.context = None
        
        if self.browser:
            self.browser.close()
            self.browser = None
        
        if self.playwright:
            self.playwright.stop()
            self.playwright = None
        
        logger.info("Playwright closed")
    
    def __del__(self):
        """析构时关闭。"""
        self.close()
