"""
Web Renderer Marketplace Provider

扩展离线市场环境，添加网页渲染功能。
启动一个 FastAPI 服务器来渲染商品页面，供截图工具使用。

架构设计:
- 网页渲染属于 Environment（本模块）
- 截图属于 Tool（ScreenshotTool）
- 截图结果经过 Perception Filter 给到 Agent
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from aces.environments.protocols import (
    MarketplaceProvider,
    MarketplaceMode,
    Product,
    SearchResult,
    PageState,
)
from aces.config.settings import resolve_datasets_dir
from aces.environments.product_utils import product_from_dict

logger = logging.getLogger(__name__)


class WebRendererMarketplace(MarketplaceProvider):
    """
    带网页渲染功能的离线市场环境。
    
    特点:
    - 继承离线环境的所有功能（数据加载、搜索、干预）
    - 额外启动 FastAPI 服务器渲染商品页面
    - 支持截图工具访问渲染页面
    - 服务器在后台运行，不阻塞主流程
    
    Usage:
        marketplace = WebRendererMarketplace()
        marketplace.initialize({
            "datasets_dir": "../ACES/datasets",
            "server_port": 5000,
            "auto_start_server": True
        })
        
        # 搜索商品（同时更新网页内容）
        results = marketplace.search_products("mousepad")
        
        # 截图工具可以访问 http://localhost:5000/search?q=mousepad
    """
    
    def __init__(self):
        """初始化 Web Renderer marketplace."""
        self.datasets_dir: Optional[Path] = None
        self.server_port = 5000
        self.server_url: Optional[str] = None
        self.server_thread: Optional[threading.Thread] = None
        self.app = None
        
        # Product data
        self.products: List[Product] = []
        self.product_lookup: Dict[str, Product] = {}
        
        # Current state
        self.current_query: Optional[str] = None
        self.current_results: List[Product] = []
        self.cart: List[Dict] = []
        
        # Web page data (shared with FastAPI)
        self._web_page_data: Optional[List[Dict]] = None
        
        logger.info("Initialized WebRendererMarketplace")
    
    def get_current_page_url(self) -> str:
        """获取当前搜索页面的 URL（供截图工具使用）"""
        if not self.server_url:
            return f"http://localhost:{self.server_port}/"
        
        if not self.current_query:
            return f"{self.server_url}/"
        
        return f"{self.server_url}/search?q={self.current_query}"
    
    def get_screenshot_with_playwright(
        self, 
        query: str = None,
        viewport: Dict[str, int] = None,
        wait_time: float = 1.0
    ) -> bytes:
        """
        使用 Playwright 截取当前搜索页面。
        
        Args:
            query: 搜索查询（如果为None则使用当前查询）
            viewport: 视口大小 {"width": 1280, "height": 800}
            wait_time: 等待时间（秒）
            
        Returns:
            Screenshot bytes (PNG)
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise ImportError(
                "Playwright screenshot requires 'playwright'. "
                "Install with: pip install playwright && playwright install"
            )
        
        if viewport is None:
            viewport = {"width": 1280, "height": 800}
        
        # 确定 URL
        if query:
            url = f"{self.server_url or f'http://localhost:{self.server_port}'}/search?q={query}"
        else:
            url = self.get_current_page_url()
        
        logger.info(f"Capturing screenshot from {url}")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport=viewport,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            page = context.new_page()
            
            # 导航到页面
            page.goto(url, wait_until="networkidle", timeout=30000)
            
            # 额外等待
            if wait_time > 0:
                time.sleep(wait_time)
            
            # 截图
            screenshot_bytes = page.screenshot(type="png", full_page=False)
            
            browser.close()
        
        logger.info(f"Screenshot captured ({len(screenshot_bytes)} bytes)")
        return screenshot_bytes
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        初始化并启动 web 服务器。
        
        Config:
            datasets_dir: ACES 数据集目录
            server_port: Web 服务器端口（默认: 5000）
            auto_start_server: 是否自动启动服务器（默认: True）
        """
        self.datasets_dir = resolve_datasets_dir(config.get("datasets_dir"))
        self.server_port = config.get("server_port", 5000)
        auto_start = config.get("auto_start_server", True)
        
        logger.info(f"Loading products from {self.datasets_dir}...")
        
        # 加载所有商品
        self.products = self._load_all_products()
        self.product_lookup = {p.id: p for p in self.products}
        
        logger.info(f"Loaded {len(self.products)} products")
        
        # 启动 web 服务器
        if auto_start:
            self.start_server()
    
    def start_server(self) -> None:
        """启动 FastAPI web 服务器（后台线程）。"""
        if self.server_thread and self.server_thread.is_alive():
            logger.warning("Server already running")
            return
        
        # Lazy import to avoid dependency if not used
        try:
            from fastapi import FastAPI, Request
            from fastapi.responses import HTMLResponse
            from fastapi.staticfiles import StaticFiles
            from fastapi.templating import Jinja2Templates
            import uvicorn
        except ImportError:
            raise ImportError(
                "Web renderer requires 'fastapi' and 'uvicorn'. "
                "Install with: pip install fastapi uvicorn jinja2"
            )
        
        # 创建 FastAPI app
        self.app = FastAPI()
        
        # 挂载静态文件和模板
        static_dir = Path(__file__).parent.parent.parent / "web" / "static"
        templates_dir = Path(__file__).parent.parent.parent / "web" / "templates"
        
        if static_dir.exists():
            self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        
        templates = Jinja2Templates(directory=str(templates_dir))
        
        # 定义路由
        @self.app.get("/", response_class=HTMLResponse)
        async def home(request: Request):
            return templates.TemplateResponse("search.html", {"request": request})
        
        @self.app.get("/search", response_class=HTMLResponse)
        async def search(request: Request, q: str = ""):
            query = q
            
            if query:
                # 获取当前网页数据（由 search_products 更新）
                products_data = self._web_page_data or []
            else:
                products_data = []
            
            return templates.TemplateResponse(
                "search.html",
                {
                    "request": request,
                    "query": query,
                    "products": products_data,
                }
            )
        
        # 在后台线程启动服务器
        def run_server():
            config = uvicorn.Config(
                self.app,
                host="127.0.0.1",
                port=self.server_port,
                log_level="warning"
            )
            server = uvicorn.Server(config)
            server.run()
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        # 等待服务器启动
        time.sleep(2)
        
        self.server_url = f"http://127.0.0.1:{self.server_port}"
        logger.info(f"Web server started at {self.server_url}")
    
    def stop_server(self) -> None:
        """停止 web 服务器。"""
        if self.server_thread and self.server_thread.is_alive():
            logger.info("Stopping web server...")
            # Note: uvicorn server in thread can't be easily stopped
            # In production, use a proper process manager
            self.server_thread = None
            self.server_url = None
    
    def search_products(
        self,
        query: str,
        sort_by: str = "relevance",
        limit: int = 10,
        **kwargs
    ) -> SearchResult:
        """
        搜索商品并更新网页内容。
        
        这个方法会：
        1. 从数据集加载匹配的商品
        2. 更新网页数据（供截图工具使用）
        3. 返回搜索结果
        """
        self.current_query = query
        
        # 简单文件名匹配（与 ACES v1 兼容）
        json_file = self.datasets_dir / f"{query.lower().replace(' ', '_')}.json"
        
        products = []
        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    products_data = json.load(f)
                
                for idx, data in enumerate(products_data):
                    product = product_from_dict(
                        data,
                        index=idx,
                        source="web_renderer",
                        default_id_prefix=query,
                    )
                    products.append(product)
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")
        
        # 排序
        if sort_by == "price_asc":
            products.sort(key=lambda p: p.price)
        elif sort_by == "price_desc":
            products.sort(key=lambda p: p.price, reverse=True)
        elif sort_by == "rating" and products:
            products.sort(key=lambda p: p.rating or 0, reverse=True)
        
        # 限制返回数量
        products = products[:limit]
        
        # 更新位置
        for i, p in enumerate(products):
            p.position = i
        
        self.current_results = products
        
        # 更新网页数据（供 FastAPI 使用）
        self._web_page_data = [
            {
                "title": p.title,
                "price": p.price,
                "rating": p.rating or 0,
                "rating_count": p.rating_count or 0,
                "image_url": p.image_url or "",
            }
            for p in products
        ]
        
        logger.info(f"Search '{query}' returned {len(products)} results, web page updated")
        
        return SearchResult(
            query=query,
            products=products,
            total_count=len(products),
            metadata={
                "mode": "web_renderer",
                "server_url": self.server_url,
            }
        )
    
    def get_product_details(self, product_id: str) -> Product:
        """获取商品详情。"""
        if product_id in self.product_lookup:
            return self.product_lookup[product_id]
        
        raise ValueError(f"Product {product_id} not found")
    
    def get_page_state(self) -> PageState:
        """获取当前页面状态。"""
        return PageState(
            products=self.current_results,
            query=self.current_query,
            metadata={
                "mode": "web_renderer",
                "server_url": self.server_url,
            }
        )
    
    def add_to_cart(self, product_id: str, quantity: int = 1) -> Dict[str, Any]:
        """加入购物车。"""
        product = self.get_product_details(product_id)
        
        self.cart.append({
            "product_id": product_id,
            "product_title": product.title,
            "quantity": quantity,
            "price": product.price,
        })
        
        cart_total = sum(item["price"] * item["quantity"] for item in self.cart)
        
        logger.info(f"Added {quantity}x '{product.title}' to cart")
        
        return {
            "success": True,
            "cart": self.cart,
            "cart_total": cart_total,
        }
    
    def reset(self) -> PageState:
        """重置状态。"""
        self.current_query = None
        self.current_results = []
        self.cart = []
        self._web_page_data = None
        
        return PageState(products=[], metadata={"mode": "web_renderer"})
    
    def get_mode(self) -> MarketplaceMode:
        """返回模式。"""
        return MarketplaceMode.OFFLINE
    
    def close(self) -> None:
        """清理资源。"""
        self.stop_server()
        logger.info("Closed Web Renderer marketplace")
    
    # ========================================================================
    # Private Helper Methods
    # ========================================================================
    
    def _load_all_products(self) -> List[Product]:
        """从所有 JSON 文件加载商品。"""
        all_products = []
        
        if not self.datasets_dir.exists():
            logger.warning(f"Datasets directory not found: {self.datasets_dir}")
            return []
        
        # 遍历所有 JSON 文件
        for json_file in self.datasets_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    products_data = json.load(f)
                
                category = json_file.stem  # 文件名作为类别
                
                for idx, data in enumerate(products_data):
                    data_with_category = {"category": category, **data}
                    product = product_from_dict(
                        data_with_category,
                        index=idx,
                        source="web_renderer",
                        category=category,
                    )
                    all_products.append(product)
            
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")
                continue
        
        return all_products
