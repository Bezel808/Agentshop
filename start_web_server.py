"""
启动 Web 服务器

用于在服务器上启动商品展示页面，支持本地和远程访问。
"""

import sys
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from aces.environments import WebRendererMarketplace
from aces.config.settings import resolve_datasets_dir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def start_server(
    datasets_dir: str = "../ACES/datasets",
    port: int = 5000,
    host: str = "0.0.0.0",  # 0.0.0.0 允许远程访问
    use_llamaindex: bool = True  # 使用 LlamaIndex RAG 检索
):
    """
    启动 Web 服务器
    
    Args:
        datasets_dir: 数据集目录
        port: 端口号（默认 5000）
        host: 监听地址（默认 0.0.0.0，允许远程访问）
    """
    print("\n" + "="*60)
    print("启动 ACES-v2 Web 服务器")
    print("="*60)
    
    # 检查数据集目录
    data_path = resolve_datasets_dir(datasets_dir)
    if not data_path.exists():
        logger.error(f"数据集目录不存在: {data_path}")
        return
    
    logger.info(f"数据集目录: {data_path}")
    
    # 创建 marketplace
    if use_llamaindex:
        logger.info("使用 LlamaIndex RAG 检索")
        from aces.environments import LlamaIndexMarketplace
        marketplace = LlamaIndexMarketplace()
        marketplace.initialize({
            "datasets_dir": str(data_path),
            "use_reranker": True
        })
    else:
        logger.info("使用简单文件名匹配")
        marketplace = WebRendererMarketplace()
        marketplace.initialize({
            "datasets_dir": str(data_path),
            "server_port": port,
            "auto_start_server": False
        })
    
    # 手动启动服务器（支持远程访问）
    try:
        from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse
        from fastapi.staticfiles import StaticFiles
        from fastapi.templating import Jinja2Templates
        import uvicorn
    except ImportError as e:
        logger.error(f"缺少依赖: {e}")
        logger.error("请安装: pip install fastapi uvicorn jinja2")
        return
    
    # 创建 FastAPI app
    app = FastAPI(title="ACES-v2 Product Search")
    
    # WebSocket 连接管理
    active_connections = []
    
    # 挂载静态文件和模板
    static_dir = Path(__file__).parent / "web" / "static"
    templates_dir = Path(__file__).parent / "web" / "templates"
    
    if not static_dir.exists():
        logger.warning(f"静态文件目录不存在: {static_dir}")
    else:
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    if not templates_dir.exists():
        logger.error(f"模板目录不存在: {templates_dir}")
        return
    
    templates = Jinja2Templates(directory=str(templates_dir))
    
    # 定义路由
    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        return templates.TemplateResponse("search.html", {
            "request": request,
            "query": "",
            "products": []
        })
    
    @app.get("/search", response_class=HTMLResponse)
    async def search(request: Request, q: str = ""):
        query = q
        products_data = []
        
        if query:
            # 搜索商品（强制限制为 8 个）
            try:
                results = marketplace.search_products(query, limit=8)
                products_data = [
                    {
                        "id": p.id,
                        "title": p.title,
                        "price": p.price,
                        "rating": p.rating or 0,
                        "rating_count": p.rating_count or 0,
                        "image_url": p.image_url or "",
                        "description": (p.description or "")[:500],
                    }
                    for p in results.products
                ]
                logger.info(f"搜索 '{query}' 返回 {len(products_data)} 个商品")
            except Exception as e:
                logger.error(f"搜索失败: {e}")
        
        return templates.TemplateResponse("search.html", {
            "request": request,
            "query": query,
            "products": products_data,
        })
    
    @app.get("/product/{product_id}", response_class=HTMLResponse)
    async def product_detail(request: Request, product_id: str, q: str = ""):
        """商品详情页：展示 description 等完整信息"""
        product_data = None
        try:
            p = marketplace.get_product_details(product_id)
            product_data = {
                "id": p.id,
                "title": p.title,
                "price": p.price,
                "rating": p.rating or 0,
                "rating_count": p.rating_count or 0,
                "image_url": p.image_url or "",
                "description": p.description or "",
                "sponsored": getattr(p, "sponsored", False),
                "best_seller": getattr(p, "best_seller", False),
                "overall_pick": getattr(p, "overall_pick", False),
            }
        except Exception as e:
            logger.warning(f"商品详情获取失败 {product_id}: {e}")
        return templates.TemplateResponse("product_detail.html", {
            "request": request,
            "product": product_data,
            "query": q,
        })
    
    # 实时查看器页面
    @app.get("/viewer", response_class=HTMLResponse)
    async def viewer(request: Request):
        return templates.TemplateResponse("live_viewer.html", {
            "request": request
        })
    
    # WebSocket 端点（实时推送浏览器截图和日志）
    from fastapi import WebSocket, WebSocketDisconnect
    
    active_viewers: list = []
    
    @app.websocket("/ws/viewer")
    async def websocket_viewer(websocket: WebSocket):
        await websocket.accept()
        active_viewers.append(websocket)
        logger.info(f"Viewer 已连接 (总数: {len(active_viewers)})")
        
        try:
            while True:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            active_viewers.remove(websocket)
            logger.info(f"Viewer 已断开 (剩余: {len(active_viewers)})")
    
    # API 端点：推送截图和日志
    @app.post("/api/push")
    async def push_to_viewers(data: dict):
        """Agent 推送数据到所有 viewer"""
        disconnected = []
        for viewer in active_viewers:
            try:
                await viewer.send_json(data)
            except:
                disconnected.append(viewer)
        
        for v in disconnected:
            if v in active_viewers:
                active_viewers.remove(v)
        
        return {"status": "ok", "viewers": len(active_viewers)}
    
    # 健康检查端点
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "products_loaded": len(marketplace.products),
            "categories": len(list(data_path.glob("*.json"))),
            "active_viewers": len(active_viewers)
        }
    
    # 打印启动信息
    print(f"\n✓ 数据集已加载: {len(marketplace.products)} 个商品")
    print(f"\n服务器配置:")
    print(f"  - 监听地址: {host}:{port}")
    print(f"  - 数据集目录: {data_path}")
    print(f"  - 静态文件: {static_dir}")
    print(f"  - 模板目录: {templates_dir}")
    
    # 获取服务器 IP
    import socket
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"\n访问地址:")
        print(f"  - 本地: http://localhost:{port}")
        print(f"  - 局域网: http://{local_ip}:{port}")
        print(f"  - 远程: http://<服务器IP>:{port}")
    except:
        pass
    
    print(f"\n可用的搜索词:")
    json_files = list(data_path.glob("*.json"))
    for json_file in json_files[:10]:  # 只显示前10个
        print(f"  - {json_file.stem}")
    
    print(f"\n页面:")
    print(f"  - 商品搜索: http://localhost:{port}/search?q=mousepad")
    print(f"  - 实时查看器: http://localhost:{port}/viewer (在 MacBook 打开此页面)")
    print(f"  - 健康检查: http://localhost:{port}/health")
    
    if local_ip and local_ip != "127.0.0.1":
        print(f"\n从 MacBook 访问:")
        print(f"  http://{local_ip}:{port}/viewer  ← 实时查看 Agent 操作")
    
    print(f"\n按 Ctrl+C 停止服务器")
    print("="*60 + "\n")
    
    # 启动服务器
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n服务器已停止")
    except Exception as e:
        logger.error(f"服务器启动失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="启动 ACES-v2 Web 服务器")
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=5000,
        help="端口号（默认: 5000）"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="监听地址（默认: 0.0.0.0，允许远程访问）"
    )
    parser.add_argument(
        "--datasets-dir", "-d",
        type=str,
        default="../ACES/datasets",
        help="数据集目录路径"
    )
    parser.add_argument(
        "--use-llamaindex",
        action="store_true",
        default=True,
        help="使用 LlamaIndex RAG 检索（默认: True）"
    )
    parser.add_argument(
        "--simple-search",
        action="store_true",
        help="使用简单文件名匹配（禁用 LlamaIndex）"
    )
    
    args = parser.parse_args()
    
    # 确定是否使用 LlamaIndex
    use_llamaindex = not args.simple_search
    
    start_server(
        datasets_dir=args.datasets_dir,
        port=args.port,
        host=args.host,
        use_llamaindex=use_llamaindex
    )


if __name__ == "__main__":
    main()
