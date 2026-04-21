"""
启动 Web 服务器

用于在服务器上启动商品展示页面，支持本地和远程访问。
"""

import os
import json
import re
import sys
import logging
import math
from pathlib import Path
from urllib.parse import quote

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from aces.environments import WebRendererMarketplace
from aces.config.settings import resolve_datasets_dir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
DEFAULT_WEB_MAX_PAGES = 5   # 每次最多展示 N 页（默认 5 页，每页 8 个 = 最多 40 个商品）


def _normalize_image_urls(raw_data: dict | None, primary_image: str = "") -> list[str]:
    """从多种字段结构里提取图片列表，兼容旧/新数据格式。"""
    if not raw_data:
        return [primary_image] if primary_image else []

    urls: list[str] = []

    def _add(url: str | None):
        if isinstance(url, str) and url.startswith("http") and url not in urls:
            urls.append(url)

    # 直接数组字段
    for key in ("description_images", "image_urls", "gallery_images", "gallery", "images"):
        value = raw_data.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    _add(item)
                elif isinstance(item, dict):
                    for image_key in ("large", "hi_res", "thumb", "url"):
                        _add(item.get(image_key))

    # Amazon Reviews 2023 常见结构: {"images": {"large": [...], "hi_res": [...], "thumb": [...]} }
    images_obj = raw_data.get("images")
    if isinstance(images_obj, dict):
        for key in ("large", "hi_res", "thumb"):
            image_list = images_obj.get(key)
            if isinstance(image_list, list):
                for url in image_list:
                    _add(url)

    # 保证主图优先
    if primary_image:
        urls = [primary_image] + [u for u in urls if u != primary_image]

    return urls


def _amazon_image_id(url: str) -> str | None:
    """Extract Amazon image ID (e.g. 31hfCPDzuiL) from URL. Returns None if not Amazon."""
    m = re.search(r'/I/([^/]+)\._', url)
    return m.group(1) if m else None


def _image_resolution_score(url: str) -> int:
    """Higher = prefer this URL when deduping same image in different sizes."""
    if "_SL1500_" in url or "_UL1500_" in url:
        return 5
    if "_SL1000_" in url or "_UL1000_" in url or "_SL1002_" in url:
        return 4
    if "_SL" in url or "_UL" in url:
        return 3
    if "_AC_" in url and "_US40_" not in url and "_SR" not in url:
        return 2
    if "_AC_" in url:
        return 1
    return 0


def _is_amazon_high_res(url: str) -> bool:
    """True if URL is a high-res variant (one per visual - avoids 51xx/81xx duplicate)."""
    return bool(
        re.search(r'_[SU]L1500_|_[SU]L1000_|_[SU]L1002_', url)
    )


def _dedupe_image_urls(urls: list[str]) -> list[str]:
    """Deduplicate: (1) same ID different sizes -> keep best. (2) Prefer high-res only to avoid 51xx/81xx same-image duplicates."""
    if not urls:
        return []
    high_res = [u for u in urls if (u or "").strip().startswith("http") and _is_amazon_high_res((u or "").strip())]
    if high_res:
        # Use only high-res URLs: each is one per visual, no 51xx/81xx duplicates
        by_id = {}
        for url in high_res:
            url = url.strip()
            aid = _amazon_image_id(url)
            key = aid or url
            score = _image_resolution_score(url)
            if key not in by_id or score > by_id[key][0]:
                by_id[key] = (score, url)
        return [v[1] for v in by_id.values()]

    # No high-res: fallback to per-ID dedupe
    by_id = {}
    for url in urls:
        url = (url or "").strip()
        if not url or not url.startswith("http"):
            continue
        aid = _amazon_image_id(url)
        if aid:
            score = _image_resolution_score(url)
            if aid not in by_id or score > by_id[aid][0]:
                by_id[aid] = (score, url)
        else:
            by_id[url] = (0, url)
    return [v[1] for v in by_id.values()]


def _to_search_image_url(url: str) -> str:
    """列表页使用更轻量的缩略图，减少首屏等待时间。"""
    if not isinstance(url, str):
        return ""
    source = url.strip()
    if not source or not source.startswith("http"):
        return ""

    # Amazon 图片通常可通过规格后缀切到更小尺寸（例如 _AC_SX300_）。
    if "m.media-amazon.com/images/" in source:
        return re.sub(
            r"\._[^.]*\.(jpg|jpeg|png|webp)$",
            r"._AC_SX300_.\1",
            source,
            flags=re.IGNORECASE,
        )
    return source


def _normalize_reviews(raw_data: dict | None) -> list[dict]:
    """提取并标准化评论结构，统一为 title/text/rating/author/date。"""
    if not raw_data:
        return []

    raw_reviews = (
        raw_data.get("reviews")
        or raw_data.get("review_snippets")
        or raw_data.get("comments")
        or []
    )

    normalized: list[dict] = []
    for item in raw_reviews:
        if isinstance(item, str):
            text = item.strip()
            if text:
                normalized.append({"title": "", "text": text, "rating": None, "author": "", "date": ""})
            continue

        if isinstance(item, dict):
            text = (
                item.get("text")
                or item.get("content")
                or item.get("body")
                or item.get("review_text")
                or ""
            )
            title = item.get("title") or item.get("summary") or ""
            rating = item.get("rating") or item.get("score")
            author = item.get("author") or item.get("user") or item.get("user_id") or ""
            date = item.get("date") or item.get("time") or item.get("timestamp") or ""

            if text or title:
                raw_review_images = item.get("review_images") or item.get("images") or []
                review_images = []
                if isinstance(raw_review_images, list):
                    for img in raw_review_images:
                        if isinstance(img, str) and img.startswith("http"):
                            review_images.append(img)
                        elif isinstance(img, dict):
                            for k in ("large_image_url", "medium_image_url", "small_image_url", "url"):
                                v = img.get(k)
                                if isinstance(v, str) and v.startswith("http"):
                                    review_images.append(v)
                review_images = list(dict.fromkeys(review_images))
                normalized.append(
                    {
                        "title": str(title).strip(),
                        "text": str(text).strip(),
                        "rating": rating,
                        "author": str(author).strip(),
                        "date": str(date).strip(),
                        "review_images": review_images,
                    }
                )

    return normalized


def _count_product_images(product: dict | None) -> int:
    """统计商品可用图片数量（优先 image_urls/description_images，回退到 image_url）。"""
    if not isinstance(product, dict):
        return 0
    urls: list[str] = []
    for key in ("image_urls", "description_images"):
        value = product.get(key)
        if isinstance(value, list):
            for u in value:
                if isinstance(u, str) and u.startswith("http") and u not in urls:
                    urls.append(u)
    if urls:
        return len(urls)
    image_url = product.get("image_url")
    return 1 if isinstance(image_url, str) and image_url.startswith("http") else 0


def _count_product_reviews(product: dict | None) -> int:
    """统计商品评论数。"""
    if not isinstance(product, dict):
        return 0
    reviews = product.get("reviews")
    return len(reviews) if isinstance(reviews, list) else 0


def _apply_price_rating_filter(
    products: list,
    price_min: float | None = None,
    price_max: float | None = None,
    rating_min: float | None = None,
) -> list:
    """按价格和星级筛选商品，排序不变。"""
    filtered = []
    for p in products:
        price = getattr(p, "price", 0) or 0
        rating = getattr(p, "rating", None) or 0
        if price_min is not None and price < price_min:
            continue
        if price_max is not None and price > price_max:
            continue
        if rating_min is not None and rating < rating_min:
            continue
        filtered.append(p)
    return filtered


def _normalize_non_negative_float(v) -> float | None:
    """解析为非负浮点数，空值/非法值返回 None。"""
    if v is None or (isinstance(v, str) and not str(v).strip()):
        return None
    try:
        f = float(v)
        return f if f >= 0 else None
    except (TypeError, ValueError):
        return None


def _normalize_filter_bounds(
    price_min,
    price_max,
    rating_min,
) -> tuple[float | None, float | None, float | None]:
    """标准化筛选边界，并在上下限反转时自动纠正。"""
    pmin = _normalize_non_negative_float(price_min)
    pmax = _normalize_non_negative_float(price_max)
    rmin = _normalize_non_negative_float(rating_min)
    if pmin is not None and pmax is not None and pmin > pmax:
        pmin, pmax = pmax, pmin
    return pmin, pmax, rmin


def _compute_display_pages(current: int, total: int, window: int = 2) -> list:
    """生成分页显示的页码列表，可含 'ellipsis' 表示省略。"""
    if total <= 9:
        return list(range(1, total + 1))
    pages = []
    if current <= window + 2:
        pages.extend(range(1, min(current + window + 1, total + 1)))
        if total > current + window + 1:
            pages.append("ellipsis")
            pages.append(total)
    elif current >= total - window - 1:
        pages.append(1)
        if current - window - 1 > 1:
            pages.append("ellipsis")
        pages.extend(range(max(1, current - window), total + 1))
    else:
        pages.append(1)
        pages.append("ellipsis")
        pages.extend(range(current - window, current + window + 1))
        pages.append("ellipsis")
        pages.append(total)
    return pages


def _expand_products_for_pagination(
    products: list,
    query: str,
    all_products: list,
    target_count: int,
) -> list:
    """当搜索结果不足时，只补充与查询词相关的候选，避免引入无关商品。"""
    if len(products) >= target_count:
        return products

    existing_ids = {p.id for p in products}
    query_terms = [t for t in query.lower().replace("_", " ").split() if t]

    scored_candidates = []
    for p in all_products:
        if p.id in existing_ids:
            continue
        title_text = (p.title or "").lower()
        desc_text = (p.description or "").lower()
        title_score = 0
        desc_score = 0
        for term in query_terms:
            if term in title_text:
                title_score += 1
            if term in desc_text:
                desc_score += 1
        # 仅在标题命中时作为补齐候选，避免引入语义弱相关商品。
        if title_score > 0:
            score = title_score * 10 + desc_score
            scored_candidates.append((score, p))

    # 先补最相关，再补剩余任意商品
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    for _, p in scored_candidates:
        products.append(p)
        existing_ids.add(p.id)
        if len(products) >= target_count:
            return products

    return products


def _is_product_dataset_file(json_file: Path) -> bool:
    """仅接受 list[dict] 结构的数据文件，自动跳过质量报告等元数据文件。"""
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    return isinstance(data, list) and all(isinstance(item, dict) for item in data)


def _list_product_dataset_files(data_path: Path) -> list[Path]:
    return [fp for fp in sorted(data_path.glob("*.json")) if _is_product_dataset_file(fp)]


def start_server(
    datasets_dir: str = "../ACES/datasets",
    port: int = 5000,
    host: str = "0.0.0.0",  # 0.0.0.0 允许远程访问
    use_llamaindex: bool = True,  # 使用 LlamaIndex RAG 检索
    condition_file: str | None = None,  # 实验条件 YAML 文件路径
    max_pages: int = DEFAULT_WEB_MAX_PAGES,  # 每次最多展示 N 页
    index_cache_dir: str = ".cache/llamaindex",
    rebuild_index: bool = False,
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
    product_dataset_files = _list_product_dataset_files(data_path)
    skipped_count = len(list(data_path.glob("*.json"))) - len(product_dataset_files)
    if skipped_count > 0:
        logger.warning(f"已跳过 {skipped_count} 个非商品数据文件（如报告/元数据 JSON）")
    
    # 创建 marketplace
    if use_llamaindex:
        logger.info("使用 LlamaIndex RAG 检索")
        from aces.environments import LlamaIndexMarketplace
        marketplace = LlamaIndexMarketplace()
        marketplace.initialize({
            "datasets_dir": str(data_path),
            "use_reranker": True,
            "index_cache_dir": index_cache_dir,
            "rebuild_index": rebuild_index,
        })
    else:
        logger.info("使用简单文件名匹配")
        marketplace = WebRendererMarketplace()
        marketplace.initialize({
            "datasets_dir": str(data_path),
            "server_port": port,
            "auto_start_server": False
        })

    # 加载实验条件（可选）
    condition_set = None
    if condition_file:
        cond_path = Path(condition_file)
        if not cond_path.is_absolute():
            cond_path = Path(__file__).parent / cond_path
        if cond_path.exists():
            try:
                from aces.experiments.control_variables import load_conditions_from_yaml, load_conditions_from_json
                if cond_path.suffix in (".yaml", ".yml"):
                    condition_set = load_conditions_from_yaml(str(cond_path))
                else:
                    condition_set = load_conditions_from_json(str(cond_path))
                logger.info(f"已加载实验条件: {condition_set.name}, 条件: {condition_set.all_condition_names()}")
            except Exception as e:
                logger.warning(f"加载条件文件失败 {condition_file}: {e}")
        else:
            logger.warning(f"条件文件不存在: {cond_path}")

    def _product_to_dict(p, desc_max_len: int = 500):
        """将 Product 转为模板用 dict，含标签和原价"""
        raw_data = p.raw_data or {}
        primary = (p.image_url or "").strip()
        image_urls = _normalize_image_urls(raw_data, primary)
        # gallery_images：排除主图 + 同图多分辨率去重
        gallery_images = [u for u in image_urls if u and u.strip() != primary]
        gallery_images = _dedupe_image_urls(gallery_images)
        description_images = _normalize_image_urls(raw_data, "")
        reviews = _normalize_reviews(raw_data)

        d = {
            "id": p.id,
            "title": p.title,
            "price": p.price,
            "rating": p.rating or 0,
            "rating_count": p.rating_count or 0,
            "image_url": primary or "",
            "search_image_url": _to_search_image_url(primary or ""),
            "image_urls": image_urls,
            "gallery_images": gallery_images,
            "description_images": description_images,
            "description": (p.description or "")[:desc_max_len],
            "sponsored": getattr(p, "sponsored", False),
            "best_seller": getattr(p, "best_seller", False),
            "overall_pick": getattr(p, "overall_pick", False),
            "low_stock": getattr(p, "low_stock", False),
            "custom_badges": raw_data.get("custom_badges", []),
            "original_price": raw_data.get("original_price"),
            "reviews": reviews,
        }
        return d

    # 手动启动服务器（支持远程访问）
    try:
        from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
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
    viewer_token = (os.getenv("ACES_VIEWER_TOKEN") or "").strip()

    if viewer_token:
        logger.info("Viewer token 鉴权已启用")
    else:
        logger.warning("ACES_VIEWER_TOKEN 未设置，Viewer 相关接口将拒绝访问")

    def _extract_http_token(request: Request) -> str:
        header_token = (request.headers.get("X-ACES-Token") or "").strip()
        if header_token:
            return header_token
        return (request.query_params.get("token") or "").strip()

    def _extract_ws_token(websocket: WebSocket) -> str:
        header_token = (websocket.headers.get("x-aces-token") or "").strip()
        if header_token:
            return header_token
        return (websocket.query_params.get("token") or "").strip()

    def _require_viewer_token_http(request: Request):
        if not viewer_token:
            raise HTTPException(status_code=401, detail="ACES_VIEWER_TOKEN 未配置")
        if _extract_http_token(request) != viewer_token:
            raise HTTPException(status_code=401, detail="unauthorized")

    async def _require_viewer_token_ws(websocket: WebSocket) -> bool:
        if not viewer_token or _extract_ws_token(websocket) != viewer_token:
            await websocket.close(code=1008)
            return False
        return True
    
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
            "products": [],
            "filter_params": {},
        })
    
    @app.get("/search", response_class=HTMLResponse)
    async def search(
        request: Request,
        q: str = "",
        condition_name: str = None,
        page: int = 1,
        page_size: int = 8,
        price_min: str | float | None = None,
        price_max: str | float | None = None,
        rating_min: str | float | None = None,
    ):
        query = q
        products_data = []
        cond_params = {}
        filter_params = {}
        pagination = None

        pmin, pmax, rmin = _normalize_filter_bounds(price_min, price_max, rating_min)
        if pmin is not None:
            filter_params["price_min"] = pmin
        if pmax is not None:
            filter_params["price_max"] = pmax
        if rmin is not None:
            filter_params["rating_min"] = rmin

        if query:
            try:
                page = max(page, 1)
                page_size = max(1, min(page_size, 40))
                filter_active = any(v is not None for v in (pmin, pmax, rmin))
                all_products_list = list(getattr(marketplace, "products", []))

                # 启用筛选时扩大候选池，避免“先截断后筛选”导致单边筛选无结果。
                fetch_limit_base = page_size * max(max_pages, page)
                fetch_limit = fetch_limit_base
                if filter_active:
                    if all_products_list:
                        fetch_limit = max(fetch_limit_base, len(all_products_list))
                    else:
                        fetch_limit = fetch_limit_base * 5
                results = marketplace.search_products(query, limit=fetch_limit)
                products = list(results.products)

                # 应用实验条件
                if condition_set and condition_name:
                    cond = next((c for c in condition_set.conditions if c.name == condition_name), None)
                    if cond:
                        products = cond.apply(products)
                        cond_params = {"condition_name": condition_name}
                        logger.info(f"已应用条件 '{condition_name}'")

                # 兼容小数据集：自动从全库补齐
                target_count = min(page_size * max_pages, len(all_products_list) if all_products_list else page_size)
                if filter_active and all_products_list:
                    target_count = len(all_products_list)
                if len(products) < target_count and all_products_list:
                    products = _expand_products_for_pagination(
                        products=products,
                        query=query,
                        all_products=all_products_list,
                        target_count=target_count,
                    )

                # 应用价格和星级筛选
                products = _apply_price_rating_filter(
                    products,
                    price_min=pmin,
                    price_max=pmax,
                    rating_min=rmin,
                )

                # 最多展示 N 页，超出部分截断
                max_items = page_size * max_pages
                products = products[:max_items]
                total_items = len(products)
                total_pages = max(1, min(math.ceil(total_items / page_size), max_pages)) if total_items else 1
                page = min(page, total_pages)
                start_idx = (page - 1) * page_size
                end_idx = start_idx + page_size
                paged_products = products[start_idx:end_idx]

                products_data = [_product_to_dict(p) for p in paged_products]
                # 计算分页显示的页码（页数多时只显示部分，避免过长）
                display_pages = _compute_display_pages(page, total_pages)
                pagination = {
                    "page": page,
                    "page_size": page_size,
                    "total_items": total_items,
                    "total_pages": total_pages,
                    "has_prev": page > 1,
                    "has_next": page < total_pages,
                    "prev_page": max(page - 1, 1),
                    "next_page": min(page + 1, total_pages),
                    "display_pages": display_pages,
                }
                logger.info(f"搜索 '{query}' 返回 {len(products_data)} 个商品（第 {page}/{total_pages} 页）")
            except Exception as e:
                logger.error(f"搜索失败: {e}")

        return templates.TemplateResponse("search.html", {
            "request": request,
            "query": query,
            "products": products_data,
            "condition_params": cond_params,
            "filter_params": filter_params,
            "pagination": pagination,
        })
    
    @app.get("/product/{product_id}", response_class=HTMLResponse)
    async def product_detail(
        request: Request,
        product_id: str,
        q: str = "",
        condition_name: str = None,
        page: int = 1,
        page_size: int = 8,
        price_min: float | None = None,
        price_max: float | None = None,
        rating_min: float | None = None,
    ):
        """商品详情页：展示 description 等完整信息，支持条件变量"""
        product_data = None
        cond_params = {}
        try:
            p = marketplace.get_product_details(product_id)
            # 应用实验条件
            if condition_set and condition_name:
                cond = next((c for c in condition_set.conditions if c.name == condition_name), None)
                if cond:
                    p = cond.apply_single_product(p)
                    cond_params = {"condition_name": condition_name}
            product_data = _product_to_dict(p, desc_max_len=99999)
        except Exception as e:
            logger.warning(f"商品详情获取失败 {product_id}: {e}")
        price_min, price_max, rating_min = _normalize_filter_bounds(price_min, price_max, rating_min)
        filter_params = {}
        if price_min is not None:
            filter_params["price_min"] = price_min
        if price_max is not None:
            filter_params["price_max"] = price_max
        if rating_min is not None:
            filter_params["rating_min"] = rating_min
        return templates.TemplateResponse("product_detail.html", {
            "request": request,
            "product": product_data,
            "query": q,
            "page": page,
            "page_size": page_size,
            "condition_params": cond_params,
            "filter_params": filter_params,
        })
    
    # 实时查看器页面
    @app.get("/viewer", response_class=HTMLResponse)
    async def viewer(request: Request):
        _require_viewer_token_http(request)
        return templates.TemplateResponse("live_viewer.html", {
            "request": request
        })

    @app.get("/dataset-admin", response_class=HTMLResponse)
    async def dataset_admin(request: Request):
        """数据集人工清洗页面（删除不符合类目要求的商品）。"""
        _require_viewer_token_http(request)
        return templates.TemplateResponse("dataset_admin.html", {
            "request": request
        })

    @app.get("/dataset-adminr", response_class=HTMLResponse)
    async def dataset_admin_typo_alias(request: Request):
        """兼容常见拼写错误: /dataset-adminr -> /dataset-admin."""
        return await dataset_admin(request)
    
    # WebSocket 端点（实时推送浏览器截图和日志）
    active_viewers: list = []
    
    @app.websocket("/ws/viewer")
    async def websocket_viewer(websocket: WebSocket):
        if not await _require_viewer_token_ws(websocket):
            return
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

    def _category_to_file(category: str) -> Path:
        cat = (category or "").strip()
        if not cat:
            raise HTTPException(status_code=400, detail="category 不能为空")
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", cat):
            raise HTTPException(status_code=400, detail="category 非法")
        fp = data_path / f"{cat}.json"
        if not fp.exists():
            raise HTTPException(status_code=404, detail=f"类目不存在: {cat}")
        if not _is_product_dataset_file(fp):
            raise HTTPException(status_code=400, detail=f"非商品数据文件: {cat}")
        return fp

    def _load_category_products(category: str) -> list[dict]:
        fp = _category_to_file(category)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"读取失败: {e}") from e
        if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
            raise HTTPException(status_code=500, detail="类目数据结构非法（必须是 list[dict]）")
        return data

    def _save_category_products(category: str, products: list[dict]):
        fp = _category_to_file(category)
        try:
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(products, f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"写入失败: {e}") from e

    def _reload_marketplace_products():
        """删除后刷新内存商品表，保证立即生效。"""
        try:
            if hasattr(marketplace, "_load_all_products"):
                products = marketplace._load_all_products()
                marketplace.products = products
                marketplace.product_lookup = {p.id: p for p in products}
                logger.info(f"已刷新内存商品索引: {len(products)} items")
        except Exception as e:
            logger.warning(f"刷新内存商品索引失败: {e}")

    @app.get("/api/admin/categories")
    async def admin_categories(request: Request):
        _require_viewer_token_http(request)
        items = []
        for fp in _list_product_dataset_files(data_path):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    arr = json.load(f)
                count = len(arr) if isinstance(arr, list) else 0
            except Exception:
                count = 0
            items.append({"name": fp.stem, "count": count})
        return {"categories": items}

    @app.get("/api/admin/products")
    async def admin_products(
        request: Request,
        category: str,
        q: str = "",
        page: int = 1,
        page_size: int = 30,
    ):
        _require_viewer_token_http(request)
        products = _load_category_products(category)
        page = max(1, int(page))
        page_size = max(1, min(int(page_size), 200))
        keyword = (q or "").strip().lower()

        indexed = list(enumerate(products))
        if keyword:
            def _match(item: tuple[int, dict]) -> bool:
                _, p = item
                text = f"{p.get('title', '')} {p.get('description', '')} {p.get('sku', '')}".lower()
                return keyword in text
            indexed = [x for x in indexed if _match(x)]

        total = len(indexed)
        start = (page - 1) * page_size
        end = start + page_size
        paged = indexed[start:end]

        rows = []
        for idx, p in paged:
            product_id = str(p.get("id") or p.get("sku") or "").strip()
            detail_url = (
                f"/product/{quote(product_id, safe='')}?q={quote(category)}&page=1&page_size=8"
                if product_id else ""
            )
            rows.append({
                "index": idx,
                "sku": p.get("sku") or p.get("id") or f"{category}_{idx}",
                "product_id": product_id,
                "detail_url": detail_url,
                "title": p.get("title", ""),
                "price": p.get("price"),
                "rating": p.get("rating"),
                "image_url": p.get("image_url", ""),
                "image_count": _count_product_images(p),
                "review_count": _count_product_reviews(p),
                "description": (p.get("description") or "")[:240],
            })

        total_pages = max(1, math.ceil(total / page_size)) if total else 1
        return {
            "category": category,
            "query": q,
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
            "products": rows,
        }

    @app.post("/api/admin/delete")
    async def admin_delete_products(request: Request):
        _require_viewer_token_http(request)
        try:
            body = await request.json()
        except Exception:
            body = {}

        category = (body.get("category") or "").strip()
        indices = body.get("indices") or []
        if not category:
            raise HTTPException(status_code=400, detail="category 不能为空")
        if not isinstance(indices, list) or not indices:
            raise HTTPException(status_code=400, detail="indices 不能为空")

        normalized: list[int] = []
        for i in indices:
            try:
                normalized.append(int(i))
            except Exception:
                raise HTTPException(status_code=400, detail=f"非法 index: {i}")
        normalized = sorted(set(normalized), reverse=True)

        products = _load_category_products(category)
        before = len(products)
        removed = []
        for idx in normalized:
            if 0 <= idx < len(products):
                p = products.pop(idx)
                removed.append({
                    "index": idx,
                    "sku": p.get("sku") or p.get("id") or f"{category}_{idx}",
                    "title": p.get("title", ""),
                })

        _save_category_products(category, products)
        _reload_marketplace_products()
        after = len(products)
        return {
            "ok": True,
            "category": category,
            "before": before,
            "after": after,
            "deleted_count": before - after,
            "deleted": list(reversed(removed)),
            "note": "若当前使用 LlamaIndex 向量索引，建议重启服务并 rebuild-index 以彻底同步。",
        }
    
    # JSON API: 商品搜索（供 verbal 模式等程序化调用）
    @app.get("/api/search")
    async def api_search(
        q: str = "",
        condition_name: str = None,
        page: int = 1,
        page_size: int = 8,
        price_min: float | None = None,
        price_max: float | None = None,
        rating_min: float | None = None,
    ):
        """返回与 /search 完全相同的 top-k 商品，JSON 格式"""
        if not q:
            return {"query": q, "products": [], "total": 0, "page": 1, "total_pages": 1}
        try:
            page = max(page, 1)
            page_size = max(1, min(page_size, 40))
            price_min, price_max, rating_min = _normalize_filter_bounds(price_min, price_max, rating_min)
            filter_active = any(v is not None for v in (price_min, price_max, rating_min))
            all_products_list = list(getattr(marketplace, "products", []))
            fetch_limit_base = page_size * max(max_pages, page)
            fetch_limit = fetch_limit_base
            if filter_active:
                if all_products_list:
                    fetch_limit = max(fetch_limit_base, len(all_products_list))
                else:
                    fetch_limit = fetch_limit_base * 5
            results = marketplace.search_products(q, limit=fetch_limit)
            products = list(results.products)
            if condition_set and condition_name:
                cond = next((c for c in condition_set.conditions if c.name == condition_name), None)
                if cond:
                    products = cond.apply(products)

            target_count = min(page_size * max_pages, len(all_products_list) if all_products_list else page_size)
            if filter_active and all_products_list:
                target_count = len(all_products_list)
            if len(products) < target_count and all_products_list:
                products = _expand_products_for_pagination(
                    products=products,
                    query=q,
                    all_products=all_products_list,
                    target_count=target_count,
                )

            products = _apply_price_rating_filter(
                products,
                price_min=price_min,
                price_max=price_max,
                rating_min=rating_min,
            )

            max_items = page_size * max_pages
            products = products[:max_items]
            total_items = len(products)
            total_pages = max(1, min(math.ceil(total_items / page_size), max_pages)) if total_items else 1
            page = min(page, total_pages)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paged_products = products[start_idx:end_idx]

            return {
                "query": q,
                "products": [_product_to_dict(p) for p in paged_products],
                "total": total_items,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
            }
        except Exception as e:
            logger.error(f"API 搜索失败: {e}")
            return {"query": q, "products": [], "total": 0, "page": 1, "total_pages": 1, "error": str(e)}

    @app.get("/api/product/{product_id}")
    async def api_product_detail(product_id: str, condition_name: str = None):
        """返回单个商品完整详情，JSON 格式"""
        try:
            p = marketplace.get_product_details(product_id)
            if condition_set and condition_name:
                cond = next((c for c in condition_set.conditions if c.name == condition_name), None)
                if cond:
                    p = cond.apply_single_product(p)
            return {"product": _product_to_dict(p, desc_max_len=99999)}
        except Exception as e:
            logger.warning(f"API 商品详情获取失败 {product_id}: {e}")
            return {"product": None, "error": str(e)}

    # 后台 Agent 进程（用于可交互启动）
    agent_process = {"proc": None, "last_exit_code": None}

    @app.post("/api/run-agent")
    async def run_agent_api(request: Request):
        """从 Viewer 网页触发展开 Agent。请求体: {query: str, perception: "visual"|"verbal"}"""
        import subprocess
        _require_viewer_token_http(request)
        try:
            body = await request.json()
        except Exception:
            body = {}
        query = (body.get("query") or "").strip()
        perception = (body.get("perception") or "visual").lower()
        if perception not in ("visual", "verbal"):
            perception = "visual"
        if not query:
            return {"ok": False, "error": "query 不能为空"}
        if agent_process["proc"] is not None and agent_process["proc"].poll() is None:
            return {"ok": False, "error": "Agent 正在运行中，请稍后再试"}
        server_url = f"http://127.0.0.1:{port}"
        script = Path(__file__).parent / "run_browser_agent.py"
        # Keep viewer-run defaults aligned with the repeated distribution experiment pipeline.
        viewer_llm = (os.getenv("ACES_VIEWER_LLM") or "qwen").strip() or "qwen"
        try:
            viewer_max_steps = int(os.getenv("ACES_VIEWER_MAX_STEPS", "45"))
        except Exception:
            viewer_max_steps = 45
        verbal_use_vlm_default = (os.getenv("ACES_VIEWER_VERBAL_USE_VLM", "1") or "").strip().lower()
        use_verbal_vlm = verbal_use_vlm_default not in {"0", "false", "no", "off"}
        cmd = [
            sys.executable, str(script),
            "--llm", viewer_llm,
            "--query", query,
            "--perception", perception,
            "--server", server_url,
            "--max-steps", str(max(1, viewer_max_steps)),
            "--once",
        ]
        if perception == "verbal" and use_verbal_vlm:
            cmd.append("--verbal-use-vlm")
        try:
            env = os.environ.copy()
            if viewer_token:
                env["ACES_VIEWER_TOKEN"] = viewer_token
            proc = subprocess.Popen(
                cmd,
                cwd=str(Path(__file__).parent),
                # Inherit parent stdio to avoid PIPE backpressure deadlocks.
                env=env,
                start_new_session=True,
            )
            agent_process["proc"] = proc
            agent_process["last_exit_code"] = None
            logger.info(
                "已启动 Agent: query=\"%s...\", perception=%s, llm=%s, max_steps=%s, verbal_use_vlm=%s",
                query[:50],
                perception,
                viewer_llm,
                max(1, viewer_max_steps),
                bool(perception == "verbal" and use_verbal_vlm),
            )
            return {"ok": True, "message": f"Agent 已启动 ({perception} 模式)"}
        except Exception as e:
            logger.exception("启动 Agent 失败")
            return {"ok": False, "error": str(e)}

    @app.get("/api/agent-status")
    async def agent_status(request: Request):
        """查询 Agent 是否在运行"""
        _require_viewer_token_http(request)
        proc = agent_process.get("proc")
        if proc is None:
            return {
                "running": False,
                "pid": None,
                "exit_code": agent_process.get("last_exit_code"),
            }
        exit_code = proc.poll()
        running = exit_code is None
        if running:
            return {"running": True, "pid": proc.pid, "exit_code": None}
        # Process finished; clear handle but keep exit code for UI.
        agent_process["last_exit_code"] = int(exit_code)
        agent_process["proc"] = None
        return {"running": False, "pid": proc.pid, "exit_code": int(exit_code)}

    @app.post("/api/stop-agent")
    async def stop_agent_api(request: Request):
        """终止当前运行中的 Agent 子进程。"""
        import signal
        _require_viewer_token_http(request)

        proc = agent_process.get("proc")
        if proc is None or proc.poll() is not None:
            return {"ok": True, "message": "当前没有运行中的 Agent"}

        try:
            # Prefer killing the whole process group to avoid orphan children.
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass

        try:
            proc.wait(timeout=3)
            agent_process["proc"] = None
            agent_process["last_exit_code"] = int(proc.returncode) if proc.returncode is not None else None
            return {"ok": True, "message": "Agent 已终止"}
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            agent_process["proc"] = None
            agent_process["last_exit_code"] = int(proc.returncode) if proc.returncode is not None else None
            return {"ok": True, "message": "Agent 已强制终止"}

    # API 端点：推送截图和日志
    @app.post("/api/push")
    async def push_to_viewers(request: Request, data: dict):
        """Agent 推送数据到所有 viewer"""
        _require_viewer_token_http(request)
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
            "categories": len(product_dataset_files),
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
    for json_file in product_dataset_files[:10]:  # 只显示前10个
        print(f"  - {json_file.stem}")
    
    viewer_suffix = f"?token={viewer_token}" if viewer_token else ""

    print(f"\n页面:")
    print(f"  - 商品搜索: http://localhost:{port}/search?q=mousepad")
    print(f"  - 实时查看器: http://localhost:{port}/viewer{viewer_suffix} (在 MacBook 打开此页面)")
    print(f"  - 健康检查: http://localhost:{port}/health")
    
    if local_ip and local_ip != "127.0.0.1":
        print(f"\n从 MacBook 访问:")
        print(f"  http://{local_ip}:{port}/viewer{viewer_suffix}  ← 实时查看 Agent 操作")
    
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
        default="datasets/current",
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
    parser.add_argument(
        "--condition-file", "-C",
        type=str,
        default=None,
        help="实验条件 YAML/JSON 文件路径（如 configs/experiments/example_price_anchoring.yaml）"
    )
    parser.add_argument(
        "--max-pages", "-n",
        type=int,
        default=DEFAULT_WEB_MAX_PAGES,
        help=f"每次最多展示页数（默认 {DEFAULT_WEB_MAX_PAGES}）"
    )
    parser.add_argument(
        "--index-cache-dir",
        type=str,
        default=".cache/llamaindex",
        help="LlamaIndex 向量索引缓存目录（默认: .cache/llamaindex）"
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="强制重建向量索引并覆盖缓存"
    )
    
    args = parser.parse_args()
    
    # 确定是否使用 LlamaIndex
    use_llamaindex = not args.simple_search
    
    start_server(
        datasets_dir=args.datasets_dir,
        port=args.port,
        host=args.host,
        use_llamaindex=use_llamaindex,
        condition_file=args.condition_file,
        max_pages=args.max_pages,
        index_cache_dir=args.index_cache_dir,
        rebuild_index=args.rebuild_index,
    )


if __name__ == "__main__":
    main()
