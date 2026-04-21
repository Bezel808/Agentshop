#!/usr/bin/env python3
"""
ACES-v2 数据集管理脚本（合并版）

统一入口，支持：扩充（按类目/按关键词）、下载 ACE 数据集、补齐多图与评论。

数据来源:
  1. Amazon Reviews 2023 (McAuley Lab, HuggingFace)
     - meta_categories/meta_{Category}.jsonl
     - review_categories/{Category}.jsonl
  2. My-Custom-AI 系列 (ACE-RS, ACE-SR, ACE-BB) - 实验记录格式，需转换

用法:
  python manage_datasets.py expand-from-amazon [options]
  python manage_datasets.py expand-by-keyword [options]
  python manage_datasets.py download-ace [options]
  python manage_datasets.py enrich [options]
  python manage_datasets.py filter-reviews [options]  # 只保留有评论的商品
  python manage_datasets.py list-sources
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

HF_AMAZON_BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"

AMAZON_CATEGORIES = [
    "All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing",
    "Automotive", "Baby_Products", "Beauty_and_Personal_Care",
    "Books", "CDs_and_Vinyl", "Cell_Phones_and_Accessories",
    "Clothing_Shoes_and_Jewelry", "Digital_Music", "Electronics",
    "Gift_Cards", "Grocery_and_Gourmet_Food", "Handmade_Products",
    "Health_and_Household", "Home_and_Kitchen", "Industrial_and_Scientific",
    "Kindle_Store", "Magazine_Subscriptions", "Movies_and_TV",
    "Musical_Instruments", "Office_Products", "Patio_Lawn_and_Garden",
    "Pet_Supplies", "Software", "Sports_and_Outdoors",
    "Subscription_Boxes", "Tools_and_Home_Improvement", "Toys_and_Games",
    "Video_Games", "Unknown",
]

RECOMMENDED_CATEGORIES = [
    "Electronics", "Office_Products", "Home_and_Kitchen", "Sports_and_Outdoors",
    "Health_and_Household", "Beauty_and_Personal_Care", "Toys_and_Games",
    "Tools_and_Home_Improvement", "Pet_Supplies", "Cell_Phones_and_Accessories",
]

# 文件名 -> Amazon 类目（或类目列表，按顺序尝试）
FILE_TO_AMAZON_CATEGORY: Dict[str, str | List[str]] = {
    "beauty_and_personal_care": "Beauty_and_Personal_Care",
    "cell_phones_and_accessories": "Cell_Phones_and_Accessories",
    "electronics": "Electronics",
    "health_and_household": "Health_and_Household",
    "home_and_kitchen": "Home_and_Kitchen",
    "office_products": "Office_Products",
    "pet_supplies": "Pet_Supplies",
    "sports_and_outdoors": "Sports_and_Outdoors",
    "tools_and_home_improvement": "Tools_and_Home_Improvement",
    "toys_and_games": "Toys_and_Games",
    "mousepad": ["Office_Products", "Electronics"],
    "ski_jacket": ["Sports_and_Outdoors", "Clothing_Shoes_and_Jewelry"],
    # 新增品类 (按关键词扩充)
    "smart_watch": ["Electronics", "Cell_Phones_and_Accessories"],
    "athletic_shoes_men": ["Sports_and_Outdoors", "Clothing_Shoes_and_Jewelry"],
    "athletic_shoes_women": ["Sports_and_Outdoors", "Clothing_Shoes_and_Jewelry"],
    "bluetooth_speaker": ["Electronics"],
    "insulated_cup_men": ["Home_and_Kitchen", "Sports_and_Outdoors"],
    "insulated_cup_women": ["Home_and_Kitchen", "Sports_and_Outdoors"],
    "backpack_men": ["Office_Products", "Sports_and_Outdoors", "Clothing_Shoes_and_Jewelry"],
    "backpack_women": ["Office_Products", "Sports_and_Outdoors", "Clothing_Shoes_and_Jewelry"],
}

KEYWORD_CONFIG: Dict[str, Tuple[List[str], str]] = {
    "mousepad": (["Office_Products", "Electronics"], "mousepad.json"),
    "ski": (["Sports_and_Outdoors", "Clothing_Shoes_and_Jewelry"], "ski_jacket.json"),
    # 1. 智能手表 (享乐+功能双重价值)
    "smart_watch": (["Electronics", "Cell_Phones_and_Accessories"], "smart_watch.json"),
    # 2. 运动鞋 (分男女款)
    "athletic_shoes_men": (["Sports_and_Outdoors", "Clothing_Shoes_and_Jewelry"], "athletic_shoes_men.json"),
    "athletic_shoes_women": (["Sports_and_Outdoors", "Clothing_Shoes_and_Jewelry"], "athletic_shoes_women.json"),
    # 3. 蓝牙音箱 (设计+音质)
    "bluetooth_speaker": (["Electronics"], "bluetooth_speaker.json"),
    # 4. 保温杯 (分男女款)
    "insulated_cup_men": (["Home_and_Kitchen", "Sports_and_Outdoors"], "insulated_cup_men.json"),
    "insulated_cup_women": (["Home_and_Kitchen", "Sports_and_Outdoors"], "insulated_cup_women.json"),
    # 5. 双肩包 (分男女款)
    "backpack_men": (["Office_Products", "Sports_and_Outdoors", "Clothing_Shoes_and_Jewelry"], "backpack_men.json"),
    "backpack_women": (["Office_Products", "Sports_and_Outdoors", "Clothing_Shoes_and_Jewelry"], "backpack_women.json"),
}

# 关键词 -> 标题匹配词列表 (用于 expand-by-keyword，若未定义则使用 [keyword])
SEARCH_TERMS_CONFIG: Dict[str, List[str]] = {
    "ski": ["ski", "snow jacket", "snowsuit", "snow suit"],
    "smart_watch": [
        "smart watch",
        "smartwatch",
        "fitness watch",
        "gps smartwatch",
        "running smartwatch",
        "galaxy watch",
        "apple watch",
        "forerunner",
        "fitbit watch",
        "amazfit watch",
    ],
    "athletic_shoes_men": ["men's running shoe", "men running shoe", "men's sneaker", "men sneaker", "mens athletic shoe", "men's athletic shoe"],
    "athletic_shoes_women": ["women's running shoe", "women running shoe", "women's sneaker", "women sneaker", "womens athletic shoe", "women's athletic shoe"],
    "bluetooth_speaker": ["bluetooth speaker", "portable bluetooth speaker", "wireless speaker"],
    "insulated_cup_men": ["tumbler", "water bottle", "thermos", "insulated bottle", "stainless steel bottle", "travel mug", "vacuum flask", "hydro flask", "yeti", "men's tumbler", "men tumbler"],
    "insulated_cup_women": ["tumbler", "water bottle", "thermos", "insulated bottle", "stainless steel bottle", "travel mug", "vacuum flask", "hydro flask", "yeti", "women's tumbler", "women tumbler"],
    "backpack_men": ["backpack", "rucksack", "laptop backpack", "daypack", "hiking backpack", "travel backpack", "men's backpack", "men backpack", "mens backpack"],
    "backpack_women": ["backpack", "rucksack", "laptop backpack", "daypack", "women's backpack", "women backpack", "womens backpack", "travel backpack"],
}

SMART_WATCH_ACCESSORY_PATTERNS: List[str] = [
    r"\bband(s)?\b",
    r"\bstrap(s)?\b",
    r"\bwrist\s?band(s)?\b",
    r"\bwatch\s?band(s)?\b",
    r"\breplacement\b",
    r"\bcase(s)?\b",
    r"\bcover(s)?\b",
    r"\bprotector(s)?\b",
    r"\bscreen\s+protector\b",
    r"\bguard\b",
    r"\bfilm\b",
    r"\bcharger(s)?\b",
    r"\bcharging\s+(cable|dock|stand)\b",
    r"\bcharging\s+(station|pad)\b",
    r"\bdock\b",
    r"\bstand\b",
    r"\bstation\b",
    r"\bwireless\s+charger\b",
    r"\bcharging\s+base\b",
    r"\bcharging\s+hub\b",
    r"\bholder\b",
    r"\badapter\b",
    r"\bbumper\b",
    r"\bshell\b",
    r"\bbezel\b",
    r"\bframe\b",
]


def _looks_like_smart_watch_accessory(title: str) -> bool:
    t = str(title or "").lower()
    return any(re.search(p, t) for p in SMART_WATCH_ACCESSORY_PATTERNS)


# ── 通用工具 ──────────────────────────────────────────────────────────────

def _parse_price(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw) if raw > 0 else None
    s = re.sub(r"[^\d.]", "", str(raw).strip())
    if not s:
        return None
    try:
        v = float(s)
        return v if v > 0 else None
    except ValueError:
        return None


def _pick_image_url(images: Any) -> Optional[str]:
    if not images:
        return None
    if isinstance(images, list):
        for entry in images:
            if isinstance(entry, dict):
                for key in ("large", "hi_res", "thumb"):
                    url = entry.get(key)
                    if url and isinstance(url, str) and url.startswith("http"):
                        return url
        return None
    if isinstance(images, dict):
        for key in ("large", "hi_res", "thumb"):
            urls = images.get(key)
            if isinstance(urls, list):
                for url in urls:
                    if url and isinstance(url, str) and url.startswith("http"):
                        return url
    return None


def _join_text(lst: Any) -> str:
    if isinstance(lst, list):
        return " ".join(str(x).strip() for x in lst if x)
    if isinstance(lst, str):
        return lst.strip()
    return ""


def _sanitize_category_name(category: str) -> str:
    return category.lower().replace(" ", "_").replace("&", "and")


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").strip().lower())


def _split_csv_terms(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    return [x.strip() for x in str(raw).split(",") if x and x.strip()]


def _extract_image_urls_from_item(images: Any) -> List[str]:
    """从 Amazon images 结构提取所有图片 URL（用于 expand 阶段）。"""
    urls: List[str] = []
    def _add(u: Any):
        if isinstance(u, str) and u.startswith("http") and u not in urls:
            urls.append(u)
    if isinstance(images, dict):
        for k in ("large", "hi_res", "thumb"):
            v = images.get(k)
            if isinstance(v, list):
                for x in v:
                    _add(x)
    elif isinstance(images, list):
        for item in images:
            if isinstance(item, str):
                _add(item)
            elif isinstance(item, dict):
                for k in ("large", "hi_res", "thumb", "url"):
                    _add(item.get(k))
    return urls


def convert_amazon_item(item: Dict[str, Any], category: str, index: int) -> Optional[Dict[str, Any]]:
    """将 Amazon 元数据转为 ACES 格式。含 parent_asin、image_urls 便于 enrich 仅拉评论。"""
    title = (item.get("title") or "").strip()
    if not title or len(title) < 5:
        return None
    price = _parse_price(item.get("price"))
    if price is None:
        return None
    image_url = _pick_image_url(item.get("images"))
    if not image_url:
        return None
    imgs = _extract_image_urls_from_item(item.get("images"))
    if not imgs and image_url:
        imgs = [image_url]
    elif image_url and imgs and imgs[0] != image_url:
        imgs = [image_url] + [u for u in imgs if u != image_url]
    avg_rating = item.get("average_rating")
    if avg_rating is not None:
        try:
            avg_rating = round(float(avg_rating), 1)
        except (TypeError, ValueError):
            avg_rating = None
    rating_number = item.get("rating_number")
    if rating_number is not None:
        try:
            rating_number = int(rating_number)
        except (TypeError, ValueError):
            rating_number = None
    cat_slug = _sanitize_category_name(category)
    sku = f"{cat_slug}_{index}"
    description = _join_text(item.get("description", []))
    features = _join_text(item.get("features", []))
    if features and not description:
        description = features
    elif features and description:
        description = description + " " + features
    description = description[:2000]
    out = {
        "sku": sku, "title": title, "price": price, "rating": avg_rating, "rating_count": rating_number,
        "image_url": image_url, "description": description,
        "sponsored": False, "best_seller": False, "overall_pick": False, "low_stock": False,
    }
    if imgs:
        out["image_urls"] = imgs
    if item.get("parent_asin"):
        out["parent_asin"] = item["parent_asin"]
    return out


# ── 图片检查 ───────────────────────────────────────────────────────────────

async def _check_image_urls(products: List[Dict], concurrency: int = 20, timeout: float = 8.0) -> List[Dict]:
    import aiohttp
    sem = asyncio.Semaphore(concurrency)
    valid: List[Dict] = []
    total = len(products)
    checked = 0

    async def _check(product: Dict):
        nonlocal checked
        url = product.get("image_url", "")
        try:
            async with sem:
                async with aiohttp.ClientSession() as session:
                    async with session.head(url, timeout=aiohttp.ClientTimeout(total=timeout), allow_redirects=True) as resp:
                        checked += 1
                        if resp.status < 400:
                            valid.append(product)
                        if checked % 50 == 0:
                            logger.info(f"  图片检查: {checked}/{total} (有效: {len(valid)})")
        except Exception:
            checked += 1

    await asyncio.gather(*[_check(p) for p in products])
    return valid


# ── 流式读取 HuggingFace ───────────────────────────────────────────────────

def _stream_jsonl_from_hf(category: str, max_lines: int) -> List[Dict]:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    url = f"{HF_AMAZON_BASE}/meta_categories/meta_{category}.jsonl"
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504])))
    items: List[Dict] = []
    try:
        with session.get(url, stream=True, timeout=(15, 120)) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                if len(items) >= max_lines:
                    break
                try:
                    items.append(json.loads(raw_line))
                except json.JSONDecodeError:
                    continue
                if len(items) % 1000 == 0 and items:
                    logger.info(f"    已读取 {len(items)} 行...")
    except Exception as e:
        logger.error(f"  下载失败: {e}")
    return items


def _iter_jsonl_from_url(url: str, scan_limit: Optional[int]) -> Iterable[Dict]:
    import requests
    with requests.get(url, stream=True, timeout=(60, 300)) as resp:
        resp.raise_for_status()
        cnt = 0
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line.decode("utf-8"))
            except Exception:
                continue
            yield obj
            cnt += 1
            if scan_limit and cnt >= scan_limit:
                return


def _load_review_asins(categories: List[str], scan_limit: int = 100000) -> set:
    """从 review 文件预加载有评论的 parent_asin 集合，供 expand 时过滤"""
    asins: set = set()
    for cat in categories:
        url = f"{HF_AMAZON_BASE}/review_categories/{cat}.jsonl"
        logger.info(f"  预加载 review ASIN: {cat} (最多 {scan_limit} 行)")
        try:
            for item in _iter_jsonl_from_url(url, scan_limit):
                a = item.get("parent_asin")
                if a and isinstance(a, str) and a.strip():
                    asins.add(a.strip())
        except Exception as e:
            logger.warning(f"  读取 {cat} review 失败: {e}")
    logger.info(f"  共 {len(asins)} 个 ASIN 有评论")
    return asins


# ── 子命令: expand-from-amazon ────────────────────────────────────────────

def cmd_expand_from_amazon(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    categories = [c.strip() for c in args.categories.split(",")] if args.categories else RECOMMENDED_CATEGORIES
    total = 0
    for i, cat in enumerate(categories, 1):
        logger.info(f"[{i}/{len(categories)}] {cat}")
        raw_items = _stream_jsonl_from_hf(cat, max_lines=args.scan_limit)
        candidates = []
        for idx, item in enumerate(raw_items):
            p = convert_amazon_item(item, cat, idx)
            if p is None:
                continue
            if (p.get("rating_count") or 0) < args.min_reviews:
                continue
            candidates.append(p)
        candidates.sort(key=lambda x: (x.get("rating_count") or 0), reverse=True)
        shortlist = candidates[: args.per_category * 3]
        if not args.skip_image_check and shortlist:
            shortlist = asyncio.run(_check_image_urls(shortlist))
        final = shortlist[: args.per_category]
        for j, p in enumerate(final):
            p["sku"] = f"{_sanitize_category_name(cat)}_{j}"
        if final:
            out_file = output_dir / f"{_sanitize_category_name(cat)}.json"
            out_file.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
            total += len(final)
            logger.info(f"  => {out_file.name}: {len(final)}")
    print(f"\n完成! 共 {total} 个商品 -> {output_dir}/")


# ── 子命令: expand-by-keyword ─────────────────────────────────────────────

def _expand_single_keyword(
    *,
    kw: str,
    categories: List[str],
    out_filename: str,
    search_terms: List[str],
    output_dir: Path,
    max_products: int,
    scan_limit: int,
    min_reviews: int,
    skip_image_check: bool,
    ensure_reviews: bool,
    review_preload_limit: int,
    include_terms: Optional[List[str]] = None,
    exclude_terms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    review_asins: set = set()
    if ensure_reviews:
        review_asins = _load_review_asins(list(categories), scan_limit=review_preload_limit)

    include_lc = [t.lower() for t in (include_terms or []) if t]
    exclude_lc = [t.lower() for t in (exclude_terms or []) if t]
    search_lc = [t.lower() for t in (search_terms or []) if t]

    all_candidates: List[Dict[str, Any]] = []
    seen = set()
    dropped_accessory = 0
    dropped_include = 0
    dropped_exclude = 0

    for cat in categories:
        raw = _stream_jsonl_from_hf(cat, max_lines=scan_limit)
        for idx, item in enumerate(raw):
            title = (item.get("title") or "").strip()
            title_l = title.lower()
            if search_lc and not any(t in title_l for t in search_lc):
                continue
            if include_lc and not any(t in title_l for t in include_lc):
                dropped_include += 1
                continue
            if exclude_lc and any(t in title_l for t in exclude_lc):
                dropped_exclude += 1
                continue
            if kw == "smart_watch" and _looks_like_smart_watch_accessory(title):
                dropped_accessory += 1
                continue

            norm = re.sub(r"\s+", " ", title_l)
            if norm in seen:
                continue
            p = convert_amazon_item(item, cat, idx)
            if p is None or (p.get("rating_count") or 0) < min_reviews:
                continue
            if review_asins:
                asin = p.get("parent_asin") or item.get("parent_asin")
                if not asin or str(asin).strip() not in review_asins:
                    continue
            seen.add(norm)
            all_candidates.append(p)

    all_candidates.sort(key=lambda x: (x.get("rating_count") or 0), reverse=True)
    shortlist = all_candidates[: max_products * 2]
    if not skip_image_check and shortlist:
        shortlist = asyncio.run(_check_image_urls(shortlist))
    final = shortlist[: max_products]

    slug = kw.lower().replace(" ", "_")
    for i, p in enumerate(final):
        p["sku"] = f"{slug}_{i}"

    out_file = output_dir / out_filename
    if final:
        out_file.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"  => {out_file.name}: {len(final)}")
    return {
        "keyword": kw,
        "out_file": str(out_file),
        "count": len(final),
        "dropped": {
            "smart_watch_accessory": dropped_accessory,
            "custom_include": dropped_include,
            "custom_exclude": dropped_exclude,
        },
    }


def cmd_expand_by_keyword(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keywords = _split_csv_terms(args.keywords)
    total = 0
    for kw in keywords:
        if kw not in KEYWORD_CONFIG:
            logger.warning(f"未知关键词 '{kw}'，跳过")
            continue
        categories, out_filename = KEYWORD_CONFIG[kw]
        search_terms = SEARCH_TERMS_CONFIG.get(kw, [kw])
        result = _expand_single_keyword(
            kw=kw,
            categories=list(categories),
            out_filename=out_filename,
            search_terms=search_terms,
            output_dir=output_dir,
            max_products=args.max_products,
            scan_limit=args.scan_limit,
            min_reviews=args.min_reviews,
            skip_image_check=args.skip_image_check,
            ensure_reviews=getattr(args, "ensure_reviews", False),
            review_preload_limit=getattr(args, "review_preload_limit", 80000),
        )
        total += int(result.get("count", 0))
        dropped = result.get("dropped", {})
        if kw == "smart_watch" and dropped.get("smart_watch_accessory", 0):
            logger.info(
                f"    smart_watch accessory dropped during expand: {dropped.get('smart_watch_accessory', 0)}"
            )
    print(f"\n完成! 共 {total} 个商品 -> {output_dir}/")


def cmd_expand_custom_keyword(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kw = (args.name or "").strip()
    if not kw:
        raise ValueError("--name 不能为空")

    categories = _split_csv_terms(args.categories)
    if not categories:
        raise ValueError("--categories 不能为空，示例: Home_and_Kitchen,Tools_and_Home_Improvement")

    search_terms = _split_csv_terms(args.search_terms) or [kw]
    out_filename = (args.out_file or f"{kw}.json").strip()
    if not out_filename.lower().endswith(".json"):
        out_filename += ".json"

    include_terms = _split_csv_terms(args.include_terms)
    exclude_terms = _split_csv_terms(args.exclude_terms)

    result = _expand_single_keyword(
        kw=kw,
        categories=categories,
        out_filename=out_filename,
        search_terms=search_terms,
        output_dir=output_dir,
        max_products=args.max_products,
        scan_limit=args.scan_limit,
        min_reviews=args.min_reviews,
        skip_image_check=args.skip_image_check,
        ensure_reviews=getattr(args, "ensure_reviews", False),
        review_preload_limit=getattr(args, "review_preload_limit", 80000),
        include_terms=include_terms,
        exclude_terms=exclude_terms,
    )

    dropped = result.get("dropped", {})
    print(
        "\n完成! "
        f"{kw} -> {result.get('out_file')} | count={result.get('count', 0)} "
        f"| dropped(include={dropped.get('custom_include', 0)}, "
        f"exclude={dropped.get('custom_exclude', 0)})"
    )


# ── 子命令: download-ace ──────────────────────────────────────────────────

def cmd_download_ace(args: argparse.Namespace) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("需安装: pip install datasets")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [
        {"name": "My-Custom-AI/ACE-RS", "subset": "absolute_and_random_price"},
        {"name": "My-Custom-AI/ACE-RS", "subset": "instruction_following"},
        {"name": "My-Custom-AI/ACE-RS", "subset": "rating"},
        {"name": "My-Custom-AI/ACE-RS", "subset": "relative_price"},
        {"name": "My-Custom-AI/ACE-SR", "subset": None},
        {"name": "My-Custom-AI/ACE-BB", "subset": "choice_behavior"},
    ]
    all_products: Dict[str, List[Dict]] = defaultdict(list)
    for ds in datasets:
        try:
            ds_load = load_dataset(ds["name"], ds["subset"], split="data") if ds["subset"] else load_dataset(ds["name"], split="data")
        except Exception as e:
            logger.warning(f"  跳过 {ds['name']}/{ds['subset']}: {e}")
            continue
        for row in ds_load:
            query = row.get("query", "unknown")
            titles = row.get("title", [])
            if not isinstance(titles, list):
                continue
            for i in range(len(titles)):
                p = {
                    "sku": row.get("sku", [f"{query}_{i}"])[i] if isinstance(row.get("sku"), list) else f"{query}_{i}",
                    "title": titles[i],
                    "price": float(str(row.get("price", [0])[i]).replace("$", "").replace(",", "")) if "price" in row else 0,
                    "rating": row.get("rating", [None])[i] if "rating" in row else None,
                    "rating_count": row.get("rating_count", [0])[i] if "rating_count" in row else 0,
                    "image_url": (row.get("image_url", [""])[i] if "image_url" in row else "") or "",
                    "description": (row.get("description", [""])[i] if "description" in row else "") or "",
                    "sponsored": False, "best_seller": False, "overall_pick": False, "low_stock": False,
                }
                all_products[query].append(p)
    for query, prods in all_products.items():
        seen: set[str] = set()
        unique = []
        for p in prods:
            if p["sku"] not in seen:
                seen.add(p["sku"])
                unique.append(p)
        out_file = output_dir / f"{query}.json"
        out_file.write_text(json.dumps(unique, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"  {out_file.name}: {len(unique)}")
    print(f"\n完成! 输出: {output_dir}/")


# ── 子命令: enrich ────────────────────────────────────────────────────────

def _extract_image_urls(images: Any) -> List[str]:
    urls: List[str] = []
    def _add(u: Any):
        if isinstance(u, str) and u.startswith("http") and u not in urls:
            urls.append(u)
    if isinstance(images, dict):
        for k in ("large", "hi_res", "thumb"):
            v = images.get(k)
            if isinstance(v, list):
                for x in v:
                    _add(x)
    elif isinstance(images, list):
        for item in images:
            if isinstance(item, str):
                _add(item)
            elif isinstance(item, dict):
                for k in ("large", "hi_res", "thumb", "url"):
                    _add(item.get(k))
    return urls


def _extract_description(item: Dict) -> str:
    d = _join_text(item.get("description", []))
    f = _join_text(item.get("features", []))
    if f and not d:
        return f
    if f and d:
        return d + " " + f
    return d


def _choose_best_idx(indices: List[int], products: List[Dict], meta_price: Optional[float], used: set) -> Optional[int]:
    cand = [i for i in indices if i not in used]
    if not cand:
        return None
    if meta_price is None:
        return cand[0]
    return min(cand, key=lambda i: abs(float(products[i].get("price", 0)) - meta_price))


def _stable_bucket(text: str, buckets: int = 2) -> int:
    digest = hashlib.md5((text or "").encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(buckets, 1)


def cmd_enrich(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = list(input_dir.glob("*.json"))
    if args.files:
        requested = {x.strip() for x in args.files.split(",")}
        files = [f for f in files if f.stem in requested]
    total_img, total_rev, total_dropped = 0, 0, 0
    report: Dict[str, Any] = {"input_dir": str(input_dir), "output_dir": str(output_dir), "files": {}}
    for f in files:
        stem = f.stem
        if stem not in FILE_TO_AMAZON_CATEGORY:
            (output_dir / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
            continue
        cats = [FILE_TO_AMAZON_CATEGORY[stem]] if isinstance(FILE_TO_AMAZON_CATEGORY[stem], str) else list(FILE_TO_AMAZON_CATEGORY[stem])
        products = json.loads(f.read_text(encoding="utf-8"))
        for p in products:
            p.setdefault("image_urls", p.get("image_urls") or [])
            p.setdefault("reviews", [])

        idx_to_asin: Dict[int, str] = {}
        needs_meta: List[int] = []
        for i, p in enumerate(products):
            asin = p.get("parent_asin")
            if asin:
                idx_to_asin[i] = asin
                p.setdefault("enrich_provenance", {})["asin_source"] = "existing"
            else:
                needs_meta.append(i)
            imgs = p.get("image_urls") or []
            if imgs and not p.get("image_url"):
                p["image_url"] = imgs[0]
            if imgs and not p.get("description_images"):
                p["description_images"] = list(imgs)

        if needs_meta:
            title_to_idx: Dict[str, List[int]] = {}
            for i in needs_meta:
                key = _normalize_title(products[i].get("title", ""))
                if key:
                    title_to_idx.setdefault(key, []).append(i)
            used = set()
            for amazon_cat in cats:
                if not title_to_idx or len(used) >= len(needs_meta):
                    break
                meta_url = f"{HF_AMAZON_BASE}/meta_categories/meta_{amazon_cat}.jsonl"
                for item in _iter_jsonl_from_url(meta_url, args.meta_scan_limit or None):
                    key = _normalize_title(item.get("title", ""))
                    if key not in title_to_idx:
                        continue
                    chosen = _choose_best_idx(title_to_idx[key], products, _parse_price(item.get("price")), used)
                    if chosen is None:
                        continue
                    used.add(chosen)
                    imgs = _extract_image_urls(item.get("images"))
                    if products[chosen].get("image_url"):
                        main = products[chosen]["image_url"]
                        imgs = [main] + [u for u in imgs if u != main]
                    if imgs:
                        products[chosen]["image_urls"] = imgs
                        products[chosen]["description_images"] = list(imgs)
                        if not products[chosen].get("image_url"):
                            products[chosen]["image_url"] = imgs[0]
                    asin = item.get("parent_asin")
                    if asin:
                        idx_to_asin[chosen] = asin
                        products[chosen]["parent_asin"] = asin
                        products[chosen].setdefault("enrich_provenance", {})["asin_source"] = "meta_title_price_match"
                    desc = _extract_description(item)
                    if desc:
                        cur = (products[chosen].get("description") or "").strip()
                        if not cur or len(desc) > len(cur):
                            products[chosen]["description"] = desc[:2000]
                    if len(used) >= len(needs_meta):
                        break

        for i, p in enumerate(products):
            imgs = p.get("image_urls") or []
            if imgs and not p.get("description_images"):
                p["description_images"] = list(imgs)
            p.setdefault("enrich_provenance", {}).setdefault(
                "quality_snapshot",
                {
                    "image_count": len(p.get("image_urls") or []),
                    "review_count": len(p.get("reviews") or []),
                },
            )

        asin_to_idx: Dict[str, List[int]] = defaultdict(list)
        for i, a in idx_to_asin.items():
            asin_to_idx[a].append(i)
        all_asins = set(asin_to_idx.keys())
        reviews_by_asin: Dict[str, List[Dict]] = {a: [] for a in all_asins}
        scan_limit = args.review_scan_limit
        if scan_limit <= 0:
            scan_limit = None
            logger.warning("  review_scan_limit=0 将流式读取完整文件（单文件约22GB），可能很慢")
        else:
            logger.info(f"  review 扫描上限: {scan_limit} 行/类目")
        for amazon_cat in cats:
            rv_url = f"{HF_AMAZON_BASE}/review_categories/{amazon_cat}.jsonl"
            for item in _iter_jsonl_from_url(rv_url, scan_limit or None):
                a = item.get("parent_asin")
                if a not in all_asins or len(reviews_by_asin[a]) >= args.max_reviews:
                    continue
                text = (item.get("text") or "").strip()
                title = (item.get("title") or "").strip()
                if not text and not title:
                    continue
                reviews_by_asin[a].append({
                    "title": title, "text": text, "rating": item.get("rating"),
                    "author": item.get("user_id", ""), "date": str(item.get("timestamp", "")),
                    "helpful_vote": item.get("helpful_vote", 0),
                })
            if all(len(reviews_by_asin[a]) >= args.max_reviews for a in all_asins):
                logger.info(f"  已收满 {args.max_reviews} 条/商品，提前结束")
                break

        for a, idxs in asin_to_idx.items():
            ranked = sorted(reviews_by_asin.get(a, []), key=lambda r: int(r.get("helpful_vote") or 0), reverse=True)[: args.max_reviews]
            for i in idxs:
                products[i]["reviews"] = ranked
                products[i].setdefault("enrich_provenance", {})["review_source"] = "amazon_reviews_2023"

        if getattr(args, "require_complete", False):
            orig_len = len(products)
            kept = []
            for p in products:
                has_img = bool((p.get("image_url") or "").strip())
                has_urls = isinstance(p.get("image_urls"), list) and len(p.get("image_urls", [])) > 0
                has_desc_imgs = isinstance(p.get("description_images"), list) and len(p.get("description_images", [])) > 0
                has_revs = isinstance(p.get("reviews"), list) and len(p.get("reviews", [])) > 0
                if has_img and (has_urls or has_desc_imgs) and has_revs:
                    kept.append(p)
                else:
                    total_dropped += 1
            products = kept
            logger.info(f"  {f.name}: 保留 {len(products)} 个（需含主图+详情图+评论），丢弃 {orig_len - len(products)}")

        # Additional explicit quality thresholds for v2 output.
        if getattr(args, "min_images", 0) > 0 or getattr(args, "min_reviews", 0) > 0:
            before = len(products)
            min_images = max(int(getattr(args, "min_images", 0)), 0)
            min_reviews = max(int(getattr(args, "min_reviews", 0)), 0)
            filtered = []
            for p in products:
                image_count = len(p.get("image_urls") or [])
                review_count = len(p.get("reviews") or [])
                if image_count >= min_images and review_count >= min_reviews:
                    filtered.append(p)
            products = filtered
            logger.info(f"  {f.name}: 门槛过滤(min_images={min_images}, min_reviews={min_reviews}) {before} -> {len(products)}")

        out_path = output_dir / f.name
        out_path.write_text(json.dumps(products, ensure_ascii=False, indent=2), encoding="utf-8")
        img_cov = sum(1 for p in products if isinstance(p.get("image_urls"), list) and len(p.get("image_urls", [])) > 0)
        rev_cov = sum(1 for p in products if isinstance(p.get("reviews"), list) and len(p.get("reviews", [])) > 0)
        total_img += img_cov
        total_rev += rev_cov
        logger.info(f"  {f.name}: image_covered={img_cov}/{len(products)} review_covered={rev_cov}/{len(products)}")
        report["files"][f.name] = {
            "items": len(products),
            "image_covered": img_cov,
            "review_covered": rev_cov,
            "dropped_by_require_complete": total_dropped,
        }
    report["summary"] = {
        "total_image_covered": total_img,
        "total_review_covered": total_rev,
        "total_dropped": total_dropped,
    }
    (output_dir / "dataset_quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n完成! image_covered={total_img} review_covered={total_rev} -> {output_dir}/")


# ── 子命令: filter-reviews ─────────────────────────────────────────────────

# ── 子命令: filter-quality ─────────────────────────────────────────────────

# 每个类目：必须包含任一 INCLUDE（产品类型词），且不能包含 EXCLUDE
FILTER_QUALITY_RULES: Dict[str, Dict[str, List[str]]] = {
    "athletic_shoes_men": {
        "include": ["shoe", "sneaker", "runner", "running shoe", "walking shoe", "tennis shoe"],
        "exclude": ["sweatpant", "short", "jean", "pant", "shirt", "jacket", "swimsuit", "legging",
                   "women's", "womens ", " women ", "for women"],
    },
    "athletic_shoes_women": {
        "include": ["shoe", "sneaker", "runner", "running shoe", "walking shoe", "tennis shoe"],
        "exclude": ["men's", "mens ", "for men", "short", "jean", "pant", "legging",
                   "medal hanger", "display rack"],
    },
    "backpack_men": {
        "include": ["backpack", "rucksack", "daypack", "laptop backpack", "messenger bag"],
        "exclude": ["umbrella", "water bottle", "hammock", "tent", "camping chair", "cooler",
                   "percolator", "coffee maker", "sleeping bag", "sleeping pad", "compression sack",
                   "hiking boots", "hiking boot", "trekking boot"],
    },
    "backpack_women": {
        "include": ["backpack", "rucksack", "daypack", "laptop backpack", "messenger bag"],
        "exclude": ["umbrella", "water bottle", "hammock", "tent", "camping chair", "cooler",
                   "percolator", "coffee maker", "sleeping bag", "sleeping pad", "compression sack",
                   "hiking boots", "hiking boot", "trekking boot"],
    },
    "bluetooth_speaker": {
        "include": ["speaker"],
        "exclude": ["earbud", "earbuds", "headphone", "headphones", "ear phone"],
    },
    "insulated_cup_men": {
        "include": ["tumbler", "water bottle", "thermos", "insulated", "travel mug", "flask", "cup"],
        "exclude": [],
    },
    "insulated_cup_women": {
        "include": ["tumbler", "water bottle", "thermos", "insulated", "travel mug", "flask", "cup"],
        "exclude": [],
    },
    "smart_watch": {
        "include": [
            "smart watch",
            "smartwatch",
            "fitness tracker",
            "fitness watch",
            "wearable",
            "running watch",
            "gps watch",
            "galaxy watch",
            "apple watch",
            "forerunner",
            "amazfit",
            "fitbit",
            "garmin",
            "wear os",
            "watch",
        ],
        "exclude": [
            "band",
            "strap",
            "wristband",
            "watch band",
            "case",
            "cover",
            "protector",
            "screen protector",
            "guard",
            "film",
            "charger",
            "charging cable",
            "charging dock",
            "charging station",
            "charging pad",
            "wireless charger",
            "charging base",
            "charging hub",
            "replacement",
            "adapter",
            "holder",
            "stand",
            "bumper",
        ],
    },
}

def _product_matches_category(stem: str, title: str) -> Tuple[bool, str]:
    """Return (matches, reason)."""
    if stem not in FILTER_QUALITY_RULES:
        return True, ""
    t = title.lower()
    if stem == "smart_watch" and _looks_like_smart_watch_accessory(t):
        return False, "exclude:smart_watch_accessory"
    rules = FILTER_QUALITY_RULES[stem]
    for ex in rules.get("exclude", []):
        if ex in t:
            # "men's" in "women's" 会误伤，需排除该情况
            if ex in ("men's", "mens "):
                if "women" in t or "womens" in t:
                    continue
            return False, f"exclude:{ex}"
    for inc in rules.get("include", []):
        if inc in t:
            return True, ""
    return False, "no_include_match"


def cmd_filter_quality(args: argparse.Namespace) -> None:
    """筛除分类不符、跨文件重复的商品。需先 filter-reviews 确保有评论。"""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = args.target
    files = list(input_dir.glob("*.json"))
    if args.files:
        requested = {x.strip() for x in args.files.split(",")}
        files = [f for f in files if f.stem in requested]
    files = [f for f in files if f.stem in FILE_TO_AMAZON_CATEGORY]

    # 1. 加载所有数据，建立 asin -> [(stem, idx, product)]
    all_data: Dict[str, List[Dict]] = {}
    asin_locs: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for f in files:
        stem = f.stem
        data = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        all_data[stem] = data
        for i, p in enumerate(data):
            a = p.get("parent_asin")
            if a and isinstance(a, str) and a.strip():
                asin_locs[a.strip()].append((stem, i))

    # 2. 跨文件重复：对 men/women 对，按标题分配 ASIN 到更匹配的文件
    men_women_pairs = [
        ("athletic_shoes_men", "athletic_shoes_women"),
        ("backpack_men", "backpack_women"),
        ("insulated_cup_men", "insulated_cup_women"),
    ]
    asin_to_keep_in: Dict[str, str] = {}
    for m, w in men_women_pairs:
        for asin, locs in asin_locs.items():
            stems = set(x[0] for x in locs)
            if stems != {m, w}:
                continue
            if asin in asin_to_keep_in:
                continue
            # 选更匹配的文件：标题含 women->女，含 men->男，否则按 asin 哈希平分
            t = ""
            for stem, idx in locs:
                if stem not in all_data:
                    continue
                p = all_data[stem][idx]
                t = (p.get("title") or "").lower()
                break
            if "women" in t or "womens" in t or "for her" in t:
                asin_to_keep_in[asin] = w
            elif "men" in t or "mens" in t or "for him" in t:
                asin_to_keep_in[asin] = m
            else:
                asin_to_keep_in[asin] = w if (_stable_bucket(asin, buckets=2) == 0) else m

    # 3. 逐文件过滤
    for stem, data in all_data.items():
        kept = []
        dropped_cat, dropped_dup = 0, 0
        seen_asin = set()
        for i, p in enumerate(data):
            if not isinstance(p.get("reviews"), list) or len(p.get("reviews", [])) == 0:
                dropped_cat += 1
                continue
            title = p.get("title") or ""
            ok, reason = _product_matches_category(stem, title)
            if not ok:
                dropped_cat += 1
                if args.verbose:
                    logger.info(f"  [{stem}] 筛除({reason}): {title[:60]}...")
                continue
            asin = (p.get("parent_asin") or "").strip()
            if asin:
                preferred = asin_to_keep_in.get(asin)
                if preferred and preferred != stem:
                    dropped_dup += 1
                    continue
                if asin in seen_asin:
                    dropped_dup += 1
                    continue
                seen_asin.add(asin)
            kept.append(p)
        kept = kept[:target]
        for i, p in enumerate(kept):
            p["sku"] = f"{stem}_{i}"
        out_path = output_dir / f"{stem}.json"
        out_path.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"  {stem}.json: {len(data)} -> {len(kept)} (筛除 分类不符:{dropped_cat} 重复:{dropped_dup})")
    print(f"\n完成! -> {output_dir}/")


def cmd_filter_reviews(args: argparse.Namespace) -> None:
    """只保留有评论的商品，并按目标数量裁剪。用于 enrich 后整理。"""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = args.target
    report: Dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "target": target,
        "min_images": int(args.min_images),
        "min_reviews": int(args.min_reviews),
        "files": {},
    }
    files = list(input_dir.glob("*.json"))
    if args.files:
        requested = {x.strip() for x in args.files.split(",")}
        files = [f for f in files if f.stem in requested]
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        filtered = []
        for p in data:
            reviews = p.get("reviews") if isinstance(p.get("reviews"), list) else []
            imgs = p.get("image_urls") if isinstance(p.get("image_urls"), list) else []
            if len(reviews) < int(args.min_reviews):
                continue
            if len(imgs) < int(args.min_images):
                continue
            filtered.append(p)
        filtered.sort(
            key=lambda x: (
                int(x.get("rating_count") or 0),
                float(x.get("rating") or 0.0),
                len(x.get("reviews") or []),
                len(x.get("image_urls") or []),
            ),
            reverse=True,
        )
        kept = filtered[:target] if len(filtered) >= target else filtered
        stem = f.stem
        for i, p in enumerate(kept):
            p["sku"] = f"{stem}_{i}"
        out_path = output_dir / f.name
        out_path.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(
            f"  {f.name}: {len(data)} -> {len(kept)} "
            f"(min_images={args.min_images}, min_reviews={args.min_reviews}), 目标 {target}"
        )
        report["files"][f.name] = {"before": len(data), "after": len(kept)}
    (output_dir / "dataset_quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── 子命令: list-sources ──────────────────────────────────────────────────

def cmd_list_sources(args: argparse.Namespace) -> None:
    print("""
ACES-v2 数据来源
================

1. Amazon Reviews 2023 (McAuley Lab, HuggingFace)
   地址: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
   meta:  """ + HF_AMAZON_BASE + """/meta_categories/meta_<Category>.jsonl
   review: """ + HF_AMAZON_BASE + """/review_categories/<Category>.jsonl  (单文件约22GB)
   用途: expand-from-amazon, expand-by-keyword, enrich

2. My-Custom-AI 系列 (HuggingFace)
   - ACE-RS: Rationality Suite
   - ACE-SR: Search Results
   - ACE-BB: Buying Behavior
   用途: download-ace（实验记录格式，转为商品 JSON）

3. datasets_unified 现有文件
""")
    ud = Path("datasets_unified")
    if ud.exists():
        for j in sorted(ud.glob("*.json")):
            try:
                n = len(json.loads(j.read_text()))
            except Exception:
                n = "?"
            print(f"   {j.name}: {n} 个商品")
    else:
        print("   (目录不存在)")
    print()


# ── main ──────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="ACES-v2 数据集管理")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # expand-from-amazon
    p1 = sub.add_parser("expand-from-amazon", help="按 Amazon 类目扩充")
    p1.add_argument("-c", "--categories", help="逗号分隔类目")
    p1.add_argument("-n", "--per-category", type=int, default=50)
    p1.add_argument("--scan-limit", type=int, default=5000)
    p1.add_argument("--min-reviews", type=int, default=10)
    p1.add_argument("-o", "--output-dir", default="datasets_unified_v2")
    p1.add_argument("--skip-image-check", action="store_true")
    p1.add_argument("--list-categories", action="store_true")
    p1.set_defaults(func=cmd_expand_from_amazon)

    # expand-by-keyword
    p2 = sub.add_parser("expand-by-keyword", help="按关键词扩充 (mousepad, ski, smart_watch, athletic_shoes_men/women, bluetooth_speaker, insulated_cup_men/women, backpack_men/women)")
    p2.add_argument("-k", "--keywords", default="mousepad,ski")
    p2.add_argument("-n", "--max-products", type=int, default=80)
    p2.add_argument("--scan-limit", type=int, default=15000)
    p2.add_argument("--min-reviews", type=int, default=5)
    p2.add_argument("-o", "--output-dir", default="datasets_unified_v2")
    p2.add_argument("--skip-image-check", action="store_true", default=True)
    p2.add_argument("--no-skip-image-check", action="store_false", dest="skip_image_check")
    p2.add_argument("--ensure-reviews", action="store_true",
        help="仅保留 review 文件中存在的商品（需预扫 review，enrich 时必有评论）")
    p2.add_argument("--review-preload-limit", type=int, default=80000,
        help="预扫 review 行数限制（--ensure-reviews 时用）")
    p2.set_defaults(func=cmd_expand_by_keyword)

    # expand-custom-keyword
    p2b = sub.add_parser("expand-custom-keyword", help="自定义关键词扩充（无需改脚本内置 KEYWORD_CONFIG）")
    p2b.add_argument("--name", required=True, help="类目名/slug，例如 vase 或 usb_flash_drive")
    p2b.add_argument("--categories", required=True,
        help="Amazon 类目，逗号分隔，例如 Home_and_Kitchen,Tools_and_Home_Improvement")
    p2b.add_argument("--search-terms", default="",
        help="标题匹配词，逗号分隔；为空时默认使用 --name")
    p2b.add_argument("--out-file", default="",
        help="输出文件名（默认 <name>.json）")
    p2b.add_argument("--include-terms", default="",
        help="二次包含词（可选），例如 vase,flower vase,ceramic")
    p2b.add_argument("--exclude-terms", default="",
        help="二次排除词（可选），例如 artificial flower,cover,case")
    p2b.add_argument("-n", "--max-products", type=int, default=80)
    p2b.add_argument("--scan-limit", type=int, default=15000)
    p2b.add_argument("--min-reviews", type=int, default=5)
    p2b.add_argument("-o", "--output-dir", default="datasets_unified_v2")
    p2b.add_argument("--skip-image-check", action="store_true", default=True)
    p2b.add_argument("--no-skip-image-check", action="store_false", dest="skip_image_check")
    p2b.add_argument("--ensure-reviews", action="store_true",
        help="仅保留 review 文件中存在的商品（enrich 时必有评论）")
    p2b.add_argument("--review-preload-limit", type=int, default=80000,
        help="预扫 review 行数限制（--ensure-reviews 时用）")
    p2b.set_defaults(func=cmd_expand_custom_keyword)

    # download-ace
    p3 = sub.add_parser("download-ace", help="下载 My-Custom-AI ACE 数据集")
    p3.add_argument("-o", "--output-dir", default="datasets_unified")
    p3.set_defaults(func=cmd_download_ace)

    # enrich
    p4 = sub.add_parser("enrich", help="补齐多图与评论")
    p4.add_argument("--input-dir", default="datasets_unified")
    p4.add_argument("--output-dir", default="datasets_unified_v2")
    p4.add_argument("--files", help="逗号分隔，如 electronics,mousepad")
    p4.add_argument("--max-reviews", type=int, default=10, help="每个商品最多保留几条评论")
    p4.add_argument("--meta-scan-limit", type=int, default=0, help="meta 扫描行数限制，0=无限制")
    p4.add_argument("--review-scan-limit", type=int, default=200000,
        help="review 扫描行数限制（默认20万，单文件22GB慎用0=无限制）")
    p4.add_argument("--require-complete", action="store_true", help="仅保留含主图+多图+评论的商品")
    p4.add_argument("--min-images", type=int, default=0, help="仅保留图片数>=N的商品（0表示不限制）")
    p4.add_argument("--min-reviews", type=int, default=0, help="仅保留评论数>=N的商品（0表示不限制）")
    p4.set_defaults(func=cmd_enrich)

    # filter-reviews
    p5 = sub.add_parser("filter-reviews", help="只保留有评论的商品并裁剪到目标数量")
    p5.add_argument("--input-dir", default="datasets_unified")
    p5.add_argument("--output-dir", default="datasets_unified_v2")
    p5.add_argument("-n", "--target", type=int, default=40)
    p5.add_argument("--min-images", type=int, default=3, help="仅保留图片数>=N的商品")
    p5.add_argument("--min-reviews", type=int, default=5, help="仅保留评论数>=N的商品")
    p5.add_argument("--files", help="逗号分隔文件名（不含.json）")
    p5.set_defaults(func=cmd_filter_reviews)

    # filter-quality
    p5b = sub.add_parser("filter-quality", help="筛除分类不符、跨文件重复的商品")
    p5b.add_argument("--input-dir", default="datasets_unified")
    p5b.add_argument("--output-dir", default="datasets_unified_v2")
    p5b.add_argument("-n", "--target", type=int, default=40)
    p5b.add_argument("--files", help="逗号分隔文件名（不含.json）")
    p5b.add_argument("-v", "--verbose", action="store_true", help="打印筛除详情")
    p5b.set_defaults(func=cmd_filter_quality)

    # list-sources
    p6 = sub.add_parser("list-sources", help="列出数据来源与现有数据集")
    p6.set_defaults(func=cmd_list_sources)

    args = parser.parse_args()

    if args.cmd == "expand-from-amazon" and getattr(args, "list_categories", False):
        print("\nAmazon Reviews 2023 类目:\n")
        for i, c in enumerate(AMAZON_CATEGORIES, 1):
            m = " *" if c in RECOMMENDED_CATEGORIES else ""
            print(f"  {i:2d}. {c}{m}")
        print("\n  * = 推荐类目")
        return 0

    args.func(args)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n用户中断")
        sys.exit(130)
