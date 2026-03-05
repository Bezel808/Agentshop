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
  python manage_datasets.py list-sources
"""

from __future__ import annotations

import argparse
import asyncio
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
}

KEYWORD_CONFIG: Dict[str, Tuple[List[str], str]] = {
    "mousepad": (["Office_Products", "Electronics"], "mousepad.json"),
    "ski": (["Sports_and_Outdoors", "Clothing_Shoes_and_Jewelry"], "ski_jacket.json"),
}


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


def convert_amazon_item(item: Dict[str, Any], category: str, index: int) -> Optional[Dict[str, Any]]:
    """将 Amazon 元数据转为 ACES 格式。"""
    title = (item.get("title") or "").strip()
    if not title or len(title) < 5:
        return None
    price = _parse_price(item.get("price"))
    if price is None:
        return None
    image_url = _pick_image_url(item.get("images"))
    if not image_url:
        return None
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
    return {
        "sku": sku, "title": title, "price": price, "rating": avg_rating, "rating_count": rating_number,
        "image_url": image_url, "description": description,
        "sponsored": False, "best_seller": False, "overall_pick": False, "low_stock": False,
    }


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
    with requests.get(url, stream=True, timeout=(20, 180)) as resp:
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

def cmd_expand_by_keyword(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    total = 0
    for kw in keywords:
        if kw not in KEYWORD_CONFIG:
            logger.warning(f"未知关键词 '{kw}'，跳过")
            continue
        categories, out_filename = KEYWORD_CONFIG[kw]
        search_terms = ["ski", "snow jacket", "snowsuit", "snow suit"] if kw == "ski" else [kw]
        all_candidates = []
        seen = set()
        for cat in categories:
            raw = _stream_jsonl_from_hf(cat, max_lines=args.scan_limit)
            for idx, item in enumerate(raw):
                title = (item.get("title") or "").strip()
                if not any(t.lower() in title.lower() for t in search_terms):
                    continue
                norm = re.sub(r"\s+", " ", title.lower())
                if norm in seen:
                    continue
                p = convert_amazon_item(item, cat, idx)
                if p is None or (p.get("rating_count") or 0) < args.min_reviews:
                    continue
                seen.add(norm)
                all_candidates.append(p)
        all_candidates.sort(key=lambda x: (x.get("rating_count") or 0), reverse=True)
        shortlist = all_candidates[: args.max_products * 2]
        if not args.skip_image_check and shortlist:
            shortlist = asyncio.run(_check_image_urls(shortlist))
        final = shortlist[: args.max_products]
        slug = kw.lower().replace(" ", "_")
        for i, p in enumerate(final):
            p["sku"] = f"{slug}_{i}"
        if final:
            out_file = output_dir / out_filename
            out_file.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
            total += len(final)
            logger.info(f"  => {out_file.name}: {len(final)}")
    print(f"\n完成! 共 {total} 个商品 -> {output_dir}/")


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


def cmd_enrich(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = list(input_dir.glob("*.json"))
    if args.files:
        requested = {x.strip() for x in args.files.split(",")}
        files = [f for f in files if f.stem in requested]
    total_img, total_rev = 0, 0
    for f in files:
        stem = f.stem
        if stem not in FILE_TO_AMAZON_CATEGORY:
            (output_dir / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
            continue
        cats = [FILE_TO_AMAZON_CATEGORY[stem]] if isinstance(FILE_TO_AMAZON_CATEGORY[stem], str) else list(FILE_TO_AMAZON_CATEGORY[stem])
        products = json.loads(f.read_text(encoding="utf-8"))
        title_to_idx: Dict[str, List[int]] = {}
        for i, p in enumerate(products):
            key = _normalize_title(p.get("title", ""))
            if key:
                title_to_idx.setdefault(key, []).append(i)
            p.setdefault("image_urls", [])
            p.setdefault("reviews", [])

        used = set()
        idx_to_asin: Dict[int, str] = {}
        for amazon_cat in cats:
            if len(used) >= len(products):
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
                    if not products[chosen].get("image_url"):
                        products[chosen]["image_url"] = imgs[0]
                asin = item.get("parent_asin")
                if asin:
                    idx_to_asin[chosen] = asin
                    products[chosen]["parent_asin"] = asin
                desc = _extract_description(item)
                if desc:
                    cur = (products[chosen].get("description") or "").strip()
                    if not cur or len(desc) > len(cur):
                        products[chosen]["description"] = desc[:2000]
                if len(used) >= len(products):
                    break

        asin_to_cat = {a: next(c for c in cats) for a in idx_to_asin.values()}
        asin_to_idx: Dict[str, List[int]] = defaultdict(list)
        for i, a in idx_to_asin.items():
            asin_to_idx[a].append(i)
        cat_to_asins: Dict[str, set] = defaultdict(set)
        for a, c in asin_to_cat.items():
            cat_to_asins[c].add(a)
        reviews_by_asin: Dict[str, List[Dict]] = {a: [] for a in idx_to_asin.values()}
        for amazon_cat, asins in cat_to_asins.items():
            rv_url = f"{HF_AMAZON_BASE}/review_categories/{amazon_cat}.jsonl"
            for item in _iter_jsonl_from_url(rv_url, args.review_scan_limit or None):
                a = item.get("parent_asin")
                if a not in asins or len(reviews_by_asin[a]) >= args.max_reviews:
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

        for a, idxs in asin_to_idx.items():
            ranked = sorted(reviews_by_asin.get(a, []), key=lambda r: int(r.get("helpful_vote") or 0), reverse=True)[: args.max_reviews]
            for i in idxs:
                products[i]["reviews"] = ranked

        out_path = output_dir / f.name
        out_path.write_text(json.dumps(products, ensure_ascii=False, indent=2), encoding="utf-8")
        img_cov = sum(1 for p in products if isinstance(p.get("image_urls"), list) and len(p.get("image_urls", [])) > 0)
        rev_cov = sum(1 for p in products if isinstance(p.get("reviews"), list) and len(p.get("reviews", [])) > 0)
        total_img += img_cov
        total_rev += rev_cov
        logger.info(f"  {f.name}: image_covered={img_cov} review_covered={rev_cov}")
    print(f"\n完成! image_covered={total_img} review_covered={total_rev} -> {output_dir}/")


# ── 子命令: list-sources ──────────────────────────────────────────────────

def cmd_list_sources(args: argparse.Namespace) -> None:
    print("""
ACES-v2 数据来源
================

1. Amazon Reviews 2023 (McAuley Lab, HuggingFace)
   地址: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
   文件: raw/meta_categories/meta_{Category}.jsonl
        raw/review_categories/{Category}.jsonl
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
    p1.add_argument("-o", "--output-dir", default="datasets_unified")
    p1.add_argument("--skip-image-check", action="store_true")
    p1.add_argument("--list-categories", action="store_true")
    p1.set_defaults(func=cmd_expand_from_amazon)

    # expand-by-keyword
    p2 = sub.add_parser("expand-by-keyword", help="按关键词扩充 (mousepad, ski)")
    p2.add_argument("-k", "--keywords", default="mousepad,ski")
    p2.add_argument("-n", "--max-products", type=int, default=80)
    p2.add_argument("--scan-limit", type=int, default=15000)
    p2.add_argument("--min-reviews", type=int, default=5)
    p2.add_argument("-o", "--output-dir", default="datasets_unified")
    p2.add_argument("--skip-image-check", action="store_true", default=True)
    p2.add_argument("--no-skip-image-check", action="store_false", dest="skip_image_check")
    p2.set_defaults(func=cmd_expand_by_keyword)

    # download-ace
    p3 = sub.add_parser("download-ace", help="下载 My-Custom-AI ACE 数据集")
    p3.add_argument("-o", "--output-dir", default="datasets_unified")
    p3.set_defaults(func=cmd_download_ace)

    # enrich
    p4 = sub.add_parser("enrich", help="补齐多图与评论")
    p4.add_argument("--input-dir", default="datasets_unified")
    p4.add_argument("--output-dir", default="datasets_unified_multimodal")
    p4.add_argument("--files", help="逗号分隔，如 electronics,mousepad")
    p4.add_argument("--max-reviews", type=int, default=8)
    p4.add_argument("--meta-scan-limit", type=int, default=0)
    p4.add_argument("--review-scan-limit", type=int, default=0)
    p4.set_defaults(func=cmd_enrich)

    # list-sources
    p5 = sub.add_parser("list-sources", help="列出数据来源与现有数据集")
    p5.set_defaults(func=cmd_list_sources)

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
