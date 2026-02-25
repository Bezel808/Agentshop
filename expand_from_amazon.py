#!/usr/bin/env python3
"""
从 Amazon Reviews 2023 (McAuley Lab) 数据集清洗并扩充 ACES-v2 商品数据库。

数据源: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
使用 streaming 模式，无需将整个数据集下载到本地。

用法:
  # 默认：扩充 10 个常用类目，每类 ≤20 个商品
  python expand_from_amazon.py

  # 指定类目和数量
  python expand_from_amazon.py --categories "Electronics,Office_Products,Home_and_Kitchen" --per-category 30

  # 列出所有可用类目
  python expand_from_amazon.py --list-categories

  # 跳过图片检查（加快速度，但不保证图片可用）
  python expand_from_amazon.py --skip-image-check

输出: datasets_unified/ 目录下按类目生成 JSON 文件
"""

import json
import logging
import re
import sys
import time
import argparse
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Amazon Reviews 2023 的 33 个类目 ──────────────────────────────────
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

# 适合行为经济学实验的推荐类目（商品有明确价格和图片的类目）
RECOMMENDED_CATEGORIES = [
    "Electronics",
    "Office_Products",
    "Home_and_Kitchen",
    "Sports_and_Outdoors",
    "Health_and_Household",
    "Beauty_and_Personal_Care",
    "Toys_and_Games",
    "Tools_and_Home_Improvement",
    "Pet_Supplies",
    "Cell_Phones_and_Accessories",
]

# ── 字段映射 & 清洗 ─────────────────────────────────────────────────

def _parse_price(raw: Any) -> Optional[float]:
    """将各种格式的 price 解析为 float，无效则返回 None。"""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw) if raw > 0 else None
    s = str(raw).strip()
    if not s or s.lower() in ("none", "n/a", ""):
        return None
    s = re.sub(r"[^\d.]", "", s)
    try:
        v = float(s)
        return v if v > 0 else None
    except ValueError:
        return None


def _pick_image_url(images: Any) -> Optional[str]:
    """从 Amazon 数据集的 images 字段中选取最佳图片 URL。
    
    JSONL 格式:  list[dict], 每个 dict 有 thumb/large/hi_res 键
    datasets 格式: dict with keys thumb/large/hi_res, 值为 list
    优先级: large > hi_res > thumb
    """
    if not images:
        return None

    # JSONL 格式: list of {"thumb": ..., "large": ..., "hi_res": ...}
    if isinstance(images, list):
        for entry in images:
            if not isinstance(entry, dict):
                continue
            for key in ("large", "hi_res", "thumb"):
                url = entry.get(key)
                if url and isinstance(url, str) and url.startswith("http"):
                    return url
        return None

    # datasets 库格式: {"large": [...], "hi_res": [...], "thumb": [...]}
    if isinstance(images, dict):
        for key in ("large", "hi_res", "thumb"):
            urls = images.get(key)
            if not urls or not isinstance(urls, list):
                continue
            for url in urls:
                if url and isinstance(url, str) and url.startswith("http"):
                    return url

    return None


def _join_text(lst: Any) -> str:
    """将 list[str] 拼成一段文字。"""
    if isinstance(lst, list):
        parts = [str(x).strip() for x in lst if x]
        return " ".join(parts)
    if isinstance(lst, str):
        return lst.strip()
    return ""


def _sanitize_category_name(category: str) -> str:
    """把 Amazon 类目名变成文件名友好的查询词。
    例: 'Office_Products' -> 'office_products'
    """
    return category.lower().replace(" ", "_").replace("&", "and")


def convert_item(item: Dict[str, Any], category: str, index: int) -> Optional[Dict[str, Any]]:
    """将 Amazon 元数据记录转换为 ACES 格式。返回 None 表示不合格。"""
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
    description = description[:2000]  # 截断过长描述

    return {
        "sku": sku,
        "title": title,
        "price": price,
        "rating": avg_rating,
        "rating_count": rating_number,
        "image_url": image_url,
        "description": description,
        "sponsored": False,
        "best_seller": False,
        "overall_pick": False,
        "low_stock": False,
    }


# ── 图片 URL 可用性检查（异步批量） ────────────────────────────────

async def check_image_urls(products: List[Dict], concurrency: int = 20, timeout: float = 8.0) -> List[Dict]:
    """异步并发 HEAD 检查 image_url，过滤掉不可访问的。"""
    import aiohttp

    sem = asyncio.Semaphore(concurrency)
    valid: List[Dict] = []
    checked = 0
    total = len(products)

    async def _check(product: Dict):
        nonlocal checked
        url = product["image_url"]
        try:
            async with sem:
                async with aiohttp.ClientSession() as session:
                    async with session.head(url, timeout=aiohttp.ClientTimeout(total=timeout), allow_redirects=True) as resp:
                        checked += 1
                        if resp.status < 400:
                            valid.append(product)
                        else:
                            logger.debug(f"  [SKIP] {resp.status} {url[:80]}")
                        if checked % 50 == 0:
                            logger.info(f"  图片检查进度: {checked}/{total} (有效: {len(valid)})")
        except Exception:
            checked += 1
            logger.debug(f"  [SKIP] timeout/error {url[:80]}")

    tasks = [_check(p) for p in products]
    await asyncio.gather(*tasks)

    logger.info(f"  图片检查完成: {len(valid)}/{total} 有效")
    return valid


# ── 主流程 ──────────────────────────────────────────────────────────

def _stream_jsonl_from_hf(category: str, max_lines: int) -> List[Dict]:
    """
    从 HuggingFace 仓库流式读取 JSONL 元数据文件。
    只下载需要的行数，不会拉取整个文件。
    自带重试机制。
    """
    import requests as _req
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    url = (
        f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023"
        f"/resolve/main/raw/meta_categories/meta_{category}.jsonl"
    )

    logger.info(f"  流式读取: meta_{category}.jsonl (最多 {max_lines} 行)...")

    session = _req.Session()
    retries = Retry(total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    items: List[Dict] = []
    try:
        with session.get(url, stream=True, timeout=(15, 120)) as resp:
            resp.raise_for_status()
            line_count = 0
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line_count += 1
                if line_count > max_lines:
                    break
                try:
                    items.append(json.loads(raw_line))
                except json.JSONDecodeError:
                    continue
                if line_count % 1000 == 0:
                    logger.info(f"    已读取 {line_count} 行...")
    except Exception as e:
        logger.error(f"  下载失败: {e}")
        if items:
            logger.info(f"  部分成功: 已读取 {len(items)} 条")
        return items

    logger.info(f"  读取完毕: {len(items)} 条记录")
    return items


def _stream_parquet_from_hf(category: str, max_rows: int) -> List[Dict]:
    """
    如果该类目有预拆分的 Parquet 文件夹，用 datasets 库流式读取。
    仅读取第一个 shard 的前 max_rows 行。
    """
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_url
        import requests as _req
        import io

        shard_url = hf_hub_url(
            repo_id="McAuley-Lab/Amazon-Reviews-2023",
            filename=f"raw_meta_{category}/full-00000-of-00001.parquet",
            repo_type="dataset",
        )
        # 对大类可能有多个 shard
        # 我们只取第一个 shard 的前 N 行就够了
        logger.info(f"  尝试 Parquet shard (比 JSONL 快)...")
        resp = _req.get(shard_url, timeout=30)
        resp.raise_for_status()
        table = pq.read_table(io.BytesIO(resp.content))
        df = table.to_pandas().head(max_rows)
        return df.to_dict("records")
    except Exception:
        return []  # fallback to JSONL


def stream_category(
    category: str,
    max_candidates: int = 200,
    max_products: int = 20,
    min_rating_count: int = 10,
    skip_image_check: bool = False,
) -> List[Dict]:
    """
    从 HuggingFace 流式读取一个类目的元数据，清洗后返回商品列表。
    
    Args:
        category: Amazon 类目名 (如 "Electronics")
        max_candidates: 流式扫描上限（不必下载整个类目）
        max_products: 最终保留的商品数
        min_rating_count: 最低评论数（过滤冷门商品）
        skip_image_check: 跳过图片 URL HEAD 检查
    """
    # 读取原始数据（JSONL 流式，无需全量下载）
    raw_items = _stream_jsonl_from_hf(category, max_lines=max_candidates)

    if not raw_items:
        logger.warning(f"  未能读取到数据")
        return []

    # 清洗转换
    candidates: List[Dict] = []
    for idx, item in enumerate(raw_items):
        product = convert_item(item, category, idx)
        if product is None:
            continue
        if product.get("rating_count") is not None and product["rating_count"] < min_rating_count:
            continue
        candidates.append(product)

    logger.info(f"  清洗完毕: {len(raw_items)} 条中有 {len(candidates)} 条合格")

    if not candidates:
        return []

    # 按评论数降序，取评价最多（最有代表性）的商品
    candidates.sort(key=lambda p: (p.get("rating_count") or 0), reverse=True)
    # 多取一些候选，用于图片检查后还有足够的
    shortlist = candidates[: max_products * 3]

    # 图片可用性检查
    if not skip_image_check and shortlist:
        logger.info(f"  检查 {len(shortlist)} 个候选的图片 URL...")
        shortlist = asyncio.run(check_image_urls(shortlist))

    # 取 top N
    final = shortlist[:max_products]

    # 重新编号 sku
    cat_slug = _sanitize_category_name(category)
    for i, p in enumerate(final):
        p["sku"] = f"{cat_slug}_{i}"

    return final


def main():
    parser = argparse.ArgumentParser(
        description="从 Amazon Reviews 2023 清洗扩充 ACES-v2 商品数据库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python expand_from_amazon.py                                    # 默认 10 个推荐类目
  python expand_from_amazon.py --categories Electronics,Books     # 指定类目
  python expand_from_amazon.py --per-category 30 --scan-limit 500 # 每类30个, 多扫描
  python expand_from_amazon.py --list-categories                  # 列出所有类目
  python expand_from_amazon.py --skip-image-check                 # 跳过图片检查(快)
""",
    )
    parser.add_argument(
        "--categories", "-c",
        help="逗号分隔的类目列表 (默认: 10个推荐类目)",
    )
    parser.add_argument(
        "--per-category", "-n",
        type=int, default=50,
        help="每个类目保留的商品数 (默认: 50)",
    )
    parser.add_argument(
        "--scan-limit",
        type=int, default=5000,
        help="每个类目流式扫描的最大记录数 (默认: 5000，越大越慢但商品质量越高)",
    )
    parser.add_argument(
        "--min-reviews",
        type=int, default=10,
        help="最低评论数, 过滤冷门商品 (默认: 10)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="datasets_unified",
        help="输出目录 (默认: datasets_unified)",
    )
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="跳过图片 URL 可用性检查 (加快速度)",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="列出所有可用的 Amazon 类目",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印计划, 不执行",
    )

    args = parser.parse_args()

    if args.list_categories:
        print("\nAmazon Reviews 2023 可用类目 (共 33 个):\n")
        for i, cat in enumerate(AMAZON_CATEGORIES, 1):
            marker = " *" if cat in RECOMMENDED_CATEGORIES else ""
            print(f"  {i:2d}. {cat}{marker}")
        print("\n  * = 推荐类目 (适合行为经济学实验)")
        print(f"\n使用: python expand_from_amazon.py --categories \"{','.join(RECOMMENDED_CATEGORIES[:3])}\"")
        return

    # 确定要处理的类目
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]
        for c in categories:
            if c not in AMAZON_CATEGORIES:
                logger.warning(f"  未知类目: {c} (可用类目见 --list-categories)")
    else:
        categories = RECOMMENDED_CATEGORIES

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  ACES-v2 商品数据库扩充 (Amazon Reviews 2023)")
    print("=" * 70)
    print(f"  类目: {len(categories)} 个")
    print(f"  每类: ≤{args.per_category} 个商品")
    print(f"  扫描上限: {args.scan_limit} 条/类")
    print(f"  最低评论数: {args.min_reviews}")
    print(f"  图片检查: {'跳过' if args.skip_image_check else '启用'}")
    print(f"  输出: {output_dir}/")
    print("=" * 70 + "\n")

    if args.dry_run:
        print("[DRY RUN] 以上为执行计划, 未实际下载。去掉 --dry-run 执行。")
        return

    # 统计
    total_products = 0
    stats: List[Tuple[str, int]] = []
    t0 = time.time()

    for i, category in enumerate(categories, 1):
        print(f"\n[{i}/{len(categories)}] ── {category} ──")

        products = stream_category(
            category=category,
            max_candidates=args.scan_limit,
            max_products=args.per_category,
            min_rating_count=args.min_reviews,
            skip_image_check=args.skip_image_check,
        )

        if not products:
            logger.warning(f"  {category}: 无合格商品, 跳过")
            stats.append((category, 0))
            continue

        # 保存
        slug = _sanitize_category_name(category)
        out_file = output_dir / f"{slug}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(products, f, indent=2, ensure_ascii=False)

        total_products += len(products)
        stats.append((category, len(products)))
        logger.info(f"  => 保存 {len(products)} 个商品到 {out_file.name}")

    elapsed = time.time() - t0

    # 最终统计
    print("\n" + "=" * 70)
    print("  扩充完成!")
    print("=" * 70)
    print(f"\n  {'类目':<40s} {'商品数':>6s}")
    print(f"  {'─' * 40} {'─' * 6}")

    # 统计已有的（非本次生成的）
    existing_files = set()
    for f in output_dir.glob("*.json"):
        existing_files.add(f.stem)

    for category, count in stats:
        slug = _sanitize_category_name(category)
        marker = " (new)" if slug not in existing_files else ""
        print(f"  {category:<40s} {count:>6d}{marker}")

    # 列出未被本次处理但已存在的文件
    processed_slugs = {_sanitize_category_name(c) for c in categories}
    other = sorted(existing_files - processed_slugs)
    if other:
        print(f"\n  已有数据集 (未处理):")
        for slug in other:
            fpath = output_dir / f"{slug}.json"
            try:
                n = len(json.loads(fpath.read_text()))
            except Exception:
                n = "?"
            print(f"  {slug:<40s} {n:>6}")

    total_in_dir = 0
    for f in output_dir.glob("*.json"):
        try:
            total_in_dir += len(json.loads(f.read_text()))
        except Exception:
            pass

    print(f"\n  本次新增: {total_products} 个商品")
    print(f"  数据库总计: {total_in_dir} 个商品, {len(list(output_dir.glob('*.json')))} 个类目")
    print(f"  耗时: {elapsed:.0f} 秒")
    print(f"\n  使用方式:")
    print(f"    ./aces_ctl.sh start              # 启动 Web 服务")
    print(f"    ./aces_ctl.sh datasets           # 查看所有类目")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(130)
