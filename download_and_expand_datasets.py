"""
一键下载并扩展 ACES 数据集

从 HuggingFace 下载完整数据集并转换为 marketplace 格式。
原数据: 72 个商品
扩展后: 数千个商品

数据集来源: https://huggingface.co/My-Custom-AI/collections
"""

import sys
import json
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_and_convert(
    dataset_name: str,
    subset: str = None,
    output_dir: Path = None
):
    """
    下载数据集并转换为 marketplace 格式
    
    Args:
        dataset_name: HuggingFace 数据集名称
        subset: 子集名称（可选）
        output_dir: 输出目录
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("需要安装 datasets 库")
        logger.error("运行: pip install datasets")
        return None
    
    logger.info(f"\n下载数据集: {dataset_name}")
    if subset:
        logger.info(f"子集: {subset}")
    
    try:
        # 下载数据集
        if subset:
            dataset = load_dataset(dataset_name, subset, split="data")
        else:
            dataset = load_dataset(dataset_name, split="data")
        
        logger.info(f"✓ 下载成功: {len(dataset)} 条实验记录")
        
        # 按 query 分组商品
        products_by_query = defaultdict(list)
        total_products = 0
        
        for row in dataset:
            query = row.get("query", "unknown")
            
            # 提取商品列表（假设是列表格式）
            if "title" in row and isinstance(row["title"], list):
                num_products = len(row["title"])
                
                for i in range(num_products):
                    product = {
                        "sku": row.get("sku", [f"{query}_{i}"] * num_products)[i],
                        "title": row["title"][i],
                        "price": float(str(row.get("price", [0] * num_products)[i]).replace("$", "").replace(",", "")),
                        "rating": row.get("rating", [None] * num_products)[i],
                        "rating_count": row.get("rating_count", [0] * num_products)[i],
                        "image_url": row.get("image_url", [""] * num_products)[i] if "image_url" in row else "",
                        "description": row.get("description", [""] * num_products)[i] if "description" in row else "",
                        "sponsored": bool(row.get("sponsored", [False] * num_products)[i]) if "sponsored" in row else False,
                        "best_seller": bool(row.get("best_seller", [False] * num_products)[i]) if "best_seller" in row else False,
                        "overall_pick": bool(row.get("overall_pick", [False] * num_products)[i]) if "overall_pick" in row else False,
                        "low_stock": bool(row.get("low_stock", [False] * num_products)[i]) if "low_stock" in row else False,
                    }
                    
                    products_by_query[query].append(product)
                    total_products += 1
        
        logger.info(f"✓ 提取了 {total_products} 个商品，分布在 {len(products_by_query)} 个查询类别")
        
        # 保存到文件
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for query, products in products_by_query.items():
                # 去重（根据 sku）
                seen_skus = set()
                unique_products = []
                for p in products:
                    if p["sku"] not in seen_skus:
                        seen_skus.add(p["sku"])
                        unique_products.append(p)
                
                # 保存为 JSON
                output_file = output_dir / f"{query}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(unique_products, f, indent=2, ensure_ascii=False)
                
                logger.info(f"  ✓ {output_file.name}: {len(unique_products)} 个商品")
        
        return products_by_query
    
    except Exception as e:
        logger.error(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    print("\n" + "="*70)
    print("下载并扩展 ACES 数据集")
    print("="*70)
    
    # 输出目录
    output_base = Path("./datasets_extended")
    output_base.mkdir(parents=True, exist_ok=True)
    
    # 数据集列表
    datasets = [
        # ACE-RS 有多个配置，我们下载所有配置
        {
            "name": "My-Custom-AI/ACE-RS",
            "subset": "absolute_and_random_price",
            "description": "Rationality Suite - 绝对和随机价格",
            "expected": "~1.5k samples"
        },
        {
            "name": "My-Custom-AI/ACE-RS",
            "subset": "instruction_following",
            "description": "Rationality Suite - 指令遵循",
            "expected": "~1.5k samples"
        },
        {
            "name": "My-Custom-AI/ACE-RS",
            "subset": "rating",
            "description": "Rationality Suite - 评分测试",
            "expected": "~1.5k samples"
        },
        {
            "name": "My-Custom-AI/ACE-RS",
            "subset": "relative_price",
            "description": "Rationality Suite - 相对价格",
            "expected": "~1.5k samples"
        },
        {
            "name": "My-Custom-AI/ACE-SR",
            "subset": None,
            "description": "Search Results (搜索结果)",
            "expected": "2k samples, 9 experiments"
        },
        {
            "name": "My-Custom-AI/ACE-BB",
            "subset": "choice_behavior",
            "description": "Buying Behavior (购买行为)",
            "expected": "9k samples, 56 experiments"
        },
    ]
    
    total_queries = set()
    total_products = 0
    successful_downloads = 0
    
    # 下载每个数据集
    for i, ds_info in enumerate(datasets, 1):
        print(f"\n[{i}/{len(datasets)}] " + "="*60)
        print(f"数据集: {ds_info['name']}")
        print(f"描述: {ds_info['description']}")
        print(f"预期: {ds_info['expected']}")
        print("="*70)
        
        # 创建子目录
        ds_short_name = ds_info['name'].split('/')[-1].lower()
        output_dir = output_base / ds_short_name
        
        # 下载并转换
        result = download_and_convert(
            ds_info['name'],
            ds_info.get('subset'),
            output_dir
        )
        
        if result:
            successful_downloads += 1
            total_queries.update(result.keys())
            for products in result.values():
                total_products += len(products)
    
    # 合并所有数据集到统一目录
    print(f"\n" + "="*70)
    print("合并数据集到统一目录...")
    print("="*70)
    
    unified_dir = Path("./datasets_unified")
    unified_dir.mkdir(parents=True, exist_ok=True)
    
    # 合并所有 JSON 文件（按 query）
    all_queries = defaultdict(list)
    
    for ds_dir in output_base.iterdir():
        if ds_dir.is_dir():
            for json_file in ds_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        products = json.load(f)
                    
                    query = json_file.stem
                    all_queries[query].extend(products)
                except Exception as e:
                    logger.error(f"读取失败 {json_file}: {e}")
    
    # 保存合并后的数据
    for query, products in all_queries.items():
        # 去重
        seen_skus = set()
        unique_products = []
        for p in products:
            if p["sku"] not in seen_skus:
                seen_skus.add(p["sku"])
                unique_products.append(p)
        
        output_file = unified_dir / f"{query}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(unique_products, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  ✓ {output_file.name}: {len(unique_products)} 个商品")
    
    # 最终统计
    print(f"\n" + "="*70)
    print("下载完成！")
    print("="*70)
    print(f"成功下载: {successful_downloads}/{len(datasets)} 个数据集")
    print(f"查询类别: {len(all_queries)} 个")
    print(f"总商品数: {sum(len(products) for products in all_queries.values())} 个")
    print(f"\n输出目录:")
    print(f"  - 分数据集: {output_base}")
    print(f"  - 统一目录: {unified_dir}")
    
    # 显示查询类别
    print(f"\n可用的查询类别:")
    for i, query in enumerate(sorted(all_queries.keys())[:20], 1):  # 只显示前20个
        count = len(all_queries[query])
        print(f"  {i:2d}. {query:30s} ({count:4d} 个商品)")
    
    if len(all_queries) > 20:
        print(f"  ... 还有 {len(all_queries) - 20} 个类别")
    
    # 使用方法
    print(f"\n" + "="*70)
    print("使用方法:")
    print("="*70)
    print(f"\n1. 启动 Web 服务器（使用扩展数据集）:")
    print(f"   python start_web_server.py --datasets-dir {unified_dir}")
    print(f"\n2. 或在代码中使用:")
    print(f"""   
   from aces.environments import WebRendererMarketplace
   
   marketplace = WebRendererMarketplace()
   marketplace.initialize({{
       "datasets_dir": "{unified_dir}"
   }})
   """)
    print(f"\n3. 对比原数据集:")
    print(f"   原数据集: 72 个商品（9 个类别）")
    print(f"   扩展后: {sum(len(products) for products in all_queries.values())} 个商品（{len(all_queries)} 个类别）")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        logger.error(f"执行失败: {e}")
        import traceback
        traceback.print_exc()
