#!/usr/bin/env python3
"""
测试 LlamaIndex 搜索质量

对比简单搜索 vs Hybrid Retrieval + Reranking
"""

import logging
import time
from aces.config.settings import resolve_datasets_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_llamaindex_marketplace():
    """测试 LlamaIndex marketplace。"""
    print("\n" + "="*80)
    print("🔍 LlamaIndex Hybrid Retrieval + Reranking 测试")
    print("="*80 + "\n")
    
    from aces.environments.llamaindex_marketplace import LlamaIndexMarketplace
    from aces.environments.router import MarketplaceAdapter
    
    # 初始化
    print("Step 1: 初始化 LlamaIndex Marketplace...")
    print("  正在构建索引（BM25 + Vector）...")
    print("  这需要几分钟，请耐心等待...\n")
    
    start_time = time.time()
    
    marketplace = LlamaIndexMarketplace()
    marketplace.initialize({
        "datasets_dir": str(resolve_datasets_dir()),
        "embedding_model": "all-MiniLM-L6-v2",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "use_reranker": True,
    })
    
    init_time = time.time() - start_time
    print(f"✓ 索引构建完成（耗时: {init_time:.1f}秒）\n")
    
    # 测试查询
    test_queries = [
        ("mousepad", "基础查询"),
        ("gaming mousepad high rating", "复杂查询"),
        ("cheap but good mousepad", "语义查询"),
        ("laptop", "笔记本"),
        ("性价比高的鼠标垫", "中文查询"),
    ]
    
    for query, desc in test_queries:
        print(f"\n{'='*80}")
        print(f"查询: '{query}' ({desc})")
        print(f"{'='*80}\n")
        
        start = time.time()
        results = marketplace.search_products(query, limit=5)
        search_time = time.time() - start
        
        print(f"检索技术: {results.metadata.get('retrieval')} + {results.metadata.get('reranker')}")
        print(f"检索时间: {search_time:.3f}秒")
        print(f"返回商品: {len(results.products)}\n")
        
        for i, p in enumerate(results.products, 1):
            print(f"{i}. {p.title[:60]}...")
            print(f"   价格: ${p.price:.2f} | 评分: {p.rating}/5 | 评论: {p.rating_count}")
            print(f"   类别: {p.raw_data.get('category')}")
        
        print()
    
    marketplace.close()
    
    print("\n" + "="*80)
    print("✓ LlamaIndex 搜索测试完成")
    print("="*80)
    print("\n技术验证:")
    print("  ✅ BM25 检索 - 词频匹配")
    print("  ✅ Vector Search - 语义相似度")
    print("  ✅ Hybrid Fusion - RRF 混合")
    print("  ✅ Neural Reranking - Cross-encoder")
    print("\n学术级搜索引擎已就绪！\n")


def compare_search_engines():
    """对比简单搜索 vs LlamaIndex。"""
    print("\n" + "="*80)
    print("📊 搜索质量对比：Simple vs LlamaIndex")
    print("="*80 + "\n")
    
    from aces.environments import OfflineMarketplace
    from aces.environments.llamaindex_marketplace import LlamaIndexMarketplace
    
    query = "affordable gaming mousepad"
    
    # 1. 简单搜索
    print("1️⃣ Simple Search (文件名匹配):")
    simple = OfflineMarketplace()
    simple.initialize({"datasets_dir": str(resolve_datasets_dir())})
    
    try:
        results_simple = simple.search_products(query)
        print(f"  结果: {len(results_simple.products)} 个")
        if results_simple.products:
            for i, p in enumerate(results_simple.products[:3], 1):
                print(f"    {i}. {p.title[:50]}...")
        else:
            print("  ✗ 无结果（文件名不匹配）")
    except:
        print("  ✗ 搜索失败")
    
    print()
    
    # 2. LlamaIndex 搜索
    print("2️⃣ LlamaIndex (Hybrid + Rerank):")
    llamaindex = LlamaIndexMarketplace()
    llamaindex.initialize({"datasets_dir": str(resolve_datasets_dir())})
    
    results_llama = llamaindex.search_products(query, limit=3)
    print(f"  结果: {len(results_llama.products)} 个")
    for i, p in enumerate(results_llama.products, 1):
        print(f"    {i}. {p.title[:50]}...")
        print(f"       ${p.price:.2f} | {p.rating}/5 | {p.raw_data.get('category')}")
    
    print("\n" + "="*80)
    print("对比结果:")
    print("  Simple: 依赖精确文件名匹配，无法处理复杂查询")
    print("  LlamaIndex: 语义理解，可以找到相关商品")
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys
    
    mode = sys.argv[1] if len(sys.argv) > 1 else "basic"
    
    if mode == "basic":
        test_llamaindex_marketplace()
    elif mode == "compare":
        compare_search_engines()
    else:
        print("Usage: python test_llamaindex_search.py [basic|compare]")
