#!/usr/bin/env python3
"""
ACES v2 实验运行器

运行完整的 Agent-Marketplace 实验。
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


from aces.config.settings import resolve_datasets_dir
from aces.llm_backends.factory import LLMBackendFactory
from aces.perception.factory import PerceptionFactory
from aces.core.protocols import Observation


def _load_condition_set(condition_file: str):
    """加载实验条件集（YAML 或 JSON）。"""
    from aces.experiments.control_variables import (
        load_conditions_from_yaml,
        load_conditions_from_json,
        ConditionSet,
    )
    if condition_file.endswith((".yaml", ".yml")):
        return load_conditions_from_yaml(condition_file)
    elif condition_file.endswith(".json"):
        return load_conditions_from_json(condition_file)
    else:
        raise ValueError(f"不支持的条件文件格式: {condition_file}")


def _apply_condition_to_provider(marketplace_provider, condition):
    """将实验条件注入 marketplace provider（仅 offline 模式支持）。"""
    if hasattr(marketplace_provider, "set_condition"):
        marketplace_provider.set_condition(condition)
        return True
    else:
        logger.warning("当前 marketplace 不支持实验条件注入（仅 offline 模式支持）")
        return False


def run_simple_experiment(
    query: str = "mousepad",
    num_trials: int = 5,
    perception_mode: str = "verbal",
    api_key: str = None,
    condition_file: str = None,
):
    """
    运行简单实验（不需要实际调用 LLM）。
    
    用于测试系统集成。
    """
    from aces.environments import MarketplaceFactory, MarketplaceAdapter
    from aces.perception import VerbalPerception, VisualPerception
    from aces.tools import SearchTool, AddToCartTool
    import time
    import json
    
    # 加载实验条件
    condition_set = None
    if condition_file:
        condition_set = _load_condition_set(condition_file)
        print(f"\n已加载实验条件集: {condition_set.name}")
        print(f"  设计: {condition_set.design}")
        print(f"  条件: {condition_set.all_condition_names()}")
    
    print("\n" + "="*80)
    print(f"运行简单实验: {query} ({num_trials} trials)")
    print("="*80 + "\n")
    
    # 创建输出目录
    output_dir = Path("experiment_results") / f"simple_test_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建 marketplace
    marketplace_provider = MarketplaceFactory.create({
        "mode": "offline",
        "offline": {
            "datasets_dir": str(resolve_datasets_dir()),
        }
    })
    marketplace = MarketplaceAdapter(marketplace_provider)
    
    # 创建 tools
    tools = [
        SearchTool(marketplace),
        AddToCartTool(marketplace),
    ]
    
    # 创建感知模式
    if perception_mode == "verbal":
        perception = VerbalPerception()
    else:
        perception = VisualPerception()
    
    print(f"配置:")
    print(f"  Query: {query}")
    print(f"  Perception: {perception.get_modality()}")
    print(f"  Tools: {[t.get_schema().name for t in tools]}")
    if condition_set:
        print(f"  Condition Set: {condition_set.name} ({condition_set.design})")
    print()
    
    # 保存实验条件配置
    if condition_set:
        cond_file = output_dir / "condition_set.json"
        with open(cond_file, 'w', encoding='utf-8') as f:
            json.dump(condition_set.to_dict(), f, indent=2, ensure_ascii=False)
    
    # 运行多个trial
    results = []
    
    for trial in range(num_trials):
        # 确定本 trial 的实验条件
        condition = None
        condition_name = "none"
        if condition_set:
            condition = condition_set.assign(trial)
            condition_name = condition.name
            _apply_condition_to_provider(marketplace_provider, condition)
        
        print(f"Trial {trial + 1}/{num_trials} [条件: {condition_name}]:")
        
        # 重置环境
        marketplace.reset()
        
        # 搜索商品
        search_result = tools[0].execute({"query": query, "limit": 5})
        
        if not search_result.success:
            print(f"  ✗ 搜索失败")
            continue
        
        products = search_result.data.get("products", [])
        print(f"  ✓ 找到 {len(products)} 个商品")
        
        # 模拟决策：选择第一个商品
        if products:
            product = products[0]
            print(f"  选择: {product['title'][:50]}...")
            
            # 加购
            cart_result = tools[1].execute({
                "product_id": product["id"],
                "product_title": product["title"],
            })
            
            if cart_result.success:
                print(f"  ✓ 加购成功")
                
                results.append({
                    "trial": trial,
                    "condition": condition_name,
                    "query": query,
                    "selected_product_id": product["id"],
                    "selected_product_rank": product.get("position", 0) + 1,
                    "selected_product_price": product["price"],
                    "selected_product_title": product["title"],
                    "success": True,
                })
            else:
                print(f"  ✗ 加购失败")
        
        # 清除条件，准备下一 trial
        if condition_set and hasattr(marketplace_provider, "clear_condition"):
            marketplace_provider.clear_condition()
        
        print()
    
    # 保存结果
    results_file = output_dir / "results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 汇总
    success_rate = sum(r["success"] for r in results) / len(results) if results else 0
    avg_rank = sum(r["selected_product_rank"] for r in results) / len(results) if results else 0
    
    # 分条件汇总
    if condition_set:
        print("="*80)
        print("分条件结果:")
        from collections import defaultdict
        by_cond = defaultdict(list)
        for r in results:
            by_cond[r["condition"]].append(r)
        for cond_name, cond_results in by_cond.items():
            avg_price = sum(r["selected_product_price"] for r in cond_results) / len(cond_results)
            avg_r = sum(r["selected_product_rank"] for r in cond_results) / len(cond_results)
            print(f"  [{cond_name}] trials={len(cond_results)}, "
                  f"avg_rank={avg_r:.1f}, avg_price=${avg_price:.2f}")
    
    print("="*80)
    print("实验结果:")
    print(f"  成功率: {success_rate:.1%}")
    print(f"  平均选择排名: {avg_rank:.1f}")
    print(f"  结果保存到: {results_file}")
    print("="*80 + "\n")
    
    marketplace.close()
    
    return results


def run_with_real_llm(
    llm_backend: str,
    model: str,
    api_key: str,
    query: str = "mousepad",
    perception_mode: str = "verbal",
    marketplace_mode: str = "offline",
    datasets_dir: str = "datasets_unified",
    system_prompt: str = None,
    temperature: float = 1.0,
    limit: int = 5,
    verbose: bool = False,
    condition_file: str = None,
    condition_name: str = None,
):
    """
    使用真实 LLM 运行实验。
    
    支持多种 LLM backend 和配置选项。
    
    Args:
        condition_file: 实验条件 YAML/JSON 文件路径
        condition_name: 指定使用条件集中的哪个条件（不指定则用第一个）
    """
    from aces.environments import MarketplaceFactory, MarketplaceAdapter
    from aces.agents import ComposableAgent
    from aces.tools import SearchTool, AddToCartTool
    
    print("\n" + "="*80)
    print(f"ACES v2 实验: {llm_backend.upper()} + {perception_mode.upper()}")
    print("="*80 + "\n")
    
    # 加载实验条件
    active_condition = None
    if condition_file:
        cond_set = _load_condition_set(condition_file)
        print(f"已加载实验条件集: {cond_set.name}")
        print(f"  可用条件: {cond_set.all_condition_names()}")
        if condition_name:
            active_condition = next(
                (c for c in cond_set.conditions if c.name == condition_name), None
            )
            if not active_condition:
                print(f"  ⚠️  未找到条件 '{condition_name}'，使用第一个")
                active_condition = cond_set.conditions[0]
        else:
            active_condition = cond_set.conditions[0]
        print(f"  当前使用: {active_condition.name} — {active_condition.description}\n")
    
    # 创建 marketplace
    print(f"📦 创建 Marketplace ({marketplace_mode} 模式)...")
    marketplace_config = {"mode": marketplace_mode}
    
    if marketplace_mode == "offline":
        marketplace_config["offline"] = {"datasets_dir": str(resolve_datasets_dir(datasets_dir))}
    elif marketplace_mode == "llamaindex":
        marketplace_config["llamaindex"] = {
            "datasets_dir": str(resolve_datasets_dir(datasets_dir)),
            "use_reranker": True
        }
    
    marketplace_provider = MarketplaceFactory.create(marketplace_config)
    marketplace = MarketplaceAdapter(marketplace_provider)
    print(f"   ✓ Marketplace 创建完成\n")
    
    # 注入实验条件
    if active_condition:
        _apply_condition_to_provider(marketplace_provider, active_condition)
    
    # 创建 tools
    tools = [
        SearchTool(marketplace),
        AddToCartTool(marketplace),
    ]
    
    # 创建 LLM backend
    print(f"🤖 创建 LLM Backend ({llm_backend})...")
    
    llm = LLMBackendFactory.create({
        "backend": llm_backend,
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
    })
    
    print(f"   ✓ 模型: {llm.model_name}\n")
    
    # 创建 perception
    print(f"👁️  创建感知模式 ({perception_mode})...")
    perception = PerceptionFactory.create({
        "mode": perception_mode,
        "format_style": "structured",
        "detail_level": "high",
    })
    print(f"   ✓ 模态: {perception.get_modality()}\n")
    
    # 创建 system prompt
    if not system_prompt:
        system_prompt = f"""你是一个理性的购物助手。你的任务是帮用户找到最好的{query}。

请按照以下步骤操作：
1. 使用 search_products 工具搜索商品（支持翻页 page、价格筛选 price_min/price_max、星级筛选 rating_min）
2. 若当前页没有合适商品，可传入 page=2 等翻页，或设置 price_min/price_max/rating_min 筛选
3. 仔细分析商品的价格、评分、评价数量等信息
4. 考虑性价比，选择最合适的商品
5. 使用 add_to_cart 工具将选中的商品加入购物车

请用中文解释你的决策理由。"""
    
    # 创建 agent
    print(f"🧠 组装 Agent...")
    agent = ComposableAgent(
        llm=llm,
        perception=perception,
        tools=tools,
        system_prompt=system_prompt
    )
    print(f"   ✓ Agent 就绪\n")
    
    if verbose:
        print(f"配置详情:")
        print(f"  LLM: {agent.llm.model_name}")
        print(f"  温度: {temperature}")
        print(f"  感知: {agent.perception.get_modality()}")
        print(f"  工具: {list(agent.tools.keys())}")
        print()
    
    # 工具执行循环：Agent 可自主调用 search_products 翻页/筛选，直至 add_to_cart
    search_params = {
        "query": query,
        "limit": limit,
        "page": 1,
        "price_min": None,
        "price_max": None,
        "rating_min": None,
    }
    max_steps = 15
    products = []
    step = 0

    try:
        while step < max_steps:
            step += 1
            print("="*80)
            print(f"🔍 搜索: {search_params['query']} (第 {search_params['page']} 页, limit={search_params['limit']})")
            if search_params.get("price_min") or search_params.get("price_max"):
                print(f"   价格筛选: {search_params.get('price_min')}-{search_params.get('price_max')}")
            if search_params.get("rating_min"):
                print(f"   星级筛选: >={search_params['rating_min']}")
            print("="*80 + "\n")

            results = marketplace.search_products(
                search_params["query"],
                limit=search_params["limit"],
                page=search_params["page"],
                price_min=search_params.get("price_min"),
                price_max=search_params.get("price_max"),
                rating_min=search_params.get("rating_min"),
            )
            products = results.products
            page = getattr(results, "page", 1)
            total_pages = getattr(results, "total_pages", 1) or 1

            if not products:
                print(f"❌ 未找到商品\n")
                if step == 1:
                    marketplace.close()
                    return
                break

            print(f"✓ 找到 {len(products)} 个商品 (第 {page}/{total_pages} 页):\n")
            for i, p in enumerate(products, 1):
                print(f"  {i}. {p.title}")
                print(f"     价格: ${p.price:.2f} | 评分: {p.rating or 'N/A'} ⭐ ({p.rating_count or 0} 评价)")

            base_obs = agent.perception.encode(products)
            page_info = (
                f"[当前第 {page}/{total_pages} 页。"
                f"可调用 search_products 传入 page、price_min、price_max、rating_min 翻页或筛选。]\n\n"
            )
            observation = Observation(
                data=page_info + (base_obs.data if isinstance(base_obs.data, str) else str(base_obs.data)),
                modality=base_obs.modality,
                timestamp=base_obs.timestamp,
                metadata=base_obs.metadata,
            )

            print("\n🤖 Agent 决策中...\n")
            action = agent.act(observation)

            print(f"决策: {action.tool_name} {action.parameters}")
            if action.reasoning:
                print(f"理由: {action.reasoning}\n")

            if action.tool_name == "search_products":
                params = action.parameters or {}
                search_params["query"] = params.get("query", search_params["query"])
                search_params["limit"] = params.get("limit", search_params["limit"])
                search_params["page"] = params.get("page", 1)
                search_params["price_min"] = params.get("price_min")
                search_params["price_max"] = params.get("price_max")
                search_params["rating_min"] = params.get("rating_min")
                continue

            if action.tool_name == "add_to_cart":
                selected_id = action.parameters.get("product_id")
                selected_product = next((p for p in products if p.id == selected_id), None)
                if selected_product:
                    rank = products.index(selected_product) + 1
                    print("\n" + "="*80)
                    print("✅ 已选择商品")
                    print("="*80)
                    print(f"\n  商品: {selected_product.title}")
                    print(f"  价格: ${selected_product.price:.2f}")
                    print(f"  评分: {selected_product.rating or 'N/A'} ⭐")
                    print(f"  排名: #{rank}/{len(products)}")
                else:
                    print(f"\n⚠️  未找到商品 ID: {selected_id}")
                break

            break  # noop 或其他

    except Exception as e:
        print(f"\n❌ Agent 决策失败: {e}")
        if verbose:
            import traceback
            traceback.print_exc()

    marketplace.close()
    print("\n" + "="*80)
    print("实验完成!")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='ACES v2 通用实验运行器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # VLM 实验（默认 qwen + visual）
  python run_experiment.py --query mousepad

  # 指定 query 和 API key
  python run_experiment.py --llm qwen --perception visual --query "性价比高的鼠标垫"
  
  # Verbal 模式（文本输入）
  python run_experiment.py --llm qwen --perception verbal --query toothpaste
  
  # 简单模式（不调用 LLM）
  python run_experiment.py --mode simple --query mousepad --trials 5

  # 使用实验条件文件（价格锚定实验）
  python run_experiment.py --query mousepad \\
    -C configs/experiments/example_price_anchoring.yaml --condition-name treatment_anchor

  # 简单模式 + 条件文件（自动按 trial 轮转 control/treatment）
  python run_experiment.py --mode simple --query mousepad --trials 10 \\
    -C configs/experiments/example_decoy_effect.yaml
        """
    )
    
    # 基础参数
    parser.add_argument('--mode', choices=['simple', 'llm'], default='llm',
                      help='实验模式: simple (模拟，不调用LLM) 或 llm (真实LLM)')
    parser.add_argument('--query', default='mousepad',
                      help='搜索查询 (如: mousepad, toothpaste)')
    
    # LLM 配置
    parser.add_argument('--llm', choices=['deepseek', 'openai', 'qwen'],
                      default='qwen',
                      help='LLM 后端 (默认: qwen，VLM 实验用)')
    parser.add_argument('--model', 
                      help='模型名称 (如: deepseek-chat, gpt-4o-mini, qwen-turbo)')
    parser.add_argument('--api-key',
                      help='API key (或使用环境变量)')
    parser.add_argument('--temperature', type=float, default=1.0,
                      help='采样温度 (0.0-2.0, 默认: 1.0)')
    
    # Agent 配置
    parser.add_argument('--perception', choices=['visual', 'verbal'], 
                      default='visual',
                      help='感知模式 (默认: visual，VLM 看图决策)')
    parser.add_argument('--system-prompt',
                      help='自定义系统提示')
    
    # Marketplace 配置
    parser.add_argument('--marketplace', choices=['offline', 'online', 'llamaindex'],
                      default='offline',
                      help='Marketplace 模式 (默认: offline)')
    parser.add_argument('--datasets-dir', default='datasets_unified',
                      help='数据集目录 (默认: datasets_unified)')
    
    # 实验配置
    parser.add_argument('--trials', type=int, default=1,
                      help='试次数量 (simple 模式使用)')
    parser.add_argument('--limit', type=int, default=5,
                      help='搜索结果数量 (默认: 5)')
    
    # 实验条件（控制变量）
    parser.add_argument('--condition-file', '-C',
                      help='实验条件文件 (YAML/JSON)，定义价格/标签/名称等控制变量')
    parser.add_argument('--condition-name',
                      help='指定使用条件集中的哪个条件（不指定则: simple模式按trial轮转, llm模式用第一个）')
    
    # 输出配置
    parser.add_argument('--verbose', action='store_true',
                      help='显示详细输出')
    parser.add_argument('--output-dir', default='experiment_results',
                      help='输出目录')
    
    args = parser.parse_args()
    
    if args.mode == 'simple':
        # 模拟模式：不调用实际 LLM
        run_simple_experiment(
            query=args.query,
            num_trials=args.trials,
            perception_mode=args.perception,
            condition_file=args.condition_file,
        )
    elif args.mode == 'llm':
        # 真实 LLM 模式
        # 获取 API key
        api_key = args.api_key
        if not api_key:
            if args.llm == 'deepseek':
                api_key = os.getenv("DEEPSEEK_API_KEY")
            elif args.llm == 'openai':
                api_key = os.getenv("OPENAI_API_KEY")
            elif args.llm == 'qwen':
                api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        
        if not api_key:
            print(f"错误: 需要 {args.llm.upper()} API key")
            print(f"使用 --api-key 参数或设置 {args.llm.upper()}_API_KEY 环境变量")
            sys.exit(1)
        
        # 确定模型名称
        model = args.model
        if not model:
            if args.llm == 'deepseek':
                model = 'deepseek-chat'
            elif args.llm == 'openai':
                model = 'gpt-4o-mini'
            elif args.llm == 'qwen':
                # 根据感知模式选择模型
                if args.perception == 'visual':
                    model = 'qwen-vl-plus'  # Visual 模式使用 VL 模型
                else:
                    model = 'qwen-turbo'    # Verbal 模式使用普通模型
        
        run_with_real_llm(
            llm_backend=args.llm,
            model=model,
            api_key=api_key,
            query=args.query,
            perception_mode=args.perception,
            marketplace_mode=args.marketplace,
            datasets_dir=args.datasets_dir,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            limit=args.limit,
            verbose=args.verbose,
            condition_file=args.condition_file,
            condition_name=args.condition_name,
        )


if __name__ == "__main__":
    main()
