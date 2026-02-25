"""
完整的网页渲染 + 截图工作流演示

展示 ACES-v2 的截图功能：
1. Environment 负责网页渲染
2. Tool 负责截图
3. Perception Filter 将截图传给 Agent

架构设计验证：
- 网页渲染属于 Environment (WebRendererMarketplace)
- 截图属于 Tool (ScreenshotTool/PlaywrightScreenshotTool)
- 截图结果经过 Perception Filter (VisualPerception) 给到 Agent
"""

import logging
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from aces.environments import WebRendererMarketplace
from aces.tools import ScreenshotTool, PlaywrightScreenshotTool
from aces.perception import VisualPerception
from aces.agents import ComposableAgent
from aces.llm_backends import DeepSeekBackend
from aces.config.settings import resolve_datasets_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_basic_screenshot():
    """演示1: 基础截图流程"""
    print("\n" + "="*60)
    print("演示1: 基础截图流程")
    print("="*60)
    
    # 步骤1: 创建 Web Renderer Environment（负责网页渲染）
    print("\n[1] 初始化 Environment - 负责网页渲染...")
    marketplace = WebRendererMarketplace()
    marketplace.initialize({
        "datasets_dir": str(resolve_datasets_dir()),
        "server_port": 5000,
        "auto_start_server": True
    })
    
    # 步骤2: 搜索商品（更新网页内容）
    print("\n[2] 搜索商品 - Environment 更新网页内容...")
    results = marketplace.search_products("mousepad", limit=8)
    print(f"   找到 {len(results.products)} 个商品")
    print(f"   网页 URL: {marketplace.server_url}/search?q=mousepad")
    
    # 步骤3: 创建截图工具（负责截图）
    print("\n[3] 初始化 Tool - 负责截图...")
    screenshot_tool = PlaywrightScreenshotTool(
        server_url=marketplace.server_url,
        browser="chromium",
        headless=True
    )
    
    # 步骤4: 使用工具截图
    print("\n[4] 执行截图 Tool...")
    screenshot_result = screenshot_tool.execute({
        "query": "mousepad",
        "wait_time": 1.0
    })
    
    print(f"   截图大小: {screenshot_result['size_bytes']} bytes")
    print(f"   格式: {screenshot_result['format']}")
    
    # 步骤5: 保存截图到文件
    print("\n[5] 保存截图...")
    screenshot_path = Path("screenshot_mousepad.png")
    with open(screenshot_path, "wb") as f:
        f.write(screenshot_result["screenshot_bytes"])
    print(f"   已保存到: {screenshot_path}")
    
    # 清理
    screenshot_tool.close()
    marketplace.close()
    
    print("\n✓ 基础截图流程完成")


def demo_with_perception_filter():
    """演示2: 截图 + Perception Filter"""
    print("\n" + "="*60)
    print("演示2: 截图 + Perception Filter - 传给 Agent")
    print("="*60)
    
    # 步骤1: Environment（网页渲染）
    print("\n[1] 初始化 Environment...")
    marketplace = WebRendererMarketplace()
    marketplace.initialize({
        "datasets_dir": str(resolve_datasets_dir()),
        "server_port": 5001,  # 不同端口避免冲突
        "auto_start_server": True
    })
    
    # 步骤2: 搜索商品
    print("\n[2] 搜索商品...")
    results = marketplace.search_products("toothpaste", limit=8)
    print(f"   找到 {len(results.products)} 个商品")
    
    # 步骤3: Tool 截图
    print("\n[3] 工具截图...")
    screenshot_tool = PlaywrightScreenshotTool(
        server_url=marketplace.server_url
    )
    screenshot_result = screenshot_tool.execute({
        "query": "toothpaste",
        "wait_time": 1.0
    })
    
    # 步骤4: Perception Filter 处理截图
    print("\n[4] Perception Filter 处理截图...")
    perception = VisualPerception()
    observation = perception.encode(screenshot_result["screenshot_bytes"])
    
    print(f"   观察模态: {observation.modality}")
    print(f"   数据格式: {type(observation.data)}")
    print(f"   时间戳: {observation.timestamp}")
    
    # 步骤5: 观察可以传给 Agent
    print("\n[5] 观察传给 Agent（模拟）...")
    print("   Agent 会收到:")
    print(f"   - modality: {observation.modality}")
    print(f"   - 图片数据: base64 编码的 PNG")
    print(f"   - 元数据: {observation.metadata}")
    
    # 清理
    screenshot_tool.close()
    marketplace.close()
    
    print("\n✓ Perception Filter 演示完成")


def demo_multiple_queries():
    """演示3: 多查询截图（测试服务器稳定性）"""
    print("\n" + "="*60)
    print("演示3: 多查询截图")
    print("="*60)
    
    # Environment
    marketplace = WebRendererMarketplace()
    marketplace.initialize({
        "datasets_dir": str(resolve_datasets_dir()),
        "server_port": 5002,
        "auto_start_server": True
    })
    
    # Tool
    screenshot_tool = PlaywrightScreenshotTool(
        server_url=marketplace.server_url
    )
    
    # 多个查询
    queries = ["mousepad", "toothpaste", "usb_cable", "fitness_watch"]
    
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] 处理查询: {query}")
        
        # 搜索
        results = marketplace.search_products(query, limit=8)
        print(f"   搜索结果: {len(results.products)} 个商品")
        
        # 截图
        screenshot_result = screenshot_tool.execute({
            "query": query,
            "wait_time": 0.5
        })
        
        # 保存
        screenshot_path = Path(f"screenshot_{query}.png")
        with open(screenshot_path, "wb") as f:
            f.write(screenshot_result["screenshot_bytes"])
        print(f"   已保存: {screenshot_path} ({screenshot_result['size_bytes']} bytes)")
    
    # 清理
    screenshot_tool.close()
    marketplace.close()
    
    print("\n✓ 多查询截图完成")


def demo_full_agent_workflow():
    """演示4: 完整的 Agent 工作流（需要 API key）"""
    print("\n" + "="*60)
    print("演示4: 完整 Agent 工作流（带截图）")
    print("="*60)
    
    import os
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("\n⚠️ 需要设置 DEEPSEEK_API_KEY 环境变量")
        print("   跳过此演示")
        return
    
    # Environment
    print("\n[1] 初始化 Environment...")
    marketplace = WebRendererMarketplace()
    marketplace.initialize({
        "datasets_dir": str(resolve_datasets_dir()),
        "server_port": 5003,
        "auto_start_server": True
    })
    
    # Tools
    print("\n[2] 准备 Tools...")
    from aces.tools import SearchTool, AddToCartTool
    
    screenshot_tool = PlaywrightScreenshotTool(
        server_url=marketplace.server_url
    )
    
    tools = [
        SearchTool(marketplace),
        AddToCartTool(marketplace),
        screenshot_tool,  # 添加截图工具
    ]
    
    # Perception Filter
    print("\n[3] 配置 Perception Filter...")
    perception = VisualPerception()
    
    # LLM Backend
    print("\n[4] 初始化 LLM Backend...")
    llm = DeepSeekBackend(api_key=api_key, model="deepseek-chat")
    
    # Agent
    print("\n[5] 创建 Agent...")
    agent = ComposableAgent(
        llm=llm,
        perception=perception,
        tools=tools,
        system_prompt="""你是一个购物助手。你可以使用以下工具：
        - search_products: 搜索商品
        - capture_screenshot_playwright: 截取商品页面截图
        - add_to_cart: 添加商品到购物车
        
        请帮助用户找到合适的商品。"""
    )
    
    print("\n[6] Agent 运行...")
    print("   (这里可以实现完整的对话循环)")
    print("   Agent 可以:")
    print("   - 搜索商品")
    print("   - 截图查看页面")
    print("   - 根据视觉信息做决策")
    print("   - 添加商品到购物车")
    
    # 清理
    screenshot_tool.close()
    marketplace.close()
    
    print("\n✓ Agent 工作流演示完成")


def main():
    """运行所有演示"""
    print("\n" + "="*60)
    print("ACES-v2 网页渲染 + 截图功能演示")
    print("="*60)
    print("\n架构设计:")
    print("  - Environment: 负责网页渲染 (WebRendererMarketplace)")
    print("  - Tool: 负责截图 (ScreenshotTool)")
    print("  - Perception Filter: 处理截图 (VisualPerception)")
    print("  - Agent: 接收视觉观察并决策")
    
    try:
        # 演示1: 基础截图
        demo_basic_screenshot()
        
        # 演示2: 截图 + Perception Filter
        demo_with_perception_filter()
        
        # 演示3: 多查询截图
        demo_multiple_queries()
        
        # 演示4: 完整 Agent 工作流
        demo_full_agent_workflow()
        
        print("\n" + "="*60)
        print("所有演示完成！")
        print("="*60)
        print("\n生成的截图文件:")
        for f in Path(".").glob("screenshot_*.png"):
            print(f"  - {f}")
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        logger.error(f"错误: {e}", exc_info=True)


if __name__ == "__main__":
    main()
