"""
Complete Workflow Example

Demonstrates the entire ACES v2 system in action:
1. Load configuration
2. Create marketplace (offline/online)
3. Initialize agent
4. Run shopping simulation
5. Analyze results
"""

import logging
import os
from pathlib import Path

from aces.config.settings import resolve_datasets_dir
from aces.llm_backends.factory import LLMBackendFactory
from aces.perception.factory import PerceptionFactory
from aces.tools.factory import ToolFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_complete_workflow():
    """
    Complete workflow showing all components working together.
    
    This demonstrates:
    - Marketplace factory (offline mode)
    - Agent composition (LLM + Perception + Tools)
    - Shopping simulation loop
    - Result collection
    """
    
    print("\n" + "="*80)
    print("ACES v2: Complete Workflow Demonstration")
    print("="*80 + "\n")
    
    # ========================================================================
    # Step 1: Create Marketplace
    # ========================================================================
    print("Step 1: Creating Marketplace Provider...")
    
    from aces.environments import MarketplaceFactory, MarketplaceAdapter
    
    marketplace_config = {
        "mode": "offline",  # Use controlled sandbox
        "offline": {
            "datasets_dir": str(resolve_datasets_dir()),
            "screenshots_dir": "screenshots",
        }
    }
    
    marketplace_provider = MarketplaceFactory.create(marketplace_config)
    marketplace = MarketplaceAdapter(marketplace_provider)
    
    print(f"✓ Created marketplace in {marketplace.mode.value} mode\n")
    
    # ========================================================================
    # Step 2: Initialize Tools
    # ========================================================================
    print("Step 2: Initializing MCP Tools...")
    
    tools = ToolFactory.create_many(
        ["search_products", "add_to_cart", "view_product_details"],
        marketplace_api=marketplace,
    )
    
    print(f"✓ Loaded {len(tools)} tools: {[t.get_schema().name for t in tools]}\n")
    
    # ========================================================================
    # Step 3: Create Agent Components
    # ========================================================================
    print("Step 3: Assembling Agent (LLM + Perception + Tools)...")
    
    # LLM Backend
    llm = LLMBackendFactory.create({
        "backend": "deepseek",
        "model": "deepseek-chat",
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "temperature": 1.0,
    })
    
    # Perception Mode (choose one)
    perception = PerceptionFactory.create({"mode": "verbal", "format_style": "structured"})
    # Note: Use VisualPerception for image-based mode
    
    # Assemble Agent
    from aces.agents.base_agent import ComposableAgent
    
    agent = ComposableAgent(
        llm=llm,
        perception=perception,
        tools=tools,
        system_prompt=(
            "You are a helpful shopping assistant. "
            "Your goal is to help the user find the best product for their needs. "
            "Always explain your reasoning before making a selection."
        ),
    )
    
    print(f"✓ Agent assembled:")
    print(f"  - LLM: {llm.model_name}")
    print(f"  - Perception: {perception.get_modality()}")
    print(f"  - Tools: {len(tools)}\n")
    
    # ========================================================================
    # Step 4: Run Shopping Simulation
    # ========================================================================
    print("Step 4: Running Shopping Simulation...")
    print("-" * 80)
    
    # Initialize marketplace
    initial_state = marketplace.reset()
    
    # Search for products
    print("\n[User] I need to buy a mousepad for gaming.")
    search_results = marketplace.search_products("mousepad", limit=5)
    
    print(f"[Marketplace] Found {len(search_results.products)} products:")
    for i, p in enumerate(search_results.products, 1):
        print(f"  {i}. {p.title[:60]}... - ${p.price:.2f}")
    
    # Get current page state
    page_state = marketplace.get_page_state()
    
    # Agent perceives the state
    print("\n[Agent] Processing observation...")
    observation = perception.encode(page_state.products)
    print(f"[Agent] Perceived {len(page_state.products)} products in {observation.modality} mode")
    
    # NOTE: Full agent.act() would call LLM here
    # For demo, we'll simulate the decision
    print("\n[Agent] Analyzing products...")
    print("[Agent] Considering: price, rating, features...")
    
    # Simulate agent decision (would normally come from LLM)
    selected_product = search_results.products[0]  # For demo
    
    print(f"\n[Agent] Decision: Selecting '{selected_product.title[:60]}'")
    print(f"[Agent] Reason: Good balance of price (${selected_product.price}) and rating ({selected_product.rating}/5)")
    
    # Add to cart
    cart_result = marketplace.add_to_cart(selected_product.id)
    print(f"\n[Marketplace] ✓ Added to cart. Total: ${cart_result['cart_total']:.2f}")
    
    print("-" * 80)
    print("✓ Simulation complete\n")
    
    # ========================================================================
    # Step 5: Collect Results
    # ========================================================================
    print("Step 5: Collecting Results...")
    
    agent_state = agent.get_state()
    
    results = {
        "experiment_name": "demo_workflow",
        "agent": {
            "llm": llm.model_name,
            "perception": perception.get_modality(),
            "steps": agent_state.step_count,
        },
        "marketplace": {
            "mode": marketplace.mode.value,
            "query": search_results.query,
            "products_shown": len(search_results.products),
        },
        "decision": {
            "selected_product_id": selected_product.id,
            "selected_product_title": selected_product.title,
            "selected_product_price": selected_product.price,
            "selected_product_rank": selected_product.position + 1,
        }
    }
    
    print("Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    # ========================================================================
    # Cleanup
    # ========================================================================
    marketplace.close()
    
    print("\n" + "="*80)
    print("Complete Workflow Finished Successfully!")
    print("="*80 + "\n")


def comparison_workflow():
    """
    Compare offline vs online modes.
    
    Demonstrates the "Twin Worlds" architecture.
    """
    print("\n" + "="*80)
    print("COMPARISON: Offline vs Online Modes")
    print("="*80 + "\n")
    
    from aces.environments import MarketplaceFactory
    
    query = "laptop"
    
    # ========================================================================
    # Test 1: Offline Mode
    # ========================================================================
    print("Test 1: Offline Marketplace")
    print("-" * 40)
    
    offline_config = {
        "mode": "offline",
        "offline": {"datasets_dir": str(resolve_datasets_dir())}
    }
    
    offline_marketplace = MarketplaceFactory.create(offline_config)
    offline_results = offline_marketplace.search_products(query, limit=3)
    
    print(f"Query: {query}")
    print(f"Mode: {offline_marketplace.get_mode().value}")
    print(f"Products found: {len(offline_results.products)}")
    print(f"Source: Local datasets\n")
    
    offline_marketplace.close()
    
    # ========================================================================
    # Test 2: Online Mode (commented out - requires Playwright)
    # ========================================================================
    print("Test 2: Online Marketplace")
    print("-" * 40)
    print("(Skipped - requires Playwright installation)")
    print("To enable: pip install playwright && playwright install\n")
    
    # online_config = {
    #     "mode": "online",
    #     "online": {"platform": "amazon"}
    # }
    # online_marketplace = MarketplaceFactory.create(online_config)
    # online_results = online_marketplace.search_products(query, limit=3)
    
    # ========================================================================
    # Comparison
    # ========================================================================
    print("Key Insight:")
    print("  Both modes use IDENTICAL interface:")
    print("  - marketplace.search_products()")
    print("  - marketplace.get_product_details()")
    print("  - marketplace.add_to_cart()")
    print("\n  Agent code doesn't need to change!")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Run complete workflow
    run_complete_workflow()
    
    # Run comparison
    comparison_workflow()
    
    print("\n✓ All examples completed successfully!")
    print("\nNext steps:")
    print("  1. Review docs/ARCHITECTURE.md for design details")
    print("  2. Check configs/ for example configurations")
    print("  3. Run: python -m aces.experiments.run <config.yaml>")
    print()
