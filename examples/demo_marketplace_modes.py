"""
Demo: Using Marketplace Router

Shows how to switch between offline and online modes
using the same interface.
"""

import logging
from aces.config.settings import resolve_datasets_dir
from aces.environments.router import MarketplaceFactory, MarketplaceAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_offline_mode():
    """Demo: Using offline marketplace."""
    print("\n" + "="*70)
    print("DEMO: Offline Marketplace (ACES Datasets)")
    print("="*70 + "\n")
    
    # Configuration for offline mode
    config = {
        "mode": "offline",
        "offline": {
            "datasets_dir": str(resolve_datasets_dir()),
            "screenshots_dir": "screenshots",
        }
    }
    
    # Create marketplace
    marketplace = MarketplaceFactory.create(config)
    adapter = MarketplaceAdapter(marketplace)
    
    print(f"Mode: {adapter.mode.value}")
    print(f"Is offline: {adapter.is_offline()}\n")
    
    # Search products
    results = adapter.search_products("mousepad", limit=5)
    print(f"Search results for '{results.query}':")
    print(f"Total products: {results.total_count}\n")
    
    for i, product in enumerate(results.products, 1):
        print(f"{i}. {product.title}")
        print(f"   Price: ${product.price:.2f}")
        print(f"   Rating: {product.rating}/5 ({product.rating_count} reviews)")
        print(f"   Tags: ", end="")
        tags = []
        if product.sponsored:
            tags.append("Sponsored")
        if product.best_seller:
            tags.append("Best Seller")
        print(", ".join(tags) if tags else "None")
        print()
    
    # Get page state
    page_state = adapter.get_page_state()
    print(f"Page state: {len(page_state.products)} products")
    print(f"Has screenshot: {page_state.screenshot is not None}")
    
    # Add to cart
    if results.products:
        first_product = results.products[0]
        cart_result = adapter.add_to_cart(first_product.id)
        print(f"\nAdded to cart: {cart_result['success']}")
        print(f"Cart total: ${cart_result['cart_total']:.2f}")
    
    # Cleanup
    adapter.close()
    print("\n✓ Offline demo complete\n")


def demo_online_mode():
    """Demo: Using online marketplace (requires Playwright)."""
    print("\n" + "="*70)
    print("DEMO: Online Marketplace (Live Scraping)")
    print("="*70 + "\n")
    
    # Configuration for online mode
    config = {
        "mode": "online",
        "online": {
            "platform": "amazon",
            "headless": True,
        }
    }
    
    try:
        # Create marketplace
        marketplace = MarketplaceFactory.create(config)
        adapter = MarketplaceAdapter(marketplace)
        
        print(f"Mode: {adapter.mode.value}")
        print(f"Is online: {adapter.is_online()}\n")
        
        # Search products (makes real HTTP request!)
        print("Searching Amazon for 'wireless mouse' (this will take a few seconds)...")
        results = adapter.search_products("wireless mouse", limit=3)
        
        print(f"\nSearch results for '{results.query}':")
        print(f"Total products: {results.total_count}\n")
        
        for i, product in enumerate(results.products, 1):
            print(f"{i}. {product.title[:60]}...")
            print(f"   Price: ${product.price:.2f}")
            print(f"   Source: {product.source}")
            print()
        
        # Get page state with screenshot
        page_state = adapter.get_page_state()
        print(f"Page state: {len(page_state.products)} products")
        print(f"Has screenshot: {page_state.screenshot is not None}")
        print(f"Screenshot size: {len(page_state.screenshot)} bytes" if page_state.screenshot else "N/A")
        
        # Cleanup
        adapter.close()
        print("\n✓ Online demo complete\n")
        
    except ImportError as e:
        print(f"⚠ Skipping online demo: {e}")
        print("Install Playwright: pip install playwright && playwright install\n")
    except Exception as e:
        print(f"⚠ Online demo failed: {e}\n")


def demo_env_mode():
    """Demo: Using environment variable to select mode."""
    print("\n" + "="*70)
    print("DEMO: Environment Variable Mode Selection")
    print("="*70 + "\n")
    
    import os
    
    # Set environment variable
    os.environ["ENV_MODE"] = "offline"
    os.environ["ACES_DATASETS_DIR"] = str(resolve_datasets_dir())
    
    # Create from environment
    marketplace = MarketplaceFactory.create_from_env()
    adapter = MarketplaceAdapter(marketplace)
    
    print(f"Mode (from ENV_MODE): {adapter.mode.value}")
    print(f"Datasets dir (from ACES_DATASETS_DIR): {os.getenv('ACES_DATASETS_DIR')}\n")
    
    # Quick test
    results = adapter.search_products("laptop", limit=3)
    print(f"Found {len(results.products)} products for '{results.query}'")
    
    adapter.close()
    print("\n✓ Environment variable demo complete\n")


def demo_same_interface():
    """Demo: Both modes use the same interface."""
    print("\n" + "="*70)
    print("DEMO: Unified Interface for Both Modes")
    print("="*70 + "\n")
    
    print("Key insight: Agent code doesn't need to know which mode is active!")
    print("The Tools always call the same methods:\n")
    
    print("  adapter.search_products(query) # Works in both modes")
    print("  adapter.get_product_details()  # Works in both modes")
    print("  adapter.add_to_cart()          # Works in both modes")
    print("  adapter.get_page_state()       # Works in both modes")
    
    print("\nThis abstraction enables:")
    print("  1. Testing in offline mode (fast, controlled)")
    print("  2. Validating in online mode (real-world)")
    print("  3. Same agent code for both!")
    print()


if __name__ == "__main__":
    # Run all demos
    demo_offline_mode()
    demo_env_mode()
    demo_same_interface()
    
    # Uncomment to test online mode (requires Playwright)
    # demo_online_mode()
    
    print("="*70)
    print("All demos complete!")
    print("="*70)
