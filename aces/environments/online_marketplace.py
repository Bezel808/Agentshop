"""
Online Marketplace Provider

Implements a real-world adapter using headless browser (Playwright).
Treats live websites as read-only data sources.
"""

import logging
from typing import Any, Dict, List, Optional
import time
import base64

from aces.environments.protocols import (
    MarketplaceProvider,
    MarketplaceMode,
    Product,
    SearchResult,
    PageState,
)


logger = logging.getLogger(__name__)


class OnlineMarketplace(MarketplaceProvider):
    """
    Online marketplace using live web scraping.
    
    Features:
    - Real-time data from live websites
    - Playwright-based browser automation
    - Safe read-only mode (no actual purchases)
    - Screenshot capture
    
    Supported platforms:
    - Amazon
    - Taobao
    - (Extensible to others)
    """
    
    def __init__(self):
        """Initialize online marketplace."""
        self.platform: Optional[str] = None
        self.browser = None
        self.page = None
        self.current_query: Optional[str] = None
        self.current_products: List[Product] = []
        
        logger.info("Initialized OnlineMarketplace")
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize with configuration.
        
        Config format:
            platform: "amazon" | "taobao"
            headless: bool (default True)
            timeout: int (default 30000ms)
            proxy: Optional proxy settings
        """
        self.platform = config.get("platform", "amazon")
        headless = config.get("headless", True)
        timeout = config.get("timeout", 30000)
        
        # Lazy import to avoid dependency if not used
        try:
            from playwright.sync_api import sync_playwright
            
            self._playwright = sync_playwright().start()
            self.browser = self._playwright.chromium.launch(headless=headless)
            self.page = self.browser.new_page()
            self.page.set_default_timeout(timeout)
            
            logger.info(
                f"Initialized online marketplace: "
                f"platform={self.platform}, headless={headless}"
            )
        except ImportError:
            raise ImportError(
                "Online marketplace requires 'playwright' package. "
                "Install with: pip install playwright && playwright install"
            )
    
    def search_products(
        self,
        query: str,
        sort_by: str = "relevance",
        limit: int = 10,
        **kwargs
    ) -> SearchResult:
        """
        Search products on live website.
        
        This is the "MCP Tool" that hides the complexity of
        DOM parsing and web scraping.
        """
        self.current_query = query
        
        # Navigate to search page
        if self.platform == "amazon":
            products = self._search_amazon(query, sort_by, limit)
        elif self.platform == "taobao":
            products = self._search_taobao(query, sort_by, limit)
        else:
            raise ValueError(f"Unsupported platform: {self.platform}")
        
        self.current_products = products
        
        logger.info(
            f"Online search '{query}' returned {len(products)} products "
            f"from {self.platform}"
        )
        
        return SearchResult(
            query=query,
            products=products,
            total_count=len(products),  # Approximate
            page=1,
            metadata={
                "mode": "online",
                "platform": self.platform,
                "timestamp": time.time(),
            }
        )
    
    def get_product_details(self, product_id: str) -> Product:
        """
        Get product details by navigating to product page.
        
        Note: This makes a real HTTP request.
        """
        # Find product in current results
        for product in self.current_products:
            if product.id == product_id:
                # Could navigate to detail page for more info
                # For now, return cached data
                return product
        
        raise ValueError(f"Product {product_id} not found in current results")
    
    def get_page_state(self) -> PageState:
        """
        Get current page state with screenshot.
        
        Captures real-time screenshot from browser.
        """
        # Capture screenshot
        screenshot = None
        if self.page:
            screenshot_bytes = self.page.screenshot(full_page=True)
            screenshot = screenshot_bytes
        
        # Get HTML (optional, for debugging)
        html = None
        if self.page:
            html = self.page.content()
        
        return PageState(
            products=self.current_products,
            query=self.current_query,
            html=html,
            screenshot=screenshot,
            metadata={
                "mode": "online",
                "platform": self.platform,
                "url": self.page.url if self.page else None,
            }
        )
    
    def add_to_cart(self, product_id: str, quantity: int = 1) -> Dict[str, Any]:
        """
        Add to cart (READ-ONLY MODE - doesn't actually add).
        
        This is intentionally limited to prevent accidental purchases.
        """
        logger.warning(
            "Online marketplace is in READ-ONLY mode. "
            "add_to_cart is simulated, not executed."
        )
        
        product = self.get_product_details(product_id)
        
        return {
            "success": True,
            "simulated": True,
            "product_id": product_id,
            "product_title": product.title,
            "quantity": quantity,
            "warning": "Read-only mode: No actual purchase made",
        }
    
    def reset(self) -> PageState:
        """Reset browser state."""
        self.current_query = None
        self.current_products = []
        
        # Navigate to home page
        if self.page:
            if self.platform == "amazon":
                self.page.goto("https://www.amazon.com")
            elif self.platform == "taobao":
                self.page.goto("https://www.taobao.com")
        
        logger.info("Reset online marketplace")
        
        return PageState(products=[], metadata={"mode": "online"})
    
    def get_mode(self) -> MarketplaceMode:
        """Return marketplace mode."""
        return MarketplaceMode.ONLINE
    
    def close(self) -> None:
        """Cleanup browser resources."""
        if self.browser:
            self.browser.close()
        if hasattr(self, '_playwright'):
            self._playwright.stop()
        
        logger.info("Closed online marketplace (browser cleaned up)")
    
    # ========================================================================
    # Platform-Specific Scraping Logic
    # ========================================================================
    
    def _search_amazon(
        self,
        query: str,
        sort_by: str,
        limit: int
    ) -> List[Product]:
        """
        Search Amazon and parse results.
        
        This encapsulates all the DOM parsing complexity.
        """
        # Navigate to Amazon search
        search_url = f"https://www.amazon.com/s?k={query.replace(' ', '+')}"
        self.page.goto(search_url)
        
        # Wait for results to load
        self.page.wait_for_selector('[data-component-type="s-search-result"]')
        
        # Parse product cards
        products = []
        result_elements = self.page.query_selector_all(
            '[data-component-type="s-search-result"]'
        )
        
        for index, element in enumerate(result_elements[:limit]):
            try:
                product = self._parse_amazon_product(element, index)
                if product:
                    products.append(product)
            except Exception as e:
                logger.warning(f"Failed to parse Amazon product: {e}")
                continue
        
        return products
    
    def _parse_amazon_product(self, element, index: int) -> Optional[Product]:
        """Parse a single Amazon product element."""
        try:
            # Extract data using selectors
            # (Simplified - actual implementation would be more robust)
            
            title_el = element.query_selector('h2 a span')
            title = title_el.inner_text() if title_el else "Unknown"
            
            price_el = element.query_selector('.a-price-whole')
            price = 0.0
            if price_el:
                price_text = price_el.inner_text().replace(',', '').replace('$', '')
                price = float(price_text)
            
            rating_el = element.query_selector('.a-icon-star-small')
            rating = None
            if rating_el:
                rating_text = rating_el.get_attribute('aria-label')
                if rating_text:
                    rating = float(rating_text.split()[0])
            
            rating_count_el = element.query_selector('[aria-label*="ratings"]')
            rating_count = None
            if rating_count_el:
                count_text = rating_count_el.inner_text().replace(',', '')
                try:
                    rating_count = int(count_text)
                except:
                    pass
            
            image_el = element.query_selector('img')
            image_url = image_el.get_attribute('src') if image_el else None
            
            # Product ID (ASIN)
            asin = element.get_attribute('data-asin') or f"amazon_{index}"
            
            # Check for sponsored
            sponsored = bool(element.query_selector('[aria-label="Sponsored"]'))
            
            return Product(
                id=asin,
                title=title,
                price=price,
                rating=rating,
                rating_count=rating_count,
                image_url=image_url,
                sponsored=sponsored,
                position=index,
                source=f"online:amazon",
                raw_data={"platform": "amazon"},
            )
        except Exception as e:
            logger.error(f"Error parsing Amazon product: {e}")
            return None
    
    def _search_taobao(
        self,
        query: str,
        sort_by: str,
        limit: int
    ) -> List[Product]:
        """
        Search Taobao and parse results.
        
        Note: Taobao has anti-scraping measures, so this is simplified.
        """
        logger.warning("Taobao scraping is simplified in this example")
        
        # Would implement similar to Amazon but with Taobao-specific selectors
        # For now, return empty list
        return []
