"""
Verbal Perception Mode

Converts environment state into textual/structured observations.
"""

from typing import Any, Dict
import json
import time
import logging

from aces.core.protocols import PerceptionMode, Observation


logger = logging.getLogger(__name__)


class VerbalPerception(PerceptionMode):
    """
    Verbal perception mode: Agent sees text/JSON product specifications.
    
    This is the "verbal slot implementation" - it converts environment
    state into textual observations.
    """
    
    def __init__(self, format_style: str = "structured"):
        """
        Initialize verbal perception.
        
        Args:
            format_style: How to format text
                - "structured": JSON format
                - "natural": Natural language description
                - "markdown": Markdown tables
        """
        self.format_style = format_style
        
        logger.info(f"Initialized verbal perception (style={format_style})")
    
    def encode(self, raw_state: Any) -> Observation:
        """
        Convert raw state to verbal observation.
        
        Args:
            raw_state: Could be:
                - Dict (product data)
                - List[Dict] (multiple products)
                - HTML (to be parsed)
                
        Returns:
            Observation with text/structured data
        """
        # Handle different input types
        if isinstance(raw_state, dict):
            text_data = self._format_product(raw_state)
        elif isinstance(raw_state, list):
            text_data = self._format_products(raw_state)
        elif isinstance(raw_state, str):
            # Assume it's HTML or already formatted text
            if "<html" in raw_state.lower():
                text_data = self._extract_from_html(raw_state)
            else:
                text_data = raw_state
        else:
            text_data = str(raw_state)
        
        return Observation(
            data=text_data,
            modality="verbal",
            timestamp=time.time(),
            metadata={"format": self.format_style}
        )
    
    def get_modality(self) -> str:
        """Return modality type."""
        return "verbal"
    
    def validate_observation(self, obs: Observation) -> bool:
        """Check if observation is in verbal format."""
        return obs.modality == "verbal"
    
    # ========================================================================
    # Private Helper Methods
    # ========================================================================
    
    def _format_product(self, product: Dict) -> str:
        """Format a single product."""
        if self.format_style == "structured":
            return json.dumps(product, indent=2)
        elif self.format_style == "natural":
            return self._product_to_natural_language(product)
        elif self.format_style == "markdown":
            return self._product_to_markdown(product)
        else:
            return str(product)
    
    def _format_products(self, products: list) -> str:
        """Format multiple products."""
        if self.format_style == "structured":
            # Convert Product objects to dicts if needed
            products_data = []
            for p in products:
                if hasattr(p, '__dict__'):
                    # Product object - convert to dict
                    d = {
                        "id": p.id,
                        "title": p.title,
                        "price": p.price,
                        "rating": p.rating,
                        "rating_count": p.rating_count,
                        "sponsored": p.sponsored,
                        "best_seller": p.best_seller,
                        "overall_pick": getattr(p, "overall_pick", False),
                        "low_stock": getattr(p, "low_stock", False),
                        "position": p.position,
                    }
                    raw = getattr(p, "raw_data", None)
                    if raw:
                        if "original_price" in raw:
                            d["original_price"] = raw["original_price"]
                        if "custom_badges" in raw:
                            d["custom_badges"] = raw["custom_badges"]
                    products_data.append(d)
                else:
                    # Already a dict
                    products_data.append(p)
            
            return json.dumps(products_data, indent=2)
        elif self.format_style == "natural":
            descriptions = [
                f"Product {i+1}:\n{self._product_to_natural_language(p)}"
                for i, p in enumerate(products)
            ]
            return "\n\n".join(descriptions)
        elif self.format_style == "markdown":
            return self._products_to_markdown_table(products)
        else:
            return str(products)
    
    def _product_to_natural_language(self, product) -> str:
        """Convert product to natural language description."""
        parts = []
        
        # Handle both dict and Product object
        if hasattr(product, '__dict__'):
            # Product object
            parts.append(f"Product ID: {product.id}")
            parts.append(f"Title: {product.title}")
            raw = getattr(product, "raw_data", None) or {}
            if "original_price" in raw:
                parts.append(f"Price: ${product.price:.2f} (was ${raw['original_price']:.2f})")
            else:
                parts.append(f"Price: ${product.price:.2f}")
            if product.rating:
                parts.append(f"Rating: {product.rating}/5")
            if product.rating_count:
                parts.append(f"Reviews: {product.rating_count:,}")
            
            # Add tags
            tags = []
            if product.sponsored:
                tags.append("Sponsored")
            if product.best_seller:
                tags.append("Best Seller")
            if product.low_stock:
                tags.append("Low Stock")
            if product.overall_pick:
                tags.append("Overall Pick")
            if "custom_badges" in raw:
                tags.extend(raw["custom_badges"])
            
            if tags:
                parts.append(f"Tags: {', '.join(tags)}")
        else:
            # Dict
            if "title" in product:
                parts.append(f"Title: {product['title']}")
            if "price" in product:
                parts.append(f"Price: ${product['price']}")
            if "rating" in product:
                parts.append(f"Rating: {product['rating']}/5")
            if "rating_count" in product:
                parts.append(f"Reviews: {product['rating_count']}")
            
            # Add tags
            tags = []
            if product.get("sponsored"):
                tags.append("Sponsored")
            if product.get("best_seller"):
                tags.append("Best Seller")
            if product.get("low_stock"):
                tags.append("Low Stock")
            
            if tags:
                parts.append(f"Tags: {', '.join(tags)}")
        
        return "\n".join(parts)
    
    def _product_to_markdown(self, product: Dict) -> str:
        """Convert product to markdown format."""
        lines = [f"### {product.get('title', 'Unknown Product')}"]
        
        if "price" in product:
            lines.append(f"**Price:** ${product['price']}")
        if "rating" in product:
            stars = "⭐" * int(product['rating'])
            lines.append(f"**Rating:** {stars} ({product['rating']}/5)")
        
        return "\n".join(lines)
    
    def _products_to_markdown_table(self, products: list) -> str:
        """Convert products to markdown table."""
        lines = [
            "| # | Title | Price | Rating | Tags |",
            "|---|-------|-------|--------|------|",
        ]
        
        for i, p in enumerate(products, 1):
            title = p.get("title", "Unknown")[:50]
            price = f"${p.get('price', 0)}"
            rating = f"{p.get('rating', 0)}/5"
            
            tags = []
            if p.get("sponsored"):
                tags.append("Sponsored")
            if p.get("best_seller"):
                tags.append("Best Seller")
            
            lines.append(f"| {i} | {title} | {price} | {rating} | {', '.join(tags)} |")
        
        return "\n".join(lines)
    
    def _extract_from_html(self, html: str) -> str:
        """Extract product info from HTML (simplified)."""
        # This would use BeautifulSoup in practice
        # For now, just return simplified version
        logger.warning("HTML parsing not fully implemented")
        return html[:500]  # Truncate for now
