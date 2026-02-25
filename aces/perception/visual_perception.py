"""
Visual Perception Mode

Converts environment state into visual observations (screenshots/images).
"""

from typing import Any
import base64
import logging
import time

from aces.core.protocols import PerceptionMode, Observation


logger = logging.getLogger(__name__)


class VisualPerception(PerceptionMode):
    """
    Visual perception mode: Agent sees screenshots/images.
    
    This is the "visual slot implementation" - it converts environment
    state into image observations.
    """
    
    def __init__(self, image_format: str = "png", detail_level: str = "high"):
        """
        Initialize visual perception.
        
        Args:
            image_format: Image format (png, jpeg, etc)
            detail_level: OpenAI vision detail level ("low", "high", "auto")
        """
        self.image_format = image_format
        self.detail_level = detail_level
        
        logger.info(f"Initialized visual perception (format={image_format})")
    
    def encode(self, raw_state: Any) -> Observation:
        """
        Convert raw state to visual observation.
        
        Args:
            raw_state: Could be:
                - Bytes (screenshot)
                - PIL Image
                - Path to image file
                - HTML (to be rendered)
                - List of products (will be rendered as image)
                
        Returns:
            Observation with image data
        """
        # Handle different input types
        if isinstance(raw_state, bytes):
            # Already bytes (screenshot)
            image_data = self._encode_image_bytes(raw_state)
        
        elif isinstance(raw_state, str):
            if raw_state.startswith("http"):
                # URL to image
                image_data = raw_state
            else:
                # Path to file
                with open(raw_state, "rb") as f:
                    image_data = self._encode_image_bytes(f.read())
        
        elif isinstance(raw_state, list):
            # List of products - render as image
            logger.info(f"Rendering {len(raw_state)} products as visual image")
            image_data = self._render_products_as_image(raw_state)
        
        else:
            # Try to handle PIL Image or similar
            try:
                from PIL import Image
                import io
                
                if isinstance(raw_state, Image.Image):
                    buffer = io.BytesIO()
                    raw_state.save(buffer, format=self.image_format.upper())
                    image_data = self._encode_image_bytes(buffer.getvalue())
                else:
                    raise ValueError(f"Unsupported raw_state type: {type(raw_state)}")
            except ImportError:
                raise ValueError(
                    f"Cannot handle raw_state of type {type(raw_state)} without PIL"
                )
        
        return Observation(
            data=image_data,
            modality="visual",
            timestamp=time.time(),
            metadata={"format": self.image_format, "detail": self.detail_level}
        )
    
    def get_modality(self) -> str:
        """Return modality type."""
        return "visual"
    
    def validate_observation(self, obs: Observation) -> bool:
        """Check if observation is in visual format."""
        return obs.modality == "visual"
    
    # ========================================================================
    # Private Helper Methods
    # ========================================================================
    
    def _encode_image_bytes(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 data URL."""
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:image/{self.image_format};base64,{base64_image}"
    
    def _render_products_as_image(self, products: list) -> str:
        """
        将商品列表渲染为图像。
        
        为 Visual Agent 创建一个商品展示的可视化图像。
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
            import io
        except ImportError:
            logger.error("PIL not available, cannot render products as image")
            raise ImportError("Visual perception requires PIL/Pillow. Install with: pip install Pillow")
        
        # 创建图像
        width, height = 1200, min(200 + len(products) * 180, 2000)
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # 尝试加载字体
        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            font_product = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            # 使用默认字体
            font_title = ImageFont.load_default()
            font_product = ImageFont.load_default()
            font_text = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # 绘制标题
        draw.rectangle([(0, 0), (width, 80)], fill='#f0f0f0')
        draw.text((40, 25), "🛒 Search Results", fill='#333', font=font_title)
        draw.text((40, 55), f"Found {len(products)} products", fill='#666', font=font_small)
        
        # 绘制商品
        y = 100
        for i, product in enumerate(products):
            # 商品框
            box_color = '#e8f5e9' if i == 0 else '#ffffff'
            draw.rectangle([(20, y), (width-20, y+160)], fill=box_color, outline='#ddd', width=2)
            
            # 排名徽章
            rank_color = '#4caf50' if i < 3 else '#9e9e9e'
            draw.ellipse([(30, y+10), (80, y+60)], fill=rank_color)
            rank_text = f"#{i+1}"
            draw.text((45, y+25), rank_text, fill='white', font=font_product)
            
            # 商品标题（截断）
            title = product.title
            if len(title) > 80:
                title = title[:77] + "..."
            draw.text((100, y+15), title, fill='#212121', font=font_product)
            
            # 价格
            price_text = f"${product.price:.2f}"
            draw.text((100, y+50), price_text, fill='#2e7d32', font=font_title)
            
            # 评分
            rating = product.rating if product.rating else 0
            rating_text = f"{'⭐' * int(rating)} {rating:.1f}"
            draw.text((250, y+55), rating_text, fill='#ff9800', font=font_text)
            
            # 评价数
            reviews_text = f"({product.rating_count or 0} reviews)"
            draw.text((400, y+57), reviews_text, fill='#757575', font=font_small)
            
            # 特性（如果有）
            if hasattr(product, 'features') and product.features:
                features_text = " • ".join(product.features[:3])
                if len(features_text) > 100:
                    features_text = features_text[:97] + "..."
                draw.text((100, y+90), features_text, fill='#616161', font=font_small)
            
            # 分隔线
            if i < len(products) - 1:
                draw.line([(20, y+165), (width-20, y+165)], fill='#e0e0e0', width=1)
            
            y += 180
        
        # 转换为 bytes 并编码
        buffer = io.BytesIO()
        img.save(buffer, format=self.image_format.upper())
        return self._encode_image_bytes(buffer.getvalue())