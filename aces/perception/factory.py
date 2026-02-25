"""
Perception factory for centralized creation logic.
"""

from __future__ import annotations

from typing import Any, Dict

from aces.perception.visual_perception import VisualPerception
from aces.perception.verbal_perception import VerbalPerception


class PerceptionFactory:
    @staticmethod
    def create(config: Dict[str, Any]):
        mode = (config.get("mode") or "verbal").lower()

        if mode == "visual":
            return VisualPerception(
                image_format=config.get("image_format", "png"),
                detail_level=config.get("detail_level", "high"),
            )
        if mode == "verbal":
            return VerbalPerception(
                format_style=config.get("format_style", "structured"),
            )

        raise ValueError(f"Unsupported perception mode: {mode}")
