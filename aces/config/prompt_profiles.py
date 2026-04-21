"""Prompt profile presets for shopping agent behavior tuning."""

from __future__ import annotations

from typing import Dict, Any


PROMPT_PROFILES: Dict[str, Dict[str, Any]] = {
    "robust_compare": {
        "name": "Robust Compare",
        "confidence_threshold": 85,
        "min_viewed_before_recommend": 1,
        "style": "conservative",
        "system_suffix": (
            "Prioritize requirement fidelity and avoid premature recommendation. "
            "When uncertain, compare more products and use filters explicitly. "
            "Avoid premature recommendation; finalize once constraints are satisfied with sufficient evidence."
        ),
    },
    "active_explore": {
        "name": "Active Explore",
        "confidence_threshold": 85,
        "min_viewed_before_recommend": 1,
        "style": "exploratory",
        "system_suffix": (
            "Prefer exploring multiple pages and applying filters before recommending. "
            "Maximize candidate coverage while keeping tool calls valid."
        ),
    },
    "fast_decide": {
        "name": "Fast Decide",
        "confidence_threshold": 85,
        "min_viewed_before_recommend": 1,
        "style": "decisive",
        "system_suffix": (
            "Optimize for shorter decision paths while preserving accuracy. "
            "Recommend once evidence is sufficient and constraints are met."
        ),
    },
}


def get_prompt_profile(name: str | None) -> Dict[str, Any]:
    key = (name or "robust_compare").strip().lower()
    if key not in PROMPT_PROFILES:
        key = "robust_compare"
    profile = dict(PROMPT_PROFILES[key])
    profile["key"] = key
    return profile
