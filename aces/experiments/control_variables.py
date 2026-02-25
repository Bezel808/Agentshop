"""
Experimental Control Variables System

Provides a declarative way to define and apply experimental conditions
(treatments) to product data. Designed for behavioral economics experiments
where researchers need fine-grained control over:

  - Product prices (override, multiply, add noise)
  - Product labels / badges (sponsored, best_seller, etc.)
  - Product names / titles (prefix, suffix, replacement)
  - Custom attributes (brand, origin, description, etc.)

Usage:
    from aces.experiments.control_variables import ExperimentCondition, ConditionSet

    # Define a treatment condition
    treatment = ExperimentCondition(
        name="high_price_anchor",
        description="Add a 2x price decoy at position 0",
        product_overrides={
            "mousepad_0": {
                "price": 49.99,
                "title_prefix": "[Premium] ",
                "best_seller": True,
            }
        },
        global_overrides={
            "price_multiplier": 1.0,
            "inject_labels": {"sponsored": [0]},  # make position-0 sponsored
        },
    )

    # Apply to a product list
    modified_products = treatment.apply(products)
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from aces.environments.protocols import Product

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Atomic variable transforms
# ---------------------------------------------------------------------------

@dataclass
class PriceOverride:
    """Control variable for price manipulation."""
    fixed_value: Optional[float] = None
    multiplier: Optional[float] = None
    additive_noise_range: Optional[Tuple[float, float]] = None
    original_price_display: Optional[float] = None  # show "was $X" anchoring

    def apply(self, current_price: float, rng: random.Random) -> float:
        price = current_price
        if self.fixed_value is not None:
            price = self.fixed_value
        if self.multiplier is not None:
            price = price * self.multiplier
        if self.additive_noise_range is not None:
            lo, hi = self.additive_noise_range
            price += rng.uniform(lo, hi)
        return round(max(0.0, price), 2)


@dataclass
class LabelOverride:
    """Control variable for badge / label manipulation."""
    sponsored: Optional[bool] = None
    best_seller: Optional[bool] = None
    overall_pick: Optional[bool] = None
    low_stock: Optional[bool] = None
    custom_badges: Optional[List[str]] = None  # e.g. ["Limited Edition"]


@dataclass
class TitleOverride:
    """Control variable for title manipulation."""
    fixed_value: Optional[str] = None
    prefix: Optional[str] = None
    suffix: Optional[str] = None
    replacements: Optional[Dict[str, str]] = None  # {"old": "new"}


@dataclass
class ProductOverride:
    """Composite override for a single product."""
    price: Optional[PriceOverride] = None
    label: Optional[LabelOverride] = None
    title: Optional[TitleOverride] = None
    rating: Optional[float] = None
    rating_count: Optional[int] = None
    description: Optional[str] = None
    image_url: Optional[str] = None
    extra_attributes: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProductOverride":
        """Convenient builder from a flat dict.

        Accepted keys (all optional):
            price           -> float          (fixed price)
            price_multiplier -> float
            price_noise     -> [lo, hi]
            original_price  -> float          (anchor display)
            sponsored       -> bool
            best_seller     -> bool
            overall_pick    -> bool
            low_stock       -> bool
            custom_badges   -> list[str]
            title           -> str            (replace title)
            title_prefix    -> str
            title_suffix    -> str
            title_replace   -> {old: new, ...}
            rating          -> float
            rating_count    -> int
            description     -> str
            image_url       -> str
            *               -> extra_attributes
        """
        price_kw: Dict[str, Any] = {}
        if "price" in d and isinstance(d["price"], (int, float)):
            price_kw["fixed_value"] = float(d["price"])
        if "price_multiplier" in d:
            price_kw["multiplier"] = float(d["price_multiplier"])
        if "price_noise" in d:
            price_kw["additive_noise_range"] = tuple(d["price_noise"])
        if "original_price" in d:
            price_kw["original_price_display"] = float(d["original_price"])

        label_kw: Dict[str, Any] = {}
        for k in ("sponsored", "best_seller", "overall_pick", "low_stock"):
            if k in d:
                label_kw[k] = bool(d[k])
        if "custom_badges" in d:
            label_kw["custom_badges"] = list(d["custom_badges"])

        title_kw: Dict[str, Any] = {}
        if "title" in d and isinstance(d["title"], str):
            title_kw["fixed_value"] = d["title"]
        if "title_prefix" in d:
            title_kw["prefix"] = d["title_prefix"]
        if "title_suffix" in d:
            title_kw["suffix"] = d["title_suffix"]
        if "title_replace" in d:
            title_kw["replacements"] = d["title_replace"]

        known_keys = {
            "price", "price_multiplier", "price_noise", "original_price",
            "sponsored", "best_seller", "overall_pick", "low_stock", "custom_badges",
            "title", "title_prefix", "title_suffix", "title_replace",
            "rating", "rating_count", "description", "image_url",
        }
        extra = {k: v for k, v in d.items() if k not in known_keys}

        return cls(
            price=PriceOverride(**price_kw) if price_kw else None,
            label=LabelOverride(**label_kw) if label_kw else None,
            title=TitleOverride(**title_kw) if title_kw else None,
            rating=d.get("rating"),
            rating_count=d.get("rating_count"),
            description=d.get("description"),
            image_url=d.get("image_url"),
            extra_attributes=extra,
        )


# ---------------------------------------------------------------------------
# ExperimentCondition: a single treatment / control condition
# ---------------------------------------------------------------------------

@dataclass
class ExperimentCondition:
    """
    A named experimental condition (treatment or control).

    Parameters
    ----------
    name : str
        Short identifier, e.g. "treatment_high_anchor".
    description : str
        Human-readable explanation for the lab notebook.
    product_overrides : dict
        Mapping from product_id -> override dict.  The override dict uses
        the same flat keys accepted by ProductOverride.from_dict().
    position_overrides : dict
        Mapping from position (int) -> override dict.  Applied to whatever
        product lands at that position *after* sorting.
    global_overrides : dict
        Applied to *all* products.  Same flat-key format.  Additionally
        supports:
            inject_labels -> {label_name: [positions]}
            shuffle_seed  -> int | None
    seed : int or None
        RNG seed for reproducibility.
    """
    name: str
    description: str = ""
    product_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    position_overrides: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    global_overrides: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None

    def apply(self, products: List[Product]) -> List[Product]:
        """Return a new list of Product with overrides applied (non-mutating)."""
        rng = random.Random(self.seed)
        result: List[Product] = []

        # Optional shuffle
        indices = list(range(len(products)))
        shuffle_seed = self.global_overrides.get("shuffle_seed")
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(indices)

        for new_pos, orig_idx in enumerate(indices):
            p = copy.deepcopy(products[orig_idx])

            # 1) Global overrides
            global_ov = {
                k: v for k, v in self.global_overrides.items()
                if k not in ("inject_labels", "shuffle_seed")
            }
            if global_ov:
                p = self._apply_override(p, ProductOverride.from_dict(global_ov), rng)

            # 2) Per-product overrides (by id)
            if p.id in self.product_overrides:
                ov = ProductOverride.from_dict(self.product_overrides[p.id])
                p = self._apply_override(p, ov, rng)

            # 3) Per-position overrides
            if new_pos in self.position_overrides:
                ov = ProductOverride.from_dict(self.position_overrides[new_pos])
                p = self._apply_override(p, ov, rng)

            # 4) Label injection by position
            inject = self.global_overrides.get("inject_labels", {})
            for label_name, positions in inject.items():
                if new_pos in positions:
                    if hasattr(p, label_name):
                        object.__setattr__(p, label_name, True)

            p.position = new_pos
            result.append(p)

        logger.info(
            f"Applied condition '{self.name}' to {len(result)} products"
        )
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "product_overrides": self.product_overrides,
            "position_overrides": {str(k): v for k, v in self.position_overrides.items()},
            "global_overrides": self.global_overrides,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentCondition":
        pos_ov = {}
        for k, v in d.get("position_overrides", {}).items():
            pos_ov[int(k)] = v
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            product_overrides=d.get("product_overrides", {}),
            position_overrides=pos_ov,
            global_overrides=d.get("global_overrides", {}),
            seed=d.get("seed"),
        )

    # -- internal --

    @staticmethod
    def _apply_override(
        product: Product,
        ov: ProductOverride,
        rng: random.Random,
    ) -> Product:
        if ov.price is not None:
            product.price = ov.price.apply(product.price, rng)
            if ov.price.original_price_display is not None:
                if product.raw_data is None:
                    product.raw_data = {}
                product.raw_data["original_price"] = ov.price.original_price_display

        if ov.label is not None:
            lb = ov.label
            if lb.sponsored is not None:
                product.sponsored = lb.sponsored
            if lb.best_seller is not None:
                product.best_seller = lb.best_seller
            if lb.overall_pick is not None:
                product.overall_pick = lb.overall_pick
            if lb.low_stock is not None:
                product.low_stock = lb.low_stock
            if lb.custom_badges:
                if product.raw_data is None:
                    product.raw_data = {}
                product.raw_data["custom_badges"] = lb.custom_badges

        if ov.title is not None:
            t = ov.title
            if t.fixed_value is not None:
                product.title = t.fixed_value
            if t.replacements:
                for old, new in t.replacements.items():
                    product.title = product.title.replace(old, new)
            if t.prefix:
                product.title = t.prefix + product.title
            if t.suffix:
                product.title = product.title + t.suffix

        if ov.rating is not None:
            product.rating = ov.rating
        if ov.rating_count is not None:
            product.rating_count = ov.rating_count
        if ov.description is not None:
            product.description = ov.description
        if ov.image_url is not None:
            product.image_url = ov.image_url

        if ov.extra_attributes:
            if product.raw_data is None:
                product.raw_data = {}
            product.raw_data.update(ov.extra_attributes)

        return product


# ---------------------------------------------------------------------------
# ConditionSet: manage multiple conditions for between-subjects designs
# ---------------------------------------------------------------------------

@dataclass
class ConditionSet:
    """
    A set of experimental conditions for a between-subjects or within-subjects
    design.

    Example (between-subjects A/B test):

        cs = ConditionSet(
            name="price_anchoring_study",
            conditions=[control_condition, treatment_condition],
            design="between",
        )

        # Assign trial 0 -> condition, trial 1 -> condition, ...
        condition = cs.assign(trial_number=0)
        modified = condition.apply(products)

    Example (within-subjects / full factorial):

        cs = ConditionSet(
            name="2x2_price_label",
            conditions=[low_plain, low_badge, high_plain, high_badge],
            design="within",
        )
    """
    name: str
    conditions: List[ExperimentCondition]
    design: str = "between"  # "between" | "within" | "latin_square"
    seed: Optional[int] = None

    def assign(self, trial_number: int) -> ExperimentCondition:
        """Assign a condition to a trial number."""
        if self.design == "between":
            idx = trial_number % len(self.conditions)
            return self.conditions[idx]
        elif self.design == "within":
            idx = trial_number % len(self.conditions)
            return self.conditions[idx]
        elif self.design == "latin_square":
            n = len(self.conditions)
            row = trial_number // n
            col = (trial_number + row) % n
            return self.conditions[col]
        else:
            raise ValueError(f"Unknown design: {self.design}")

    def all_condition_names(self) -> List[str]:
        return [c.name for c in self.conditions]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "design": self.design,
            "seed": self.seed,
            "conditions": [c.to_dict() for c in self.conditions],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConditionSet":
        conditions = [
            ExperimentCondition.from_dict(c) for c in d.get("conditions", [])
        ]
        return cls(
            name=d["name"],
            conditions=conditions,
            design=d.get("design", "between"),
            seed=d.get("seed"),
        )


# ---------------------------------------------------------------------------
# YAML / JSON config loader helper
# ---------------------------------------------------------------------------

def load_conditions_from_yaml(path: str) -> ConditionSet:
    """Load a ConditionSet from a YAML file."""
    import yaml
    from pathlib import Path as _P

    with open(_P(path), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return ConditionSet.from_dict(data)


def load_conditions_from_json(path: str) -> ConditionSet:
    """Load a ConditionSet from a JSON file."""
    import json
    from pathlib import Path as _P

    with open(_P(path), "r", encoding="utf-8") as f:
        data = json.load(f)

    return ConditionSet.from_dict(data)
