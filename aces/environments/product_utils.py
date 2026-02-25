"""
Product parsing and serialization helpers.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from aces.environments.protocols import Product


def _parse_price(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace("$", "").replace(",", "").strip()
    return float(text) if text else 0.0


def _parse_rating(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_rating_count(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def product_from_dict(
    data: Dict[str, Any],
    *,
    index: Optional[int] = None,
    source: Optional[str] = None,
    category: Optional[str] = None,
    default_id_prefix: str = "product",
) -> Product:
    """
    Convert a raw product dict to Product with normalized fields.
    """
    raw_id = data.get("sku") or data.get("id")
    if raw_id:
        product_id = str(raw_id)
    elif category is not None and index is not None:
        product_id = f"{category}_{index}"
    elif index is not None:
        product_id = f"{default_id_prefix}_{index}"
    else:
        product_id = default_id_prefix

    return Product(
        id=product_id,
        title=data.get("title", "Unknown Product"),
        price=_parse_price(data.get("price")),
        rating=_parse_rating(data.get("rating")),
        rating_count=_parse_rating_count(data.get("rating_count")),
        image_url=data.get("image_url"),
        description=data.get("description"),
        sponsored=bool(data.get("sponsored", False)),
        best_seller=bool(data.get("best_seller", False)),
        overall_pick=bool(data.get("overall_pick", False)),
        low_stock=bool(data.get("low_stock", False)),
        position=data.get("position", index),
        source=source,
        raw_data=data,
    )


def product_to_summary(product: Product) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "id": product.id,
        "title": product.title,
        "price": product.price,
        "rating": product.rating,
        "rating_count": product.rating_count,
        "sponsored": product.sponsored,
        "best_seller": product.best_seller,
        "overall_pick": product.overall_pick,
        "low_stock": product.low_stock,
        "position": product.position,
        "source": product.source,
    }
    if product.raw_data:
        if "original_price" in product.raw_data:
            summary["original_price"] = product.raw_data["original_price"]
        if "custom_badges" in product.raw_data:
            summary["custom_badges"] = product.raw_data["custom_badges"]
    return summary


def product_to_detail(product: Product) -> Dict[str, Any]:
    detail: Dict[str, Any] = {
        "id": product.id,
        "title": product.title,
        "price": product.price,
        "rating": product.rating,
        "rating_count": product.rating_count,
        "description": product.description,
        "sponsored": product.sponsored,
        "best_seller": product.best_seller,
        "overall_pick": product.overall_pick,
        "low_stock": product.low_stock,
        "image_url": product.image_url,
        "source": product.source,
    }
    if product.raw_data:
        if "original_price" in product.raw_data:
            detail["original_price"] = product.raw_data["original_price"]
        if "discount_pct" in product.raw_data:
            detail["discount_pct"] = product.raw_data["discount_pct"]
        if "custom_badges" in product.raw_data:
            detail["custom_badges"] = product.raw_data["custom_badges"]
    return detail


def products_to_summaries(products: Iterable[Product]) -> list[Dict[str, Any]]:
    return [product_to_summary(p) for p in products]
