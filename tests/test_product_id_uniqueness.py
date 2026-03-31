from pathlib import Path

from aces.environments.llamaindex_marketplace import LlamaIndexMarketplace
from aces.environments.web_renderer_marketplace import WebRendererMarketplace


DATASETS_DIR = Path("/home/zongze/research/ACES-v2/datasets_40_work/enriched")


def _check_unique_ids(products):
    ids = [p.id for p in products]
    assert len(ids) == len(set(ids)), "loaded product IDs should be unique"


def test_llamaindex_loader_deduplicates_ids():
    m = LlamaIndexMarketplace()
    m.datasets_dir = DATASETS_DIR
    products = m._load_all_products()
    assert products, "products should not be empty"
    _check_unique_ids(products)
    # We know this dataset contains duplicates; ensure dedup suffix exists.
    assert any("__dup" in p.id for p in products)


def test_web_renderer_loader_deduplicates_ids():
    m = WebRendererMarketplace()
    m.datasets_dir = DATASETS_DIR
    products = m._load_all_products()
    assert products, "products should not be empty"
    _check_unique_ids(products)
    assert any("__dup" in p.id for p in products)
