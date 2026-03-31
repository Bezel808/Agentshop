from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_search_template_has_sidebar_and_result_list():
    text = (ROOT / "web/templates/search.html").read_text(encoding="utf-8")
    assert "catalog-sidebar" in text
    assert "result-list" in text


def test_detail_template_has_buybox_layout():
    text = (ROOT / "web/templates/product_detail.html").read_text(encoding="utf-8")
    assert "detail-buybox-column" in text
    assert "buybox-card" in text
