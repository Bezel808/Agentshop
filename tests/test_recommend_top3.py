from aces.tools.shopping_browser_tools import (
    RecommendTool,
    FilterRatingTool,
    NextPageTool,
    PrevPageTool,
    SelectProductTool,
)


class _RecommendEnv:
    def __init__(self, current=None, viewed=None, products=None, state=None):
        self._current = current
        self._viewed = viewed or []
        self.products = products or []
        self._state = state or {}

    def get_current_product(self):
        return self._current

    def get_viewed_for_recommend(self):
        return list(self._viewed)

    def get_state(self):
        return dict(self._state)

    def back(self):
        return True, None

    def next_page(self):
        return True, None


class _RatingEnv:
    def __init__(self):
        self.last_min = None

    def filter_rating(self, min_val):
        self.last_min = min_val
        return True, None


class _SearchContextRecoveryEnv:
    def __init__(self):
        self.context = "detail"
        self.select_calls = 0
        self.next_calls = 0
        self.prev_calls = 0
        self.back_calls = 0

    def select_product(self, index):
        self.select_calls += 1
        if self.context != "search":
            return False, "Not on search page"
        return True, None

    def next_page(self):
        self.next_calls += 1
        if self.context != "search":
            return False, "Not on search page"
        return True, None

    def prev_page(self):
        self.prev_calls += 1
        if self.context != "search":
            return False, "Not on search page"
        return True, None

    def back(self):
        self.back_calls += 1
        self.context = "search"
        return True, None


def test_recommend_top3_accepts_ordered_product_ids():
    env = _RecommendEnv(
        current={"id": "p4", "title": "Product 4"},
        viewed=[
            {"id": "p1", "title": "Product 1"},
            {"id": "p2", "title": "Product 2"},
            {"id": "p3", "title": "Product 3"},
        ],
    )
    tool = RecommendTool(env)
    result = tool.execute({"product_ids": ["p2", "p1", "p3"]})
    assert result.success
    data = result.data
    assert data["recommended"] is True
    assert data["product_id"] == "p2"
    assert [x["product_id"] for x in data["recommendations"]] == ["p2", "p1", "p3"]


def test_recommend_top3_legacy_product_id_still_works_and_fills():
    env = _RecommendEnv(
        current=None,
        viewed=[
            {"id": "p1", "title": "Product 1"},
            {"id": "p2", "title": "Product 2"},
            {"id": "p3", "title": "Product 3"},
            {"id": "p4", "title": "Product 4"},
        ],
    )
    tool = RecommendTool(env)
    result = tool.execute({"product_id": "p2"})
    assert result.success
    recs = result.data["recommendations"]
    assert len(recs) == 3
    assert recs[0]["product_id"] == "p2"
    assert len({x["product_id"] for x in recs}) == 3


def test_recommend_top3_requires_enough_candidates():
    env = _RecommendEnv(
        current=None,
        viewed=[
            {"id": "p1", "title": "Product 1"},
            {"id": "p2", "title": "Product 2"},
        ],
    )
    tool = RecommendTool(env)
    result = tool.execute({})
    assert not result.success
    assert "at least 3" in (result.error or "").lower()


def test_recommend_top1_does_not_require_three_candidates(monkeypatch):
    monkeypatch.setenv("ACES_RECOMMENDATION_COUNT", "1")
    env = _RecommendEnv(
        current=None,
        viewed=[
            {"id": "p1", "title": "Product 1"},
        ],
    )
    tool = RecommendTool(env)
    result = tool.execute({})
    assert result.success
    assert result.data["product_id"] == "p1"
    assert [x["product_id"] for x in result.data["recommendations"]] == ["p1"]


def test_recommend_top3_accepts_ids_from_visible_search_results():
    env = _RecommendEnv(
        current=None,
        viewed=[
            {"id": "p1", "title": "Product 1"},
            {"id": "p2", "title": "Product 2"},
            {"id": "p3", "title": "Product 3"},
        ],
        products=[
            {"id": "p9", "title": "Product 9"},
            {"id": "p10", "title": "Product 10"},
        ],
    )
    tool = RecommendTool(env)
    result = tool.execute({"product_ids": ["p9", "p2", "p1"]})
    assert result.success
    assert [x["product_id"] for x in result.data["recommendations"]] == ["p9", "p2", "p1"]


def test_recommend_top3_no_longer_requires_cross_page_when_multi_page_results():
    env = _RecommendEnv(
        current=None,
        viewed=[
            {"id": "p1", "title": "Product 1", "page": 1},
            {"id": "p2", "title": "Product 2", "page": 1},
            {"id": "p3", "title": "Product 3", "page": 1},
        ],
        state={"total_pages": 5},
    )
    tool = RecommendTool(env)
    result = tool.execute({"product_ids": ["p1", "p2", "p3"]})
    assert result.success


def test_recommend_top3_succeeds_after_cross_page_exploration():
    env = _RecommendEnv(
        current=None,
        viewed=[
            {"id": "p1", "title": "Product 1", "page": 1},
            {"id": "p2", "title": "Product 2", "page": 2},
            {"id": "p3", "title": "Product 3", "page": 2},
        ],
        state={"total_pages": 5},
    )
    tool = RecommendTool(env)
    result = tool.execute({"product_ids": ["p2", "p1", "p3"]})
    assert result.success


def test_recommend_cross_page_hint_not_required_anymore():
    env = _RecommendEnv(
        current=None,
        viewed=[
            {"id": "p1", "title": "Product 1", "page": 1},
            {"id": "p2", "title": "Product 2", "page": 1},
            {"id": "p3", "title": "Product 3", "page": 1},
        ],
        state={"total_pages": 5, "page_num": 1, "context": "detail"},
    )
    tool = RecommendTool(env)
    result = tool.execute({"product_ids": ["p1", "p2", "p3"]})
    assert result.success


def test_filter_rating_clamps_value_into_valid_range():
    env = _RatingEnv()
    tool = FilterRatingTool(env)

    res_high = tool.execute({"min": 8})
    assert res_high.success
    assert env.last_min == 5.0

    res_low = tool.execute({"min": -2})
    assert res_low.success
    assert env.last_min == 0.0


def test_search_tools_auto_recover_from_detail_context():
    env = _SearchContextRecoveryEnv()

    select_tool = SelectProductTool(env)
    next_tool = NextPageTool(env)
    prev_tool = PrevPageTool(env)

    select_res = select_tool.execute({"index": 1})
    assert select_res.success
    assert select_res.data.get("auto_recovered") in {"direct_retry", "back_then_retry"}
    assert env.back_calls >= 1

    env.context = "detail"
    next_res = next_tool.execute({})
    assert next_res.success
    assert next_res.data.get("auto_recovered") in {"direct_retry", "back_then_retry"}

    env.context = "detail"
    prev_res = prev_tool.execute({})
    assert prev_res.success
    assert prev_res.data.get("auto_recovered") in {"direct_retry", "back_then_retry"}
