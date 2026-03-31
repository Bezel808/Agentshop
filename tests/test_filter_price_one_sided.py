from aces.tools.shopping_browser_tools import FilterPriceTool


class _Env:
    def __init__(self):
        self.called = None

    def get_state(self):
        return {"price_min": 10.0, "price_max": 200.0}

    def filter_price(self, min_val, max_val):
        self.called = (min_val, max_val)
        return True, None


def test_filter_price_only_max():
    env = _Env()
    tool = FilterPriceTool(env)
    result = tool.execute({"max": 80})
    assert result.success
    assert env.called == (0.0, 80.0)


def test_filter_price_only_min():
    env = _Env()
    tool = FilterPriceTool(env)
    result = tool.execute({"min": 50})
    assert result.success
    assert env.called == (50.0, 999999.0)
