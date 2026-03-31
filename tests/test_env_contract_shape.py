from aces.environments.api_shopping_env import APIShoppingEnv
from aces.environments.browser_shopping_env import BrowserShoppingEnv
from aces.environments.mcp_shopping_env import MCPShoppingEnv


REQUIRED_METHODS = [
    "reset",
    "get_observation",
    "get_available_actions",
    "execute_action",
    "get_state",
    "select_product",
    "next_page",
    "prev_page",
    "filter_price",
    "filter_rating",
    "back",
    "get_viewed_for_recommend",
    "get_current_product",
    "close",
]


def test_api_env_contract_methods_exist():
    for name in REQUIRED_METHODS:
        assert hasattr(APIShoppingEnv, name), name


def test_browser_env_contract_methods_exist():
    for name in REQUIRED_METHODS:
        assert hasattr(BrowserShoppingEnv, name), name


def test_mcp_env_requires_caller():
    try:
        MCPShoppingEnv(mcp_caller=None)
    except RuntimeError as e:
        assert "mcp_caller" in str(e)
    else:
        raise AssertionError("MCPShoppingEnv should fail fast when mcp_caller is missing")
