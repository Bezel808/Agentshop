from aces.config.prompt_profiles import get_prompt_profile, PROMPT_PROFILES
from run_browser_agent import LiveBrowserAgent, _build_system_prompt


def test_prompt_profile_default_and_known():
    p = get_prompt_profile(None)
    assert p["key"] == "robust_compare"
    p2 = get_prompt_profile("active_explore")
    assert p2["key"] == "active_explore"


def test_prompt_profile_fallback_unknown():
    p = get_prompt_profile("not_exists")
    assert p["key"] == "robust_compare"


def test_budget_extraction_under_and_between():
    b1 = LiveBrowserAgent._extract_budget_constraints("smart watch under 120 dollars")
    assert b1["price_max"] == 120

    b2 = LiveBrowserAgent._extract_budget_constraints("speaker between $30 and 60")
    assert b2["price_min"] == 30
    assert b2["price_max"] == 60


def test_profile_keys_exposed():
    assert {"robust_compare", "active_explore", "fast_decide"}.issubset(set(PROMPT_PROFILES.keys()))


def test_system_prompt_includes_budget_first_hint_when_provided():
    profile = get_prompt_profile("robust_compare")
    prompt = _build_system_prompt(
        profile,
        recommendation_count=1,
        budget_hint="Budget-first rule: prioritize filter_price early.",
    )
    assert "filter_price" in prompt
    assert "Budget-first rule" in prompt


def test_strip_budget_from_keywords():
    text = "smart watch under $80 dollars with heart rate"
    stripped = LiveBrowserAgent._strip_budget_from_keywords(text)
    low = stripped.lower()
    assert "under" not in low
    assert "$80" not in low
    assert "dollars" not in low
    assert "smart watch" in low


def test_strip_budget_numeric_range_from_keywords():
    text = "smart watch 50-60 with sleep tracking"
    stripped = LiveBrowserAgent._strip_budget_from_keywords(text)
    low = stripped.lower()
    assert "50-60" not in low
    assert "smart watch" in low
    assert "sleep tracking" in low


def test_parse_intent_json_separates_budget_and_keywords():
    raw = '{"keywords":"smartwatch with sleep tracking","price_min":null,"price_max":60}'
    parsed = LiveBrowserAgent._parse_intent_json(raw)
    assert parsed["keywords"] == "smartwatch with sleep tracking"
    assert parsed["price_min"] is None
    assert parsed["price_max"] == 60.0


def test_parse_intent_json_strips_budget_tokens_from_keywords():
    raw = "{'keywords': 'smart watch 50-60 sleep tracking', 'price_min': 50, 'price_max': 60}"
    parsed = LiveBrowserAgent._parse_intent_json(raw)
    low = parsed["keywords"].lower()
    assert "50-60" not in low
    assert "smart watch" in low
    assert "sleep tracking" in low
    assert parsed["price_min"] == 50.0
    assert parsed["price_max"] == 60.0
