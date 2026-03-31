from aces.agents.base_agent import ComposableAgent
from aces.core.protocols import Message, Observation


class _DummyLLM:
    def __init__(self, response):
        self._response = response

    def generate(self, messages, tools=None, **kwargs):
        return self._response

    async def agenerate(self, messages, tools=None, **kwargs):
        return self._response

    def count_tokens(self, messages):
        return 0

    @property
    def model_name(self):
        return "dummy"


class _DummyPerception:
    def encode(self, raw_state):
        return raw_state

    def get_modality(self):
        return "verbal"

    def validate_observation(self, obs):
        return True


class _DummyTool:
    def __init__(self, name):
        self.name = name

    def get_schema(self):
        from aces.core.protocols import ToolSchema
        return ToolSchema(name=self.name, description="", input_schema={"type": "object", "properties": {}})

    def execute(self, parameters):
        from aces.core.protocols import ToolResult
        return ToolResult(success=True, data=parameters)

    async def aexecute(self, parameters):
        return self.execute(parameters)

    def validate_parameters(self, parameters):
        return True


def _obs():
    return Observation(data="hi", modality="verbal", timestamp=0.0)


def test_parse_tool_calls_batch():
    llm_resp = Message(
        role="assistant",
        content={
            "tool_calls": [
                {"name": "search_products", "parameters": {"query": "watch"}},
                {"name": "add_to_cart", "parameters": {"product_id": "p1"}},
            ],
            "reasoning": "do two steps",
        },
    )
    agent = ComposableAgent(
        llm=_DummyLLM(llm_resp),
        perception=_DummyPerception(),
        tools=[_DummyTool("search_products"), _DummyTool("add_to_cart")],
        legacy_fallback=False,
    )
    actions = agent.act_batch(_obs())
    assert len(actions) == 2
    assert actions[0].tool_name == "search_products"
    assert actions[1].tool_name == "add_to_cart"


def test_legacy_fallback_toggle():
    llm_resp = Message(role="assistant", content="I cannot decide yet.")
    agent_disabled = ComposableAgent(
        llm=_DummyLLM(llm_resp),
        perception=_DummyPerception(),
        tools=[_DummyTool("search_products")],
        legacy_fallback=False,
    )
    assert agent_disabled.act(_obs()).tool_name == "noop"

    agent_enabled = ComposableAgent(
        llm=_DummyLLM(llm_resp),
        perception=_DummyPerception(),
        tools=[_DummyTool("search_products")],
        legacy_fallback=True,
    )
    assert agent_enabled.act(_obs()).tool_name == "noop"


def test_text_heuristic_select_product():
    llm_resp = Message(role="assistant", content="Action: select_product 2")
    agent = ComposableAgent(
        llm=_DummyLLM(llm_resp),
        perception=_DummyPerception(),
        tools=[_DummyTool("select_product")],
        legacy_fallback=False,
    )
    action = agent.act(_obs())
    assert action.tool_name == "select_product"
    assert action.parameters["index"] == 2


def test_parse_python_literal_tool_calls_string():
    llm_resp = Message(
        role="assistant",
        content=(
            "{'tool_calls': [{'id': 'call_x', 'name': 'select_product', "
            "'parameters': {'index': 4}}], "
            "'tool_call': {'id': 'call_x', 'name': 'select_product', "
            "'parameters': {'index': 4}}, 'reasoning': ''}"
        ),
    )
    agent = ComposableAgent(
        llm=_DummyLLM(llm_resp),
        perception=_DummyPerception(),
        tools=[_DummyTool("select_product")],
        legacy_fallback=False,
    )
    action = agent.act(_obs())
    assert action.tool_name == "select_product"
    assert action.parameters["index"] == 4


def test_parse_functions_style_text_ignores_step_prefix():
    llm_resp = Message(role="assistant", content='functions.filter_price:7_${{"max": 100}}')
    agent = ComposableAgent(
        llm=_DummyLLM(llm_resp),
        perception=_DummyPerception(),
        tools=[_DummyTool("filter_price")],
        legacy_fallback=False,
    )
    action = agent.act(_obs())
    assert action.tool_name == "filter_price"
    assert action.parameters == {"max": 100}


def test_parse_functions_style_filter_rating_uses_payload_value():
    llm_resp = Message(role="assistant", content='functions.filter_rating:8_${{"min": 4}}')
    agent = ComposableAgent(
        llm=_DummyLLM(llm_resp),
        perception=_DummyPerception(),
        tools=[_DummyTool("filter_rating")],
        legacy_fallback=False,
    )
    action = agent.act(_obs())
    assert action.tool_name == "filter_rating"
    assert action.parameters == {"min": 4}


def test_parse_functions_style_filter_rating_weird_wrapper_uses_payload_value():
    llm_resp = Message(role="assistant", content='functions.filter_rating:1>{${"min":4}}')
    agent = ComposableAgent(
        llm=_DummyLLM(llm_resp),
        perception=_DummyPerception(),
        tools=[_DummyTool("filter_rating")],
        legacy_fallback=False,
    )
    action = agent.act(_obs())
    assert action.tool_name == "filter_rating"
    assert action.parameters == {"min": 4}


def test_parse_functions_style_select_product_weird_wrapper_uses_payload_value():
    llm_resp = Message(role="assistant", content='functions.select_product:4/{${"index":4}}')
    agent = ComposableAgent(
        llm=_DummyLLM(llm_resp),
        perception=_DummyPerception(),
        tools=[_DummyTool("select_product")],
        legacy_fallback=False,
    )
    action = agent.act(_obs())
    assert action.tool_name == "select_product"
    assert action.parameters == {"index": 4}
