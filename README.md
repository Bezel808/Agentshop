# ACES-v2 — Agent-Oriented Project Specification

## PROJECT
- **Name**: ACES-v2
- **Version**: 2.0.0
- **Purpose**: Modular framework for AI agent e-commerce behavior research (behavioral economics). Agent searches products, perceives (verbal/visual), decides, adds to cart.
- **Python**: >=3.10

## DIRECTORY STRUCTURE
```
aces/
  agents/         base_agent.py → ComposableAgent
  llm_backends/   openai, deepseek, qwen (VLM: qwen-vl-plus)
  perception/     verbal_perception, visual_perception
  environments/   offline_marketplace, llamaindex_marketplace, web_renderer_marketplace, online_marketplace
  tools/          shopping_tools (SearchTool, AddToCartTool, ViewProductDetailsTool)
  experiments/    control_variables, runner, protocols, interventions, metrics
  config/         settings, loader
  core/           protocols (Observation, Action, ToolResult, etc.)
  data/           loader
configs/          experiments/*.yaml, environments/*.yaml
datasets_unified/ *.json (one file per category, product list)
web/              templates/, static/
```

## ENTRY POINTS
| Script | Purpose | Key Args |
|--------|---------|----------|
| `./aces_ctl.sh start` | Start FastAPI web server (port 5000) | ACES_PORT, ACES_DATASETS, ACES_SIMPLE_SEARCH |
| `./aces_ctl.sh stop` | Stop web server | |
| `./aces_ctl.sh status` | Health check | |
| `./aces_ctl.sh run --query X` | Run experiment (delegates to run_experiment.py) | |
| `run_experiment.py` | Main experiment runner | --mode simple\|llm, --query, --llm qwen\|openai\|deepseek, --perception visual\|verbal, --condition-file, --marketplace offline\|llamaindex |
| `run_browser_agent.py` | VLM + Playwright: navigates web, screenshots, pushes to /viewer | --api-key, --llm qwen, --query |
| `run_ranking_experiment.py` | VLM ranks products | --api-key, --llm qwen |
| `start_web_server.py` | Direct server start | --port, --host, --datasets-dir, --simple-search |
| `expand_from_amazon.py` | Expand product DB from Amazon Reviews 2023 | --categories, --per-category, --scan-limit, --skip-image-check |

## DATA FLOW (Agent Receives Product Info)
1. **Marketplace.search_products(query)** → SearchResult.products (List[Product])
2. **OfflineMarketplace** loads JSON from `datasets_unified/{query}.json`, applies ExperimentCondition if set
3. **LlamaIndexMarketplace** (web server): BM25 + vector + reranker, semantic search over all products
4. **product_to_summary(Product)** → dict with id, title, price, rating, rating_count, sponsored, best_seller, overall_pick, low_stock, position, original_price, custom_badges
5. **Perception.encode(products)** → Observation: verbal=JSON/text, visual=screenshot
6. **Agent.act(observation)** → Action(tool_name, parameters)
7. **Tool.execute** → marketplace.add_to_cart / search_products

## SCHEMAS

### Product (aces/environments/protocols.py)
```python
Product(id, title, price, rating?, rating_count?, image_url?, description?,
        sponsored=False, best_seller=False, overall_pick=False, low_stock=False,
        position?, source?, raw_data?)
```

### Product JSON (datasets_unified/*.json)
```json
[{"sku":"cat_0","title":"...","price":7.99,"rating":4.6,"rating_count":2774,
  "image_url":"https://...","description":"","sponsored":false,"best_seller":false,
  "overall_pick":false,"low_stock":false}]
```

### ExperimentCondition (YAML, aces/experiments/control_variables.py)
> **速查**：价格/描述/标签等控制变量的完整说明 → [CONDITION_REFERENCE.md](CONDITION_REFERENCE.md)
```yaml
name: study_name
design: between | within | latin_square
conditions:
  - name: treatment
    product_overrides: { "mousepad_0": { "price": 49.99, "best_seller": true } }
    position_overrides: { 0: { "title_prefix": "[Premium] " } }
    global_overrides: { "inject_labels": { "sponsored": [0] } }
```
Keys: price, price_multiplier, original_price, sponsored, best_seller, overall_pick, low_stock, title, title_prefix, title_suffix, rating, rating_count, description.

## CONFIGURATION
- **Env**: QWEN_API_KEY (VLM), DEEPSEEK_API_KEY, OPENAI_API_KEY
- **Default LLM**: qwen (VLM experiments)
- **Default perception**: visual
- **Marketplace modes**: offline (JSON files), llamaindex (RAG), web_renderer (simple match), online (Playwright scrape)

## WEB SERVER ENDPOINTS
| Path | Method | Purpose |
|------|--------|---------|
| / | GET | Search home |
| /search?q= | GET | Product search (semantic if LlamaIndex) |
| /product/<id>?q= | GET | Product detail |
| /viewer | GET | Live viewer page (WebSocket) |
| /ws/viewer | WS | Push screenshots/logs |
| /api/push | POST | Agent pushes to viewers |
| /health | GET | Status JSON |

## EXPERIMENT RUN
```bash
# VLM (default qwen + visual)
python run_experiment.py --query "性价比高的鼠标垫"

# With condition file
python run_experiment.py --query mousepad -C configs/experiments/example_price_anchoring.yaml --condition-name treatment_anchor

# Simple (no LLM)
python run_experiment.py --mode simple --query mousepad --trials 10
```

## DEPENDENCIES
- Core: pydantic, pyyaml, jsonschema
- [all]: openai, playwright, llama-index, fastapi, uvicorn, jinja2
- Install: `pip install -e ".[all]"`

## KEY IMPORTS
```python
from aces.environments import MarketplaceFactory, MarketplaceAdapter, OfflineMarketplace, LlamaIndexMarketplace
from aces.agents import ComposableAgent
from aces.tools import SearchTool, AddToCartTool
from aces.experiments.control_variables import ExperimentCondition, ConditionSet, load_conditions_from_yaml
from aces.environments.product_utils import product_from_dict, product_to_summary, product_to_detail
```
