# ACES-v2 项目详细说明报告

- 报告日期: 2026-03-24
- 项目路径: `/home/zongze/research/ACES-v2`
- 产出方式: 主代理 + 3 个 subagent 并行分析（源码架构 / 配置与工作流 / 文档数据测试）后统一汇总

---

## 1. 项目定位

ACES-v2 是一个电商购物 Agent 实验框架：

1. 用 LLM/VLM 理解用户购物需求；
2. 在多页 marketplace（本地渲染 Web + API）中检索、浏览、筛选商品；
3. 通过 Tool-Calling 决策循环完成最终推荐；
4. 支持实验条件注入（价格锚定、标签框架、标题/描述改写等）和评估产物输出。

核心入口：

- `run_browser_agent.py`（Agent 统一入口）
- `start_web_server.py`（Web/API 服务入口）
- `aces_ctl.sh`（服务与运行控制脚本）
- `manage_datasets.py`（数据扩展/补齐/筛选统一入口）

---

## 2. 仓库快照（本次核验）

- 代码检索文件数（`rg --files`）：147
- 目录结构完整，核心模块集中在 `aces/`, `configs/`, `docs/`, `tests/`, `scripts/`, `web/`
- 当前默认服务数据目录 `datasets_unified/` 下有 8 个类目 JSON：
  - `athletic_shoes_men.json` (10)
  - `athletic_shoes_women.json` (11)
  - `backpack_men.json` (22)
  - `backpack_women.json` (40)
  - `bluetooth_speaker.json` (40)
  - `insulated_cup_men.json` (22)
  - `insulated_cup_women.json` (18)
  - `smart_watch.json` (11)

样本字段基本一致：`sku,title,price,rating,rating_count,image_url,image_urls,description,reviews,sponsored,best_seller,overall_pick,low_stock,parent_asin`。

---

## 3. 总体架构

### 3.1 分层

1. 协议层（抽象契约）
- `aces/core/protocols.py`
- 定义 `Message/Observation/Action/ToolResult/Agent/Tool/LLMBackend` 等核心协议。

2. 决策层（Agent）
- `aces/agents/base_agent.py`
- `ComposableAgent` 负责将 LLM、Perception、Tools 组合，解析结构化 tool-calls 并产出 `Action`。

3. 编排层（Orchestration）
- `aces/orchestration/agent_orchestrator.py`
- 统一执行循环：`observation -> action -> tool_result -> termination`，内置重复动作与错误熔断。

4. 环境层（Environment）
- 交互环境：
  - `aces/environments/api_shopping_env.py`（verbal / API）
  - `aces/environments/browser_shopping_env.py`（visual / Playwright）
  - `aces/environments/mcp_shopping_env.py`（MCP 浏览器桥接）
- 检索 provider：
  - `offline_marketplace.py`
  - `llamaindex_marketplace.py`
  - `online_marketplace.py`

5. 工具层（Tool-First）
- `aces/tools/shopping_browser_tools.py`
- 7 个核心工具：`select_product`, `next_page`, `prev_page`, `filter_price`, `filter_rating`, `back`, `recommend`。

6. 模型后端层
- `aces/llm_backends/`（OpenAI/Qwen/Kimi/DeepSeek）。

7. 实验层
- `aces/experiments/`（条件变量、指标、runner、轨迹日志）。

### 3.2 主调用链（实际运行）

1. `run_browser_agent.py:main()` 解析 CLI + API key。
2. 创建 `LiveBrowserAgent`：选择 LLM、perception、env backend，并注入 7 个工具。
3. `extract_search_keywords()` 提炼检索关键词。
4. `_run_unified()` 启动 `AgentOrchestrator.run()`。
5. 每步循环执行：
   - `env.reset/get_observation`
   - `agent.act_batch()/act()`
   - `tool.execute()`
   - `should_stop`（recommend 成功即终止）
6. 可选写出 trace（JSONL）并推送 viewer 事件。

---

## 4. 服务与前端能力

`start_web_server.py` 提供：

1. 页面路由
- `/`、`/search`、`/product/{product_id}`、`/viewer`

2. API 路由
- `/api/search`
- `/api/product/{product_id}`
- `/api/run-agent`
- `/api/agent-status`
- `/api/stop-agent`
- `/api/push`
- `/health`

3. 实验条件注入
- 启动时可用 `--condition-file` 加载 YAML/JSON 条件
- 搜索和详情可通过 `condition_name` 应用对应条件

4. 分页与筛选
- 价格和评分筛选（`price_min/max`, `rating_min`）
- 最多页数由 `max_pages` 控制
- 小数据集场景下支持补齐分页候选

---

## 5. 配置与参数体系

### 5.1 依赖

`pyproject.toml` 中 extras 分层：

- 基础: `pydantic`, `pyyaml`, `jsonschema`, `requests`
- LLM: `openai`, `anthropic`, `transformers/torch/accelerate`
- online/browser: `playwright`, `beautifulsoup4`
- RAG: `llama-index*`, `sentence-transformers`
- web: `fastapi`, `uvicorn`, `jinja2`
- analysis/dev: `pandas/matplotlib/seaborn`, `pytest/ruff/mypy` 等

### 5.2 运行参数分层

1. 服务层
- `aces_ctl.sh` 环境变量：`ACES_PORT`, `ACES_HOST`, `ACES_DATASETS`, `ACES_MAX_PAGES`, `ACES_SIMPLE_SEARCH`, `ACES_LOG`
- `start_web_server.py`：`--port --host --datasets-dir --simple-search --condition-file --max-pages --index-cache-dir --rebuild-index`

2. Agent 层（`run_browser_agent.py`）
- LLM: `--llm --api-key --temperature`
- 决策控制: `--max-steps --max-repeated-actions --max-errors --max-history`
- 业务约束: `--query --page --price-min --price-max --rating-min`
- 后端选择: `--env-backend auto|api|playwright|mcp --mcp-endpoint`
- Prompt profile: `--prompt-profile`
- 评估: `--eval-suite --eval-limit --eval-seed --eval-output-dir`
- 可观测性: `--trace-output`

3. 条件变量层
- `configs/experiments/*.yaml` + `aces/experiments/control_variables.py`
- 支持 `product_overrides`, `position_overrides`, `global_overrides`（价格/标签/标题/评分/描述/位置标签注入/shuffle）。

---

## 6. 数据与实验工作流

### 6.1 数据流水线

统一入口: `manage_datasets.py`

子命令：

1. `expand-from-amazon`
2. `expand-by-keyword`
3. `download-ace`
4. `enrich`
5. `filter-reviews`
6. `filter-quality`
7. `list-sources`

数据目录分层（现状）：

- `datasets_40_work/raw`：扩展原始数据
- `datasets_40_work/enriched`：补齐评论/图片后数据（含质量报告）
- `datasets_expand_stage*`：阶段产物
- `datasets_unified`：当前默认服务使用集

### 6.2 实验与批处理

典型脚本：

- `run_query_0317.sh`
- `scripts/run_query_stability_10x.py`
- `scripts/run_visual_framework_and_price_highlight.py`
- `scripts/experiments/run_batch_experiments.sh`
- `scripts/experiments/extract_recommendations.sh`

常见产物目录：

- `logs/`
- `experiment_results/`
- `experiment_results/prompt_eval/`（包含 JSON + MD 汇总）

---

## 7. 测试现状（已本地核验）

执行命令：

- `./.venv_aces/bin/python -m pytest --collect-only -q`
- `./.venv_aces/bin/python -m pytest -q`

结果（本次）：

1. 收集：26 tests
2. 执行：`1 failed, 19 passed, 6 skipped`
3. 唯一失败：
- `tests/test_ui_structure.py::test_search_template_has_sidebar_and_result_list`
- 原因：测试断言 `search.html` 包含 `catalog-sidebar`，模板中不存在该字符串。

其余情况：

- legacy 集成与 llamaindex 重型测试为 `skip`（符合当前仓库策略）
- 编排层、工具解析、prompt profile、指标计算、CLI 兼容等核心单测基本可通过

---

## 8. 文档质量评估

优点：

1. 文档体系完整（总览、quickstart、guide、reference、architecture、migration、baseline）
2. 命令示例覆盖到服务、agent、数据管理、实验运行

主要问题：

1. 文档与仓库现状存在漂移
- 例：`docs/reference/DATA_SOURCES.md` 中 `datasets_unified` 列表与当前目录不一致
- 例：`FAQ.md` 对 `datasets_unified` 规模的描述与实际数据规模不一致

2. 架构历史文档存在“旧逻辑叙述”与当前 Tool-First 实现并存，易增加新成员理解成本

---

## 9. 风险与改进建议

### P0（高优先）

1. 修复 UI 结构测试与模板不一致
- 要么更新 `web/templates/search.html`
- 要么更新 `tests/test_ui_structure.py` 断言

2. 对齐文档与真实数据状态
- 更新 `DATA_SOURCES.md`、`FAQ.md` 中 `datasets_unified` 描述

### P1（中优先）

1. 增强可复现性规范
- 固定依赖版本（当前多为 `>=`）
- 固定 eval seed 与条件文件 seed
- 统一记录“数据版本 + 实验参数快照”

2. 补齐自动化覆盖
- `start_web_server.py` 路由与条件注入回归
- `manage_datasets.py` 子命令测试
- `scripts/` 批处理链路冒烟测试

### P2（中低优先）

1. 进一步收敛入口与路径
- 保留主入口 `run_browser_agent.py`
- 清晰标记 legacy/兼容脚本的用途和边界

2. 统一环境抽象认知
- 解释 `ShoppingEnv` 与 `MarketplaceProvider` 两套抽象关系，降低扩展时心智负担

---

## 10. 新成员上手建议（推荐顺序）

1. 阅读 `docs/README.md`（文档导航）
2. 走一遍 `docs/guides/QUICKSTART.md`
3. 启服务：`./aces_ctl.sh start`
4. 跑一次最小闭环：
   - `python run_browser_agent.py --query "running watch" --perception verbal --once`
5. 查看 viewer/log/trace：
   - `http://localhost:5000/viewer`
   - `logs/` / `--trace-output`
6. 再进入条件实验与批量评估

---

## 11. 关键文件索引（便于二次审阅）

- `README.md`
- `FAQ.md`
- `run_browser_agent.py`
- `start_web_server.py`
- `aces_ctl.sh`
- `manage_datasets.py`
- `aces/core/protocols.py`
- `aces/agents/base_agent.py`
- `aces/orchestration/agent_orchestrator.py`
- `aces/environments/api_shopping_env.py`
- `aces/environments/browser_shopping_env.py`
- `aces/environments/mcp_shopping_env.py`
- `aces/environments/llamaindex_marketplace.py`
- `aces/tools/shopping_browser_tools.py`
- `aces/experiments/control_variables.py`
- `tests/test_ui_structure.py`
- `docs/reference/DATA_SOURCES.md`
- `docs/guides/EXPERIMENT_GUIDE.md`

---

## 12. 结论

ACES-v2 当前已形成“Tool-First + Orchestrator 统一循环 + 多环境后端 + 实验条件注入”的可运行研究框架，核心能力完整，且命令与模块化设计可支撑持续实验迭代。当前短板主要集中在“文档与现实漂移、UI 结构测试回归、数据与实验可复现标准化、自动化覆盖深度”。如果优先完成 P0/P1 项目，框架稳定性和团队协作效率会显著提升。
