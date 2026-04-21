# ACES-v2 项目说明报告（深度版）

- 生成时间：2026-03-24
- 分析范围：`/home/zongze/research/ACES-v2` 全仓（代码、配置、脚本、文档、测试）
- 仓库分支：`main`
- 最近提交：`fb090ee`（`git log -n 1`）

---

## 1. 项目定位与核心目标

ACES-v2 是一个「电商选品 Agent 研究框架」，核心目标是：

1. 让 LLM/VLM 在电商搜索与详情浏览流程中进行多步决策。
2. 支持 `visual`（看图）与 `verbal`（读文本）两种感知路径。
3. 支持可控实验（价格锚定、标签操控、诱饵效应等）与可量化评估（轨迹与指标）。

在当前代码中，**主线可运行入口**是：

- `run_browser_agent.py`（Agent 主入口）
- `start_web_server.py`（Web + API 服务入口）
- `aces_ctl.sh`（服务与运行控制脚本）

---

## 2. 总体架构分层

### 2.1 代码层级（核心包 `aces/`）

1. 协议层（Contracts）
- `aces/core/protocols.py`
- 定义 `Observation` / `Action` / `ToolResult` / `Agent` / `LLMBackend` / `Tool` 等抽象契约。

2. Agent 组装层（Decision Core）
- `aces/agents/base_agent.py`
- `ComposableAgent` 负责将 `LLM + Perception + Tools` 绑定为统一决策体。

3. 编排层（Execution Loop）
- `aces/orchestration/agent_orchestrator.py`
- `AgentOrchestrator` 统一执行循环：`observe -> act -> execute -> stop`，并带重复动作/错误熔断。

4. 环境层（Environment/Marketplace）
- `aces/environments/*.py`
- 包含 `APIShoppingEnv`、`BrowserShoppingEnv`、`MCPShoppingEnv`（给主 Agent 使用）
- 以及 `OfflineMarketplace`、`LlamaIndexMarketplace`、`OnlineMarketplace`（偏底层 provider）。

5. 工具层（Tooling）
- `aces/tools/*.py`
- `shopping_browser_tools.py` 定义主循环使用的 7 个工具：
  `select_product / next_page / prev_page / filter_price / filter_rating / back / recommend`

6. 感知层（Perception）
- `aces/perception/visual_perception.py`
- `aces/perception/verbal_perception.py`

7. 模型后端层（LLM Backends）
- `aces/llm_backends/openai_backend.py`
- `qwen_backend.py` / `kimi_backend.py` / `deepseek_backend.py`

8. 实验层（Experiment System）
- `aces/experiments/*`
- 含轨迹协议、指标计算、条件变量（control variables）、实验 runner。

### 2.2 入口脚本层

1. `run_browser_agent.py`
- 实时运行 Agent，支持 visual/verbal/mcp。

2. `start_web_server.py`
- 提供页面与 API：`/search`、`/product/{id}`、`/api/search`、`/api/product/{id}`、`/viewer`、`/ws/viewer`。

3. `manage_datasets.py`
- 数据扩展、下载、补齐评论/图片、质量过滤。

4. `run_agent_dual_mode.py`
- 串行跑 verbal + visual，用于快速双模对比。

---

## 3. 关键调用链（从入口到执行）

## 3.1 Agent 主链路（`run_browser_agent.py`）

1. `main()`
- 解析 CLI 参数、加载 API Key、选择是否 eval-suite。

2. `LiveBrowserAgent.__init__()`
- 选择 LLM：`qwen/openai/kimi`
- 选择感知：`VisualPerception` 或 `VerbalPerception`
- 选择环境：
  - visual 默认 `BrowserShoppingEnv`
  - verbal 默认 `APIShoppingEnv`
  - 可选 `MCPShoppingEnv`
- 通过 `ToolFactory.create_shopping_browser_tools(env)` 注入 7 个工具。
- 构建 `ComposableAgent`，系统提示由 `_build_system_prompt(profile)` 生成。

3. `LiveBrowserAgent.run()`
- 先 `extract_search_keywords()`（用 LLM 把用户需求提炼成检索关键词）
- 再 `_run_unified(keywords)` 进入统一编排。

4. `_run_unified()` 内部
- 创建 `AgentOrchestrator`
- `get_initial_observation = env.reset(keywords)`
- `get_next_observation = env.get_observation`
- `execute_action = tool.execute(action.parameters)`
- `should_stop`: 成功调用 `recommend` 即停止。

5. Orchestrator 循环（`aces/orchestration/agent_orchestrator.py`）
- 拉观察 -> 让 agent 产出 action(s) -> 执行工具 -> 判断终止。
- 熔断机制：
  - `max_repeated_actions`（默认 6）
  - `max_errors`（默认 8）
  - `max_steps`（默认 40）

## 3.2 ComposableAgent 决策链（`aces/agents/base_agent.py`）

1. `act_batch()` / `act()`
- 校验 observation modality。
- observation 写入 `message_history`。
- 取所有工具 schema 给 LLM。
- 调 `llm.generate(messages, tools)`。

2. `_parse_actions()`
- 兼容多种结构：`tool_calls` / `tool_call` / `function_call` / JSON 文本块。
- 失败时可尝试文本启发解析（`_extract_action_from_text`）；否则退化 `noop`。

3. 输出 `Action` 列表给 Orchestrator。

## 3.3 环境执行链（以 API/Browser 为主）

1. API 模式（`aces/environments/api_shopping_env.py`）
- `reset()` 调 `/api/search` 拉第一页。
- `select_product()` 调 `/api/product/{id}`。
- `next_page/prev_page/filter_*` 更新内部状态并重新查询 API。
- `recommend` 从当前详情或已浏览列表产出 `product_id`。

2. Browser 模式（`aces/environments/browser_shopping_env.py`）
- `reset()` 启动 Playwright，访问 `/search`，截图并抽取商品链接。
- `select_product(index)` 打开详情页截图；`back()` 返回搜索页。
- 分页和筛选通过拼 URL 参数跳转。

3. MCP 模式（`aces/environments/mcp_shopping_env.py`）
- 视觉快照走 `SimpleBrowserController`（MCP browser_* 工具）
- 状态机与推荐账本复用 `APIShoppingEnv`。

---

## 4. Web 服务与接口说明

`start_web_server.py` 启动 FastAPI 应用，支持两种检索后端：

1. 默认：`LlamaIndexMarketplace`（Hybrid Retrieval + rerank）
2. `--simple-search`：`WebRendererMarketplace`（简化检索）

### 4.1 页面路由

- `GET /`：搜索首页
- `GET /search`：搜索结果（支持 page/page_size/price_min/price_max/rating_min/condition_name）
- `GET /product/{product_id}`：详情页
- `GET /viewer`：实时查看器

### 4.2 API 路由

- `GET /api/search`
- `GET /api/product/{product_id}`
- `POST /api/run-agent`
- `GET /api/agent-status`
- `POST /api/stop-agent`
- `POST /api/push`（向 viewer 推送日志/截图）
- `GET /health`

### 4.3 实验条件接入

`--condition-file` 加载 `ConditionSet`（YAML/JSON），并通过 `condition_name` 在 search/detail 阶段按条件改写商品属性。

---

## 5. 检索与数据路径

### 5.1 LlamaIndex 检索管线（`aces/environments/llamaindex_marketplace.py`）

检索流程：

1. 加载全部商品 JSON -> `Product`
2. 建向量索引（可缓存到 `.cache/llamaindex`）
3. BM25 + 向量召回
4. RRF 融合
5. CrossEncoder rerank（可开关）
6. 分页与价格/评分过滤

### 5.2 当前数据目录观测（本地扫描结果）

1. `datasets_unified/`
- 8 个类目文件，合计约 174 商品（每文件 10~40）
- 文件包含：`smart_watch`, `bluetooth_speaker`, `athletic_shoes_*`, `backpack_*`, `insulated_cup_*`

2. `datasets_expand_stage/`、`datasets_expand_stage2/`
- 存在更大规模扩展数据（百级到 400/类）。

3. `manage_datasets.py`
- 子命令：`expand-from-amazon`、`expand-by-keyword`、`download-ace`、`enrich`、`filter-quality`、`filter-reviews`、`list-sources`。

---

## 6. 外部依赖与系统接口

### 6.1 关键三方依赖

1. LLM API
- OpenAI 兼容接口（OpenAI/DeepSeek/Qwen/Kimi）
- 依赖环境变量：`OPENAI_API_KEY` / `QWEN_API_KEY` / `KIMI_API_KEY` 等。

2. 浏览器自动化
- `playwright`（visual 模式核心）

3. Web 服务
- `fastapi` + `uvicorn` + `jinja2`

4. 检索增强
- `llama-index`、`sentence-transformers`、`huggingface embedding`

### 6.2 配置与启动入口

1. 一键控制：`aces_ctl.sh`
- `start/stop/status/log/run/health/datasets/conditions`

2. Agent CLI：`run_browser_agent.py --help`

3. Web CLI：`start_web_server.py --help`

4. 双模脚本：`run_agent_dual_mode.py`

---

## 7. 实验系统（研究向）

### 7.1 Protocol + Runner

- `aces/experiments/protocols.py` 定义轨迹、步骤、指标协议。
- `aces/experiments/runner.py` 的 `StandardExperimentRunner` 使用 `AgentOrchestrator` 统一记录：
  - observation
  - thought
  - action
  - tool_result
  - termination

### 7.2 条件变量系统

- `aces/experiments/control_variables.py`
- 支持：
  - 价格控制（固定值/倍率/噪声/原价锚）
  - 标签操控（sponsored/best_seller/overall_pick/custom_badges）
  - 标题操控（prefix/suffix/replace/fixed）
  - 位置覆写与随机打乱

### 7.3 指标系统

- `aces/experiments/metrics.py` 提供：
  - 研究指标：`selected_product_rank`、`price_sensitivity`
  - 工程指标：`invalid_toolcall_rate`、`retry_rate`、`env_error_rate`、`steps_to_success`

---

## 8. 测试现状（本地执行结果）

在虚拟环境运行：

- 命令：`./.venv_aces/bin/python -m pytest -q`
- 结果：`19 passed, 6 skipped, 1 failed`
- 失败项：`tests/test_ui_structure.py::test_search_template_has_sidebar_and_result_list`
- 失败原因：`search.html` 未包含测试期望的 `catalog-sidebar` 字符串（UI 结构与测试断言不一致）。

这意味着当前分支整体功能测试大体可用，但 UI 结构基线断言已漂移。

---

## 9. 发现的技术债与风险清单

### 9.1 高优先级

1. 文档与实现存在漂移
- 典型表现：`docs/AGENT_REFACTORING.md` 描述的旧路径与当前 Tool-First 主线不完全一致。
- 风险：新成员误判架构，影响开发效率。

2. UI 测试基线漂移
- `search.html` 与 `tests/test_ui_structure.py` 期望不匹配。
- 风险：CI 失败或前端改动无法稳定回归。

3. 数据文档与实际数据目录不一致
- `docs/reference/DATA_SOURCES.md` 中“现有文件列表”与当前 `datasets_unified` 现实目录不完全一致。
- 风险：实验复现与数据解释偏差。

### 9.2 中优先级

4. 环境层有并存模型（Provider 与 ShoppingEnv）
- `Offline/LlamaIndex/Online` 与 `API/Browser/MCP` 是两套不同抽象层并行。
- 风险：扩展新能力时改动面大，容易重复实现。

5. LLM 输出兼容逻辑较多
- `ComposableAgent._parse_actions()` 兼容多协议 + 文本启发式。
- 风险：模型行为漂移时出现隐式回退，问题定位成本高。

6. 可选依赖较重
- RAG、Playwright、Embedding、CrossEncoder 均重，运行前置条件复杂。
- 风险：环境配置失败率高，影响试验稳定性。

### 9.3 低优先级

7. 部分 legacy 脚本/测试仅保留参考
- 多个 `*_legacy` 测试默认 skip。
- 风险较低，但应明确“废弃/保留”的生命周期策略。

---

## 10. 建议的改进优先级（落地导向）

1. 先修测试基线与文档漂移
- 更新 `test_ui_structure.py` 或补齐模板 class，保证 CI 绿灯。
- 同步 `AGENT_REFACTORING.md`、`DATA_SOURCES.md` 到当前事实。

2. 收敛运行抽象
- 明确主线是 `ShoppingEnv + Tool-First` 还是 `MarketplaceProvider + Adapter`，给出统一扩展指南。

3. 建立“运行前检查”脚本
- 一次性检查 API keys、Playwright、RAG 模型依赖、数据目录，降低新环境失败概率。

4. 增加端到端冒烟测试
- 以 mock LLM 或固定 trace 方式覆盖 `/api/search -> run_browser_agent -> recommend` 全链路。

---

## 11. 结论

ACES-v2 当前已经形成可运行的完整闭环：

- Web 服务（搜索/详情/API/viewer）
- Agent 决策（Tool-First + Orchestrator）
- 双感知执行（visual/verbal）
- 实验控制与指标评估（condition + metrics）

从工程成熟度看，主线能力具备；当前主要短板不在“功能缺失”，而在**文档/测试/数据说明与实际代码的同步治理**。把这一层补齐后，项目会更适合持续迭代和多人协作。

