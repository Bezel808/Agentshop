# ACES-v2 Agent 逻辑拆解与现代化重构建议

本文档完整拆解 ACES-v2 中 Agent 与网页互动的逻辑，并给出更现代的 Agent 框架建议。

---

## 一、当前 Agent-网页互动逻辑拆解

### 1.1 双轨架构概览

ACES-v2 存在**两套并行的执行路径**，它们共享部分组件（ComposableAgent、Perception、LLM）但互不统一：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ACES-v2 双轨架构                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【路径 A】run_browser_agent.py (LiveBrowserAgent)                        │
│  ├─ 自建 ReAct 循环，不用 agent.act()                                      │
│  ├─ tools=[] 空列表，ComposableAgent 仅作 LLM+Perception 容器               │
│  ├─ Visual: Playwright 截图 → VLM 自由文本 → 正则解析动作                   │
│  └─ Verbal: HTTP API → 文本格式化 → LLM 自由文本 → 正则解析动作              │
│                                                                         │
│  【路径 B】aces/experiments/runner.py (StandardExperimentRunner)           │
│  ├─ observation → agent.act() → tool.execute() → observation             │
│  ├─ 使用 SearchTool / AddToCartTool / ViewProductDetailsTool              │
│  ├─ MarketplaceAdapter 封装 offline/online/llamaindex                     │
│  └─ 有 intervention、metrics、trajectory 日志                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 路径 A：LiveBrowserAgent 网页互动详解

#### Visual 模式（截图 + VLM）

```
1. 初始化
   init_browser() → Playwright Chromium headless, viewport 1280x800

2. 搜索阶段
   _build_search_url(keywords, page, price_min, price_max, rating_min)
   → navigate_and_capture(url)
      ├─ page.goto(url, wait_until="networkidle")
      ├─ time.sleep(1.5)
      └─ page.screenshot(type="png", full_page=True)
   → perception.encode(screenshot) → Observation(modalilty="visual")
   → LLM.generate([system, observation, prompt_search])
   → 解析 LLM 自由文本输出：
      - 数字 1~N → 点击第 N 个商品
      - "next" → page_num += 1，重新导航
      - "prev" → page_num -= 1
      - "filter price MIN MAX" → 更新 price_min/price_max，重置 page_num
      - "filter rating N" → 更新 rating_min
      - "recommend N" → 从 viewed 列表取第 N 个，做最终推荐

3. 详情阶段
   navigate_and_capture(detail_url)
   → get_description_from_detail_page() 用 page.locator(".detail-description .text")
   → LLM 判断 recommend / back
   → back → 返回搜索页；recommend → _do_final_recommend()
```

**DOM 解析方式：**
- `get_product_detail_links_from_search()`: `page.evaluate()` 取 `a[href^="/product/"]` 的 href，解析 product_id
- `get_description_from_detail_page()`: `page.locator(".detail-description .text")` 取描述
- 无 MCP browser tools，全部用 Playwright 原生 API

#### Verbal 模式（HTTP API）

```
1. _api_search(keywords, page, price_min, price_max, rating_min)
   → GET /api/search?q=...&page_size=8&page=...
   → 返回 {products, page, total_pages}

2. _format_product_list(products) → 文本列表
   → LLM.generate() 选择：数字 / next / prev / filter price / filter rating

3. 选择商品后：_api_product_detail(product_id)
   → GET /api/product/{id}
   → _format_product_detail(detail) → 文本
   → LLM 最终推荐判断
```

**特点：** 无浏览器，纯 HTTP + 文本。

### 1.3 路径 B：Experiment Runner 网页互动

```
1. MarketplaceAdapter 封装 MarketplaceProvider
   - OfflineMarketplace / OnlineMarketplace / LlamaIndexMarketplace

2. observation = agent.perception.encode(page_state)
   - page_state 来自 provider.get_page_state()
   - 可能是 products（verbal）或 screenshot（visual）

3. action = agent.act(observation)
   - ComposableAgent 内部：llm.generate(messages, tools)
   - 期望 LLM 返回 tool_call 结构

4. tool_result = tool.execute(action.parameters)
   - SearchTool → marketplace.search_products()
   - ViewProductDetailsTool → marketplace.get_product_details()
   - AddToCartTool → marketplace.add_to_cart()

5. 循环直到 add_to_cart 成功
```

**特点：** 工具驱动、结构化 Action，但 Browser Agent 并未使用此路径。

### 1.4 未使用的组件

| 组件 | 位置 | 状态 |
|------|------|------|
| BrowserNavigateTool | aces/tools/browser_tools.py | 封装 MCP cursor-ide-browser，未被 run_browser_agent.py 使用 |
| BrowserSnapshotTool | 同上 | 同上 |
| BrowserClickTool | 同上 | 同上 |
| BrowserTypeTool | 同上 | 同上 |
| SimpleBrowserController | 同上 | 同上，且 search_products/click_product 为 TODO |
| ComposableAgent.act() | aces/agents/base_agent.py | Browser Agent 只用了 llm + perception，未用 act() |

---

## 二、当前架构问题归纳

1. **双轨割裂**：Browser 路径与 Experiment 路径逻辑、工具、循环完全不同，难以复用和扩展。
2. **动作解析脆弱**：依赖正则和自由文本解析（`_extract_choice_number`、`"next" in al[:20]` 等），易受模型输出格式影响。
3. **无结构化 Tool Calling**：Visual/Verbal 都未用 LLM 的 function calling，而是 prompt + 正则。
4. **重复实现**：API 搜索、URL 构建、商品格式化等在 LiveBrowserAgent 内手写，与 Marketplace 逻辑重叠。
5. **MCP 能力闲置**：已有 MCP browser tools，但主流程用的是 Playwright 直连。
6. **状态管理分散**：`state` 字典、`decision_path`、`page_num` 等散落在 LiveBrowserAgent，缺乏统一状态机。

---

## 三、更现代的 Agent 框架建议

### 3.1 推荐方向：统一 + 工具化 + 状态机

目标：**单一 Agent 执行模型**，支持 Visual/Verbal 及 Browser/API 多种后端，通过 **结构化 Tool Calling** 和 **显式状态机** 替代自由文本解析。

### 3.2 方案 A：基于 LangGraph 的状态机（推荐）

**思路**：用 LangGraph 的图结构统一 Visual 与 Verbal 流程，节点为「观察 → 决策 → 执行」。

```
                    ┌──────────────────┐
                    │  QueryUnderstanding │
                    └─────────┬──────────┘
                              ▼
                    ┌──────────────────┐
         ┌──────────│   SearchPage     │◄──────────┐
         │          └─────────┬──────────┘          │
         │                    │                     │
         │         ┌──────────┼──────────┐          │
         │         ▼          ▼          ▼          │
         │  [SelectProduct] [NextPage] [Filter]     │
         │         │          │          │          │
         │         ▼          │          └──────────┤
         │  ┌──────────────┐  │                     │
         │  │ DetailPage   │  │                     │
         │  └──────┬───────┘  │                     │
         │         │          │                     │
         │    [Back] [Recommend]                    │
         │         │     │                          │
         │         └─────┴──────────────────────────┘
         │                    │
         └────────────────────┘
```

**核心改造点：**
1. **Tools 定义**：将「选商品」「翻页」「筛选」「返回」「推荐」等抽象为 Tool，用 JSON Schema 约束参数。
2. **Perception 作为节点输入**：每个节点接收 `Observation`，由 Perception 编码后交给 LLM。
3. **LLM Tool Calling**：使用 `tools=[...]` 让模型返回结构化 tool_call，不再做正则解析。
4. **Environment 抽象**：`BrowserEnv` 与 `APIEnv` 实现同一接口（navigate, get_screenshot, get_products），Agent 不关心具体实现。

**优点**：控制流清晰、可观测、易扩展、支持 human-in-the-loop（LangSmith）。

### 3.3 方案 B：纯 Tool-First 重构（不引入新框架）

在不引入 LangGraph 的前提下，将 LiveBrowserAgent 改造为「工具驱动」：

1. **定义 Shopping Actions 为 Tools**：
   ```python
   # 示例
   SelectProductTool(product_index: int)
   NextPageTool()
   PrevPageTool()
   FilterPriceTool(min: float, max: float)
   FilterRatingTool(min: float)
   ViewDetailTool(product_id: str)  # 或 product_index
   BackTool()
   RecommendTool(product_id: str)
   ```

2. **统一 Environment 接口**：
   ```python
   class ShoppingEnv(Protocol):
       def get_observation(self) -> Observation: ...
       def execute(self, action: Action) -> ToolResult: ...
   ```
   - `BrowserShoppingEnv`：用 Playwright 或 MCP
   - `APIShoppingEnv`：用 HTTP API

3. **用 ComposableAgent.act()**：
   - 将上述 Tools 注入 ComposableAgent
   - 主循环：`obs = env.get_observation()` → `action = agent.act(obs)` → `env.execute(action)`
   - 与 Experiment Runner 共用同一 `act()` 逻辑

4. **移除正则解析**：完全依赖 LLM 的 function calling 返回 `tool_name` + `parameters`。

**优点**：改动相对小，复用现有 ComposableAgent、BaseTool、Protocols。

### 3.4 方案 C：MCP-First 浏览器交互

将 Browser 操作完全交给 MCP `cursor-ide-browser`：

1. **使用 browser_tools.py**：BrowserNavigateTool、BrowserSnapshotTool、BrowserClickTool、BrowserTypeTool。
2. **高层 Shopping Tools**：在 MCP 之上封装：
   - `SearchAndSnapshotTool`：navigate 到 search URL + snapshot
   - `ClickProductByIndexTool`：从 snapshot 解析商品 ref，click
3. **Perception**：Snapshot 的 accessibility tree / 截图作为 Observation。
4. **统一到 Experiment Runner 风格**：`agent.act()` + tools。

**优点**：与 Cursor IDE 生态一致，可复用 MCP 的 lock/unlock、多标签等能力。**缺点**：依赖 MCP 服务可用性，iframe 等限制需接受。

---

## 四、推荐实施路径

### 阶段 1：统一执行模型（约 1–2 周）
1. 定义 `ShoppingAction` 枚举或 Tool 集合，覆盖当前所有动作。
2. 将 LiveBrowserAgent 的 `run_verbal` 和 `run_visual` 中的「动作解析」替换为「LLM Tool Calling」。
3. 抽离 `ShoppingEnv` 接口，实现 `BrowserEnv`（Playwright）和 `APIEnv`（requests）。

### 阶段 2：引入状态机（可选，约 1 周）
4. 若选 LangGraph：将 Search / Detail / Filter 等定义为节点，边为条件转移。
5. 若选方案 B：用简单状态变量（context: search | detail）驱动分支，逻辑集中到 `execute(action)`。

### 阶段 3：MCP 与 Experiment 对齐（约 1 周）
6. 完成 `SimpleBrowserController` 的 `search_products`、`click_product`。
7. 提供「MCP 模式」配置项，使 run_browser_agent 可选 Playwright 或 MCP。
8. 让 Experiment Runner 支持 web_renderer 类型的 marketplace，与 Browser Agent 共用 Environment。

---

## 五、关键代码改造示例

### 5.1 将动作抽象为 Tool（示例）

```python
# aces/tools/browser_shopping_tools.py

class SelectProductTool(BaseTool):
    name = "select_product"
    description = "Select product by index (1-based) to view details"
    input_schema = {
        "type": "object",
        "properties": {"index": {"type": "integer", "minimum": 1}},
        "required": ["index"]
    }
    
    def __init__(self, env: ShoppingEnv):
        self.env = env
    
    def _execute_impl(self, params):
        return self.env.select_product(params["index"])
```

### 5.2 用 Tool Calling 替代正则解析（示例）

```python
# 当前（run_visual 内）：
al = analysis.strip().lower()
if "next" in al[:50]: ...
elif re.search(r"filter\s+price\s+...", al): ...

# 改造后：
response = self.agent.llm.generate(messages, tools=[NextPageTool, FilterPriceTool, ...])
if response.content.get("tool_call"):
    action = Action(
        tool_name=response.content["tool_call"]["name"],
        parameters=response.content["tool_call"]["parameters"]
    )
    result = self.tools[action.tool_name].execute(action.parameters)
```

### 5.3 统一主循环（示例）

```python
def run(self):
    keywords = self.extract_search_keywords()
    obs = self.env.reset(keywords)
    
    for step in range(self.max_steps):
        action = self.agent.act(obs)
        if action.tool_name == "recommend":
            break
        result = self.env.execute(action)
        obs = self.env.get_observation()
```

---

## 六、总结

| 维度 | 现状 | 建议 |
|------|------|------|
| 执行模型 | 双轨（Browser 自建循环 vs Experiment agent.act） | 统一为 agent.act + env.execute |
| 动作表示 | 自由文本 + 正则解析 | 结构化 Tool Calling |
| 浏览器 | Playwright 直连，MCP 未用 | 可选 Playwright / MCP，接口统一 |
| 状态 | 分散在 LiveBrowserAgent | 显式状态机或 ShoppingEnv 内部状态 |
| 扩展性 | 新动作需改 prompt+正则 | 新动作 = 新 Tool，改 schema 即可 |

优先推荐：**方案 B（Tool-First）** 作为第一步，在不引入新依赖的前提下完成逻辑统一；若后续需要更强控制流与可观测性，再引入 **方案 A（LangGraph）**。
