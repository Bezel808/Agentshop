# ACES-v2

ACES-v2 是一个面向电商选品 Agent 的实验与评测框架，支持两种感知路径：
- `visual`：基于网页截图进行视觉决策（Playwright 浏览）
- `verbal`：基于结构化文本页面信息进行决策（API 驱动）

项目同时提供：
- 商品搜索网站（`/search`、`/product/{id}`）
- Agent 实时 Viewer（`/viewer`）
- 数据集管理页面（`/dataset-admin`）

当前推荐使用的数据集目录：`datasets_40_work/enriched`（7 类，约 1190 商品）。

---

## 1. 项目整体框架

### 1.1 核心模块

- `start_web_server.py`
  - FastAPI 服务入口
  - 提供搜索/详情/API/WebSocket/Viewer/Dataset-Admin 页面
- `run_browser_agent.py`
  - Agent 主入口（单次运行、评测运行）
  - 统一 orchestrator + 工具调用流程
- `aces/environments/`
  - `browser_shopping_env.py`：visual 模式环境
  - `api_shopping_env.py`：verbal 模式环境
  - `llamaindex_marketplace.py`：检索与召回
- `aces/tools/`
  - `shopping_browser_tools.py`：`select_product` / `next_page` / `back` / `recommend` 等核心工具
- `manage_datasets.py`
  - 数据集扩充、清洗、过滤、增强

### 1.2 Agent 运行流程（高层）

1. 解析用户 query（包含商品意图与预算范围）
2. 进入搜索页，按需应用价格/评分过滤
3. 在列表页和详情页间反复比较
4. 调用 `recommend()` 输出最终推荐
5. Viewer 实时显示思考、动作、截图与推荐结果

---

## 2. 目录结构（重点）

```text
ACES-v2/
├── aces/                        # 核心框架（agent / env / tool / llm backend）
├── web/                         # 前端模板与静态资源
├── datasets_40_work/enriched/   # 当前主商品库（推荐默认）
├── start_web_server.py          # Web 服务入口
├── run_browser_agent.py         # Agent 运行入口
├── manage_datasets.py           # 数据集管理工具
└── aces_ctl.sh                  # 一站式启动/停止/隧道/实验控制脚本
```

---

## 3. 环境准备

### 3.1 Python 与依赖

- Python `>=3.10`

```bash
pip install -e "[all]"
```

### 3.2 环境变量（`.env`）

至少配置一个可用模型 Key：

```bash
QWEN_API_KEY=xxx
# KIMI_API_KEY=xxx
# OPENAI_API_KEY=xxx
```

Viewer 鉴权（建议开启）：

```bash
ACES_VIEWER_TOKEN=your_token
```

---

## 4. 启动与使用

### 4.1 启动 Web（推荐）

注意：当前数据集建议显式指定为 `datasets_40_work/enriched`。

```bash
python start_web_server.py --port 5000 --datasets-dir datasets_40_work/enriched
```

或使用控制脚本（建议先设置）：

```bash
export ACES_DATASETS=datasets_40_work/enriched
./aces_ctl.sh start
```

### 4.2 页面入口

- 搜索页：`http://localhost:5000/search?q=&page=1&page_size=8`
- Viewer：`http://localhost:5000/viewer?token=<ACES_VIEWER_TOKEN>`
- Dataset Admin：`http://localhost:5000/dataset-admin?token=<ACES_VIEWER_TOKEN>`
- 健康检查：`http://localhost:5000/health`

### 4.3 运行 Agent（CLI）

```bash
# visual 模式
python run_browser_agent.py \
  --llm qwen \
  --perception visual \
  --query "smart watch price between 55 and 59.99" \
  --server http://127.0.0.1:5000 \
  --max-steps 45 \
  --once

# verbal 模式（可选：让 verbal 也用 VLM）
python run_browser_agent.py \
  --llm qwen \
  --perception verbal \
  --verbal-use-vlm \
  --query "usb_flash_drive price between 7.20 and 12.20" \
  --server http://127.0.0.1:5000 \
  --max-steps 45 \
  --once
```

---

## 5. 数据集管理

### 6.1 查看与扩充

```bash
python manage_datasets.py list-sources
python manage_datasets.py expand-by-keyword -k smart_watch,bluetooth_speaker -n 80 -o datasets_new
```

### 6.2 清洗与过滤（示例）

```bash
python manage_datasets.py enrich --input-dir datasets_new --output-dir datasets_new --min-images 3 --min-reviews 5
python manage_datasets.py filter-reviews --input-dir datasets_new --output-dir datasets_new --min-images 3 --min-reviews 5 -n 40
```

---

## 6. 常见运行建议

- 长实验建议使用 `tmux` 或 `nohup` 后台执行。
- 当前仓库以“可运行主链路”为主，默认不包含历史实验分析脚本与分析产物。
- Viewer 如果无法访问，先检查：
  - 服务是否健康：`/health`
  - `ACES_VIEWER_TOKEN` 是否配置且 URL 附带 token
- 若外网临时 tunnel 失效，需要重新拉起 `cloudflared/localtunnel`。
- 若想确保网页与实验脚本行为一致，请对齐以下参数：
  - `llm`
  - `max_steps`
  - `verbal_use_vlm`
  - 相同 query 文本

---

## 7. 快速排障

```bash
# 服务状态
./aces_ctl.sh status

# 实时日志
./aces_ctl.sh log

# 健康检查
curl -s http://127.0.0.1:5000/health
```

---

## 8. 版本与协作

建议每次实验前固定并记录：
- 当前 git commit hash
- 数据集目录与数据文件 hash
- 关键参数（llm / max_steps / verbal_use_vlm / timeout）

这样可以保证网页试跑、脚本批跑和复现实验的一致性。
