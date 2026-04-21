# ACES-v2 常见问题 (FAQ)

---

## Q: 目前数据集大小 & 来源

**A:** 目前的数据源是：

从 **Amazon Reviews 2023** 扩展生成（README / DATA_SOURCES 有注明）

当前 `datasets_unified` 的实际数量：

| 品类 | 商品数 | 品类 | 商品数 |
|:---|---:|:---|---:|
| beauty_and_personal_care | 50 | office_products | 50 |
| cell_phones_and_accessories | 50 | pet_supplies | 50 |
| electronics | 50 | ski_jacket | 53 |
| health_and_household | 50 | sports_and_outdoors | 50 |
| home_and_kitchen | 50 | tools_and_home_improvement | 50 |
| mousepad | 31 | toys_and_games | 50 |

数据格式可进一步爬评论内容，结构示例见 `docs/reference/DATA_SOURCES.md`。

---

## Q: 目前 Web 页面给商品初步排序用的算法是什么？

**A:** 目前 Web 页（`/search`）的默认排序是**相关性排序（relevance）**，具体链路：

1. BM25 召回  
2. 向量语义检索召回  
3. RRF（Reciprocal Rank Fusion）融合  
4. Cross-Encoder 重排（`cross-encoder/ms-marco-MiniLM-L6-v2`）  
5. 取 top-k（页面上为 8）

因此页面顺序不是按价格/评分，而是按「查询相关性（混合检索 + 神经重排）」展示。

**补充：** 若用 `--simple-search` 启动（或 `ACES_SIMPLE_SEARCH=1 ./aces_ctl.sh start`），则退化为按文件名匹配数据集的简化逻辑，不走上述 RAG + rerank 流程。

---

## Q: 目前用户怎么把需求 query 传给 agent？

**A:** 通过命令行参数 `--query` 传入 agent。

例如：

```bash
python run_browser_agent.py --llm qwen --query "我想要一个美观的鼠标垫" --once
```

流程：

1. `--query` 作为**用户原始需求**传入 agent（`user_query`）  
2. agent 先用 LLM 做需求理解，提取搜索关键词  
3. 用提取出的关键词访问 `/search?q=...` 并继续后续决策  

因此 `--query` 是「需求输入」，不一定等于最终用于搜索的关键词。

---

## Q: Agent 怎么操作网页，何时停止？

**A:** 统一入口为 `run_browser_agent.py`，支持：

- **visual**：与网页 DOM + 截图交互（Playwright 截图给 VLM）
- **verbal**：与 JSON API 交互（`/api/search`、`/api/product/{id}`）
- 两者均通过同一 Web 服务（默认 5000 端口）

### 多步决策与停止条件

Agent 为 **ReAct 循环**，在搜索结果页与商品详情页之间自主决策：

- **搜索结果页**：可回复数字进入详情、`next` 下一页、`prev` 向前翻页、`filter price/rating` 筛选、或 `recommend N` 从已浏览商品中推荐
- **商品详情页**：可回复 `recommend`/`推荐` 做最终推荐，或 `back`/`返回` 回到搜索结果
- **停止条件**：做出 `recommend` 后结束；或达到内部最大步数（约 40 步）后停止

```bash
# 单次推荐后退出（适合批量实验）
python run_browser_agent.py --llm qwen --perception visual --query "我想要一个美观的鼠标垫" --once

# 不传 --once 则推荐完成后保持 viewer 连接，Ctrl+C 退出
python run_browser_agent.py --llm qwen --perception visual --query "我想要一个美观的鼠标垫"
```

---

## Q: Visual 和 Verbal 分别是怎么实现的？

**A:**

### (1) Visual

1. Query understanding：把用户需求提取为英文搜索关键词  
2. 打开网页 `GET /search?q=...`  
3. Playwright 截图搜索结果页，传给 VLM（Qwen VL / Kimi 视觉模型等）  
4. VLM 选一个商品序号，或 `next`/`prev` 翻页  
5. 打开该商品详情页，再截图（整页）+ 抽取 description  
6. VLM 基于详情截图和描述回复 `recommend` 或 `back`  
7. 若 `recommend`，则输出最终推荐并结束  

依赖：浏览器自动化 + `/viewer` 实时推送（可访问 `http://localhost:5000/viewer` 监看）

### (2) Verbal

1. 同样先做 query understanding  
2. 调 `GET /api/search?q=...&limit=8` 获取 top-8 结构化商品（与 Web 同检索管线）  
3. 把商品列表文本化后给 LLM，让其选一个或 `next`/`prev`  
4. 调 `GET /api/product/{id}` 获取完整详情文本  
5. LLM 基于详情文本做最终购买判断并给出推荐  

---

## Q: 支持哪些 LLM？

**A:** 当前支持三种 backend，通过 `--llm` 指定：

| backend | 环境变量 | 说明 |
|---------|----------|------|
| qwen | `QWEN_API_KEY` / `DASHSCOPE_API_KEY` | 默认，VL 用 qwen-vl-plus，verbal 用 qwen-plus |
| kimi | `KIMI_API_KEY` / `MOONSHOT_API_KEY` | 月之暗面，VL 用 moonshot-v1-8k-vision-preview |
| openai | `OPENAI_API_KEY` | gpt-4o |

---

## Q: 怎么运行？

**A:** 先启动服务，再运行 Agent。

### 1. 启动服务

```bash
cd /home/zongze/research/ACES-v2
./aces_ctl.sh start
```

或直接运行：

```bash
python start_web_server.py --port 5000 --datasets-dir datasets_unified
```

### 2. 运行 Agent

```bash
# Visual（看图）
python run_browser_agent.py --llm qwen --perception visual --query "我想要一个美观的鼠标垫" --once

# Verbal（读文本）
python run_browser_agent.py --llm qwen --perception verbal --query "我想要一个美观的鼠标垫" --once

# 使用 Kimi
python run_browser_agent.py --llm kimi --perception visual --query "我需要一个纯色的好看的鼠标垫" --once

# 或通过 aces_ctl 透传参数
./aces_ctl.sh run --llm qwen --perception visual --query "我想要一个美观的鼠标垫" --once
```

### 3. 批量实验

```bash
# 12 轮：kimi/qwen × visual/verbal × 3 rounds
./scripts/experiments/run_batch_experiments.sh

# 日志在 logs/，结束后自动提取推荐商品汇总
./scripts/experiments/extract_recommendations.sh
```

---

## 相关文档

- [docs/reference/DATA_SOURCES.md](docs/reference/DATA_SOURCES.md) — 数据来源与格式  
- [docs/reference/PROJECT_STRUCTURE.md](docs/reference/PROJECT_STRUCTURE.md) — 项目结构  
- [docs/guides/EXPERIMENT_GUIDE.md](docs/guides/EXPERIMENT_GUIDE.md) — 实验指南  
- [docs/guides/QUICKSTART.md](docs/guides/QUICKSTART.md) — 快速开始  
