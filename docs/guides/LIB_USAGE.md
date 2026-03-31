# ACES-v2 使用说明（Library / CLI）

## 1. 环境准备

```bash
cd /home/zongze/research/ACES-v2
python3 -m venv .venv_aces
source .venv_aces/bin/activate
pip install -e .
```

配置 API Key（任选其一）：

```bash
export QWEN_API_KEY=...
export KIMI_API_KEY=...
export OPENAI_API_KEY=...
```

或使用项目根目录 `.env`。

## 2. 启动 Web 服务

推荐先用简单检索模式（启动快）：

```bash
ACES_SIMPLE_SEARCH=1 ./aces_ctl.sh start
./aces_ctl.sh status
```

常用页面：

- 搜索页：`http://localhost:5000/`
- Viewer：`http://localhost:5000/viewer`
- 健康检查：`http://localhost:5000/health`

停止服务：

```bash
./aces_ctl.sh stop
```

## 3. 运行单次 Agent

Verbal：

```bash
python run_browser_agent.py \
  --llm qwen \
  --perception verbal \
  --query "I want the smartwatch to accurately track steps, heart rate, calories burned, and workout duration." \
  --server http://localhost:5000 \
  --once
```

Visual：

```bash
python run_browser_agent.py \
  --llm qwen \
  --perception visual \
  --query "I want the smartwatch to accurately track steps, heart rate, calories burned, and workout duration." \
  --server http://localhost:5000 \
  --once
```

## 4. 重复运行稳定性实验（同一 Query 连跑 10 次）

脚本路径：

- `scripts/run_query_stability_10x.py`

示例：

```bash
python scripts/run_query_stability_10x.py \
  --query "I want the smartwatch to accurately track steps, heart rate, calories burned, and workout duration." \
  --runs 10 \
  --llm qwen \
  --server http://localhost:5000 \
  --output-dir logs/stability_runs
```

输出：

- `summary.json`：稳定性汇总（top1 比例、熵、步数波动等）
- `records.csv`：每次运行明细（mode/run/product_id/step/exit_code）
- `*_run*.log` / `*_run*.jsonl`：原始日志和 trace

## 5. 结果解读建议

- `top1_ratio` 越高：模式越稳定（重复运行更容易推荐同一商品）
- `normalized_entropy` 越低：分布越集中、稳定性更强
- `step_std` 越低：决策路径长度更稳定

## 6. 常见问题

- 启动慢：先用 `ACES_SIMPLE_SEARCH=1`，避免向量索引初始化耗时。
- `No valid action found`：检查模型输出是否是结构化 tool call；必要时加 `--legacy-fallback`。
- 连接失败：确认 `API Key`、代理、`/health` 可访问。
