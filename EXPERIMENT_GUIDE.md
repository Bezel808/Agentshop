# ACES-v2 实验使用说明

本文档说明如何启动服务、跑各类实验，以及远程访问与排错。

---

## 一、环境准备

```bash
cd /path/to/ACES-v2

# 安装依赖
pip install -e .

# Browser Agent（Playwright + 截图）需要
pip install playwright
playwright install chromium
```

**API Key（按实验需要设置）：**

- Qwen：`export QWEN_API_KEY="your-key"` 或 `DASHSCOPE_API_KEY`
- OpenAI：`export OPENAI_API_KEY="your-key"`
- DeepSeek：`export DEEPSEEK_API_KEY="your-key"`

---

## 二、启动 Web 服务（做实验前先启动）

### 方式 A：用启动脚本（推荐）

```bash
./aces_ctl.sh start
```

- 默认：端口 **5000**，数据集 **datasets_unified**，后台运行。
- 首次启动会构建 LlamaIndex 索引，约 10–20 秒后可用。

**常用参数：**

| 参数 | 说明 |
|------|------|
| `./aces_ctl.sh stop` | 停止已运行的 Web 服务 |
| `./aces_ctl.sh status` | 查看是否在跑、健康检查 |
| `ACES_PORT=5001 ./aces_ctl.sh start` | 指定端口 |
| `ACES_DATASETS=datasets_extended/ace-bb ./aces_ctl.sh start` | 指定数据集目录 |
| `./aces_ctl.sh log` | 实时查看日志 |
| `ACES_SIMPLE_SEARCH=1 ./aces_ctl.sh start` | 不用 LlamaIndex（启动快） |

### 方式 B：直接运行 Python

```bash
python start_web_server.py --host 0.0.0.0 --port 5000 --datasets-dir datasets_unified
```

**访问地址（端口以实际为准）：**

| 页面 | URL |
|------|-----|
| 搜索首页 | `http://<服务器IP>:5000/` 或 `http://localhost:5000/` |
| 搜索示例 | `http://localhost:5000/search?q=mousepad` |
| 商品详情 | 在搜索页点击商品进入，或 `/product/<product_id>?q=...` |
| 实时查看器 | `http://localhost:5000/viewer` |
| 健康检查 | `http://localhost:5000/health` |

---

## 三、实验类型与命令

以下命令均需在 **ACES-v2 项目根目录** 下执行；除「仅搜索页」外，其余实验前请先 **启动 Web 服务**（`./aces_ctl.sh start`）。

### 1. 仅用搜索页 / 商品详情（无需 Agent）

1. 启动服务：`./aces_ctl.sh start`
2. 浏览器打开：`http://localhost:5000/search?q=mousepad`（或本机 IP:5000）
3. 点击任意商品进入详情页，可查看 **description** 等完整信息。

### 2. Browser Agent（VLM 看网页截图并决策）

Agent 用 Playwright 打开搜索页、截图，并推送到 **/viewer**；适合本地或远程看实时过程。

```bash
# 先启动 Web：./aces_ctl.sh start

export QWEN_API_KEY="your-key"
python -u run_browser_agent.py \
  --api-key "$QWEN_API_KEY" \
  --llm qwen \
  --query mousepad
```

- 看实时画面：浏览器打开 `http://localhost:5000/viewer`（若在远程服务器跑，见下文「远程访问」）。
- 可选：`--server http://localhost:5000`、`--once`（分析一次后退出）。

### 3. 标准实验（run_experiment.py）

Agent + Marketplace 实验，支持 verbal/visual、offline/llamaindex 等。

**简单模式（不调 LLM，测流程）：**

```bash
python run_experiment.py --mode simple --query mousepad --trials 5
```

**VLM 实验模式（默认）：**

```bash
# 使用环境变量中的 QWEN_API_KEY，VLM 看图决策
python run_experiment.py --query mousepad

# 指定 API key
python run_experiment.py --llm qwen --perception visual --api-key "$QWEN_API_KEY" \
  --query mousepad --marketplace offline --datasets-dir datasets_unified
```

常用参数：`--perception visual|verbal`、`--marketplace offline|llamaindex`、`--datasets-dir`、`--limit`、`--output-dir`、`--verbose`。

### 4. VLM 排名实验（run_ranking_experiment.py）

对同一 query 多次让 VLM 对商品排序，统计一致性等。

```bash
# 需指定数据文件与 API key，具体参数见脚本 --help
python run_ranking_experiment.py --api-key "$QWEN_API_KEY" --llm qwen ...
```

### 5. VLM 购物 Agent 完整流程（run_vlm_shopping_agent.py）

多轮：理解 query → 搜索 → 分析截图 → 做出购买决策并写日志。

```bash
# 需先启动 Web；API key 与数据路径见脚本 --help
python run_vlm_shopping_agent.py ...
```

### 6. A/B 测试示例（YAML 配置）

```bash
python run_experiment.py --config configs/experiments/visual_vs_verbal_example.yaml
```

（若项目已支持从 YAML 读配置；否则以当前 `run_experiment.py` 的 CLI 为准。）

---

## 四、远程服务器 + 本机浏览器

在**服务器**上跑 Web 和 Agent，在**本机**用浏览器访问：

1. **SSH 端口转发**（在本机执行）：

   ```bash
   ssh -L 5000:localhost:5000 用户名@服务器IP -p 端口
   ```

2. 在服务器上启动服务：`./aces_ctl.sh start`（或 `python start_web_server.py ...`）。
3. 本机浏览器打开：
   - `http://localhost:5000/`（搜索）
   - `http://localhost:5000/viewer`（查看器）

这样所有请求通过 SSH 隧道到服务器的 5000 端口，无需改防火墙。

---

## 五、日志与结果位置

| 内容 | 位置 |
|------|------|
| Web 服务日志（脚本启动） | 默认 `tail -f /tmp/aces_web_server.log`（可由环境变量 `LOG_FILE` 修改） |
| run_experiment 输出 | 默认 `experiment_results/` |
| 其他 run_* 脚本 | 各脚本内指定的 log_dir / output_dir |

---

## 六、常见问题

- **端口被占用**  
  `./aces_ctl.sh start --port 5001` 或 `python start_web_server.py --port 5001`

- **启动慢 / RAG 建索引久**  
  可先用简单搜索：`./aces_ctl.sh start --simple-search` 或加 `--simple-search`

- **已有一份在跑，想重启**  
  `./aces_ctl.sh stop` 再 `./aces_ctl.sh start`

- **健康检查不通过**  
  `./aces_ctl.sh status` 看状态；`./aces_ctl.sh log` 看日志排查

- **Browser Agent 报错**  
  确认：1）Web 已启动且 `/health` 正常；2）`playwright install chromium` 已执行；3）API key 已设置且有效

---

## 七、脚本与文档索引

| 文件 | 说明 |
|------|------|
| `aces_ctl.sh` | 一键启动/停止/状态检查 Web 服务 |
| `start_web_server.py` | Web 服务入口（搜索、商品详情、viewer、/health） |
| `run_browser_agent.py` | Browser Agent（截图 + 推送到 viewer） |
| `run_experiment.py` | 标准 Agent-Marketplace 实验 |
| `run_ranking_experiment.py` | VLM 排名实验 |
| `run_vlm_shopping_agent.py` | VLM 多轮购物决策 |
| `QUICKSTART.md` | 最简命令速查 |
| `README.md` | 项目总览与结构 |

以上为当前实验流程的完整使用说明，后续若新增实验类型或参数，可在此文档中追加对应小节与命令示例。
