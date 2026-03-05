# ACES-v2 使用说明

本文档说明如何启动服务、运行 Agent，以及远程访问与排错。

---

## 一、环境准备

```bash
cd /path/to/ACES-v2

# 安装依赖
pip install -e ".[all]"

# Browser Agent（Playwright + 截图）需要
pip install playwright
playwright install chromium
```

**API Key：**

- Qwen：`export QWEN_API_KEY="your-key"` 或 `DASHSCOPE_API_KEY`
- OpenAI：`export OPENAI_API_KEY="your-key"`

---

## 二、启动 Web 服务（运行 Agent 前必先启动）

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
| `ACES_DATASETS=datasets_unified ./aces_ctl.sh start` | 指定数据集目录 |
| `./aces_ctl.sh log` | 实时查看日志 |
| `ACES_SIMPLE_SEARCH=1 ./aces_ctl.sh start` | 不用 LlamaIndex（启动快） |

### 方式 B：直接运行 Python

```bash
python start_web_server.py --host 0.0.0.0 --port 5000 --datasets-dir datasets_unified
```

**访问地址：**

| 页面 | URL |
|------|-----|
| 搜索首页 | `http://localhost:5000/` |
| 实时查看器 | `http://localhost:5000/viewer` |
| 健康检查 | `http://localhost:5000/health` |

---

## 三、运行 Agent

Agent 用 VLM 理解用户需求，支持 **verbal**（文本）和 **visual**（截图）两种感知模式，在多页 Marketplace 中选品，最终推荐一个商品。

```bash
# 先启动 Web：./aces_ctl.sh start

export QWEN_API_KEY="your-key"
python -u run_browser_agent.py \
  --api-key "$QWEN_API_KEY" \
  --llm qwen \
  --perception visual \
  --query mousepad
```

- 实时画面：浏览器打开 `http://localhost:5000/viewer`
- `--perception verbal`：用文本描述代替截图（不启动浏览器）
- `--once`：分析一次后退出
- `--server http://localhost:5000`：指定 Web 服务地址

通过 aces_ctl 运行：

```bash
./aces_ctl.sh run --query mousepad
./aces_ctl.sh run --query ski --perception verbal
```

---

## 四、远程服务器 + 本机浏览器

在**服务器**上跑 Web 和 Agent，在**本机**用浏览器访问：

1. **SSH 端口转发**（在本机执行）：

   ```bash
   ssh -L 5000:localhost:5000 用户名@服务器IP -p 端口
   ```

2. 服务器上：`./aces_ctl.sh start`，然后运行 Agent
3. 本机浏览器打开：`http://localhost:5000/viewer`

---

## 五、常见问题

- **端口被占用**  
  `ACES_PORT=5001 ./aces_ctl.sh start`

- **启动慢 / RAG 建索引久**  
  `ACES_SIMPLE_SEARCH=1 ./aces_ctl.sh start`

- **已有一份在跑，想重启**  
  `./aces_ctl.sh stop` 再 `./aces_ctl.sh start`

- **Browser Agent 报错**  
  确认：1）Web 已启动且 `/health` 正常；2）`playwright install chromium` 已执行；3）API key 已设置

---

## 六、脚本与文档索引

| 文件 | 说明 |
|------|------|
| `aces_ctl.sh` | 启动/停止 Web、运行 Agent |
| `start_web_server.py` | Web 服务入口 |
| `run_browser_agent.py` | Agent 唯一入口 |
| `QUICKSTART.md` | 最简命令速查 |
| `README.md` | 项目总览 |
