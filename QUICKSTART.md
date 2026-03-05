# ACES-v2 快速开始

精简版的常用命令与流程。**完整实验说明见 [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)**；一键启动可用 `./aces_ctl.sh start`。

---

## 📋 环境准备

```bash
cd /home/zongze/research/ACES-v2

# 基础依赖
pip install -e .

# Browser Agent 需要
pip install playwright
playwright install chromium
```

---

## 🌐 启动 Web 服务器

```bash
# 推荐：一键启动（后台，默认端口 5000）
./aces_ctl.sh start

# 或手动
python start_web_server.py \
  --host 0.0.0.0 \
  --port 5000 \
  --datasets-dir datasets_unified
```

访问：
- 搜索：`http://<服务器IP>:5000/search?q=mousepad`
- 查看器：`http://<服务器IP>:5000/viewer`
- 健康检查：`http://<服务器IP>:5000/health`

---

## 🤖 运行 Agent（唯一入口）

```bash
export QWEN_API_KEY="your-qwen-key"

python -u run_browser_agent.py \
  --api-key "$QWEN_API_KEY" \
  --llm qwen \
  --perception visual \
  --query mousepad
```

- `--perception verbal`：文本模式（不启动浏览器）
- `--once`：分析一次后退出

---

## ⚙️ 实验条件配置（价格/描述/标签）

价格锚定、诱饵效应、标签框架等实验需在 YAML 中定义条件。  
**速查**：[CONDITION_REFERENCE.md](CONDITION_REFERENCE.md)

---

## 🐛 常见问题

- 端口占用：`python start_web_server.py --port 5001`
- RAG 太慢：`python start_web_server.py --simple-search`

---

更多细节请看 `README.md`。
