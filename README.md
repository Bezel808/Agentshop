# ACES-v2

电商购物 Agent 实验框架：VLM 理解用户需求，支持 verbal/visual 感知，在多页 Marketplace 中选品并推荐一个商品。

**Python**: >=3.10

---

## 安装

```bash
pip install -e ".[all]"
```

环境变量：`QWEN_API_KEY`（VLM）、`OPENAI_API_KEY`、`DEEPSEEK_API_KEY`

---

## 常用命令

### Web 服务

```bash
./aces_ctl.sh start              # 启动 (port 5000)
./aces_ctl.sh stop
./aces_ctl.sh status

# 或直接运行
python start_web_server.py --port 5000 --datasets-dir datasets_unified
```

### Agent（唯一入口）

```bash
# 需先启动 Web：./aces_ctl.sh start
python run_browser_agent.py --query mousepad
python run_browser_agent.py --query ski --perception verbal
python run_browser_agent.py --query "性价比高的鼠标垫" --llm qwen --perception visual

# 或通过 aces_ctl
./aces_ctl.sh run --query mousepad
```

### 数据集管理

```bash
python manage_datasets.py list-sources
python manage_datasets.py expand-from-amazon -c Electronics -n 50
python manage_datasets.py expand-by-keyword -k mousepad,ski
python manage_datasets.py enrich --input-dir datasets_unified --output-dir datasets_unified_multimodal
```

---

## 文档

- [FAQ.md](FAQ.md) — 常见问题（数据集、排序、Agent 操作等）
- [DATA_SOURCES.md](DATA_SOURCES.md) — 数据来源
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) — 项目结构
- [CONDITION_REFERENCE.md](CONDITION_REFERENCE.md) — 实验条件
- [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) — 使用指南
- [QUICKSTART.md](QUICKSTART.md) — 快速开始
