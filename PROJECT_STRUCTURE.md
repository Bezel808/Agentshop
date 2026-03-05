# ACES-v2 项目结构

## 目录一览

```
ACES-v2/
├── aces/                    # 核心库
│   ├── agents/              # ComposableAgent
│   ├── config/              # 路径、配置
│   ├── core/                # 协议 (Observation, Action, ToolResult)
│   ├── data/                # 数据加载
│   ├── environments/        # Marketplace (offline, llamaindex, web_renderer, online)
│   ├── experiments/         # 实验条件、干预、指标
│   ├── llm_backends/        # OpenAI, Qwen, DeepSeek
│   ├── perception/          # Verbal / Visual 感知
│   └── tools/               # SearchTool, AddToCartTool 等
│
├── configs/
│   ├── environments/        # Marketplace 配置
│   └── experiments/         # 实验条件 YAML
│
├── web/                     # 前端
│   ├── templates/           # search, product_detail, live_viewer
│   └── static/css/
│
├── datasets_unified/        # 主数据集 (12 类目 JSON)
│
├── aces_ctl.sh              # 控制脚本
├── start_web_server.py      # Web 服务
├── run_browser_agent.py     # Agent 唯一入口（VLM 理解需求 + verbal/visual + 推荐一商品）
├── manage_datasets.py       # 数据集管理（统一入口）
├── test_*.py                # 测试
│
├── README.md
├── DATA_SOURCES.md          # 数据来源
├── CONDITION_REFERENCE.md   # 实验条件说明
├── EXPERIMENT_GUIDE.md
├── QUICKSTART.md
└── examples/                # 示例脚本
```

## 入口脚本

| 脚本 | 用途 |
|------|------|
| `./aces_ctl.sh start` | 启动 Web 服务 |
| `start_web_server.py` | 直接启动 Web |
| `run_browser_agent.py` | Agent 唯一入口：VLM 理解需求 + verbal/visual + 推荐一商品 |
| `manage_datasets.py` | 数据集扩充/下载/补齐 |

## 数据集

- **datasets_unified/**：主数据集，每类目一个 JSON
- 可用 `manage_datasets.py list-sources` 查看
