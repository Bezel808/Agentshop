# ACES-v2 重构迁移说明

## 目标
- 统一执行主循环到 `aces/orchestration/AgentOrchestrator`
- 标准化 Tool Calling：`tool_calls[]`（兼容旧 `tool_call`）
- 支持可插拔环境后端：`api` / `playwright` / `mcp`

## 兼容策略
- 保留原入口脚本：`run_browser_agent.py`、`run_agent_dual_mode.py`
- 保留旧单工具解析：通过 `--legacy-fallback` 显式开启
- 根目录 `test_*.py` 保留为转发入口，主测试目录迁移到 `tests/`

## 新增参数
- `run_browser_agent.py --env-backend {auto,api,playwright,mcp}`
- `run_browser_agent.py --trace-output <path>`
- `run_browser_agent.py --legacy-fallback`

## MCP 行为
- `--env-backend mcp` 需要 MCP caller 注入。
- 若未注入，将快速失败并提示切换 `api/playwright`。

## 基线快照
使用脚本记录 CLI 基线：

```bash
bash scripts/capture_baseline_snapshot.sh
```
