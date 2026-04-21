# ACES-v2 项目阶段汇报（2026-04-01）

## 1. 执行摘要
- 本阶段已完成从“基础 agent 试验框架”向“可运行、可观测、可复现实验平台”的升级，核心包括：统一工具调用编排、Verbal 路径鲁棒解析、预算约束链路、Top-N 推荐输出兼容、Viewer 推荐卡片化展示、以及数据集目录切换与ID去重治理。
- 稳定性实验已形成完整日志资产（同一 query 多轮重复运行 + U/B/H/A/S 多任务汇总），可用于后续回归对比与上线前验收。
- 当前主分支已有新增目录集成提交（`b5747fb`），同时工作区仍有一批进行中改造（尚未统一提交）。

## 2. 稳定性实验结果整理

## 2.1 结果文件位置
- 单 query 10x 稳定性（含 verbal/visual 明细）：
  - `logs/stability_runs/query_stability_20260323_111448/summary.json`
  - `logs/stability_runs/query_stability_20260323_111448/records.csv`
  - `logs/stability_runs/query_stability_20260323_111448/*_run*.log`
  - `logs/stability_runs/query_stability_20260323_111448/*_run*.jsonl`
- U/B/H/A/S 汇总：
  - `logs/stability_runs/UBHAS_20260323_121229/combined_summary.json`
  - `logs/stability_runs/UBHAS_20260323_121229/combined_summary.md`

## 2.2 单 query（smartwatch tracking）10x 结论
数据来源：`query_stability_20260323_111448/summary.json`

| 模式 | runs | success_rate | top1_ratio | unique_recommendations | normalized_entropy | avg_recommend_step | step_std |
|---|---:|---:|---:|---:|---:|---:|---:|
| verbal | 10 | 1.0 | 0.9 | 2 | 0.4690 | 8.9 | 1.5780 |
| visual | 10 | 1.0 | 0.5 | 3 | 0.8587 | 3.2 | 0.9798 |

解读：
- verbal 在该 query 上推荐更集中（`top1_ratio` 更高，熵更低），但决策步数更长。
- visual 在该 query 上决策更快，但候选分布更分散（稳定性偏弱）。

## 2.3 U/B/H/A/S 五类 query 汇总结论
数据来源：`UBHAS_20260323_121229/combined_summary.json`

| 维度 | verbal 均值 | visual 均值 |
|---|---:|---:|
| top1_ratio | 0.80 | 0.78 |
| avg_step | 8.68 | 3.86 |

补充观察：
- 两种模式整体稳定性接近（top1_ratio 差异小）。
- visual 平均决策步数明显更短，更适合低延时交互。
- 不同 query 的模式优劣会翻转（例如 H 类 verbal 更稳、U/A 类 visual 更稳），建议按 query 类型分档选策略。

## 2.4 早期到修复后的对比线索
在同 query 的历史目录中存在“0 推荐”阶段和“可稳定产出”阶段：
- 早期：`query_stability_20260323_110050/summary.json`（verbal/visual top1_ratio 均为 0）
- 修复后：`query_stability_20260323_111448/summary.json`（verbal 0.9，visual 0.5）

这说明后续链路修复（工具解析、终止收敛、推荐触发）对实验可用性有实质提升。

## 3. 近期项目更新总览

## 3.1 已提交更新（主分支）
- 最近提交：
  - `b5747fb`（2026-03-31）Add newly created project folders
  - `d22c629`（2026-02-28）可配置 visual/verbal 流程与多步浏览决策
- 说明：`b5747fb` 主要新增目录资产（文档、脚本、测试、数据目录、编排模块等）。

## 3.2 进行中的核心改造（工作区）
以下内容已在代码中可见，属于近期持续迭代重点：

1. Agent 编排与工具调用鲁棒性
- 引入统一编排模块：`aces/orchestration/`
- 结构化工具调用解析增强与错误恢复链路（含 `noop` 回退/自动恢复）

2. Verbal 链路修复与预算理解升级
- 预算语义不再仅靠规则；新增 LLM 结构化意图抽取：
  - `run_browser_agent.py` 中 `_extract_intent_with_llm`
- 预算与关键词强制分离：预算仅用于 `filter_price`，关键词剔除价格信息。
- 无预算匹配时可直接终止并返回“无符合预算商品”。

3. 推荐策略与兼容
- 运行时可配置推荐数量（Top-1/Top-N），并与工具层通过环境变量同步。
- 推荐输出保留 Top-1 兼容字段，同时可携带 `recommendations[]` 扩展结构。

4. 前端 Viewer 展示升级
- 推荐结果由单链接升级为卡片化渲染（支持 Top-N 列表、图片、价格、评分、详情跳转）。
- Run 结束但无推荐时，提供明确空态提示。

5. 数据与检索一致性治理
- 数据目录切换到：`datasets_40_work/enriched`
- 针对重复 product_id 做去重重命名（`__dupN`）并记录 `original_id`，降低搜索页与详情页映射错位风险。
- 索引 schema 版本升级以规避旧缓存污染。

6. 实验与可观测性
- 增加稳定性实验脚本：`scripts/run_query_stability_10x.py`
- 保留 trace/log/records/summary 全链路资产，便于回归诊断。

## 3.3 测试与质量资产
- 新增/扩展了 `tests/` 目录下多类测试：
  - 工具调用解析
  - 价格过滤单边语义
  - 推荐链路
  - 编排器
  - UI 结构
  - 数据唯一性
- 当前可形成“功能修复 -> 冒烟验证 -> 稳定性回归”的闭环。

## 4. 风险与待办
- 工作区存在大量未提交改动（包括历史文件删除/迁移），建议分批整理成原子提交，降低回滚成本。
- Top-1 与 Top-N 策略并存期间，需要在提示词和工具 schema 上持续保持一致，避免行为漂移。
- 建议建立固定 nightly 稳定性基准（至少 3 条代表 query × verbal/visual），持续追踪 `top1_ratio/entropy/step_std`。

## 5. 建议的下一步（可执行）
- 整理工作区改动为 3~5 个主题提交：
  - agent/orchestration
  - budget/filter/recommend
  - viewer/ui
  - dataset/index
  - tests/docs
- 对 `query_stability` 输出增加自动对比脚本，生成“相对上一版”的差异报告。
- 在 README 或 docs 增加“发布前验收清单”（联调脚本、最小回归集、线上 tunnel 检查项）。

## 6. 附：本次补充分析资产
- 五类商品价格分布图：
  - `analysis/price_histograms_5cats.png`
  - `analysis/price_histograms_5cats.pdf`
