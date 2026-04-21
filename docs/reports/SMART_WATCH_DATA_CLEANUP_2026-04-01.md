# Smart Watch 商品库清洗与重扩充报告（2026-04-01）

## 1. 背景
- 问题：`smart_watch` 商品库混入大量手表带、保护壳、贴膜、充电底座等配件。
- 目标：清洗非手表本体条目，并重建 smart_watch 数据集。

## 2. 执行动作
- 强化 `manage_datasets.py` 中 `smart_watch` 的扩充/质量过滤规则（新增配件识别规则）。
- 重新执行：`expand-by-keyword -> enrich -> filter-quality`（仅 smart_watch）。
- 对 raw 末端残留的充电底座类条目做一次补充剔除。
- 备份旧文件后替换：
  - 备份目录：`datasets_40_work/archive_20260401_151207/`
  - 新 raw：`datasets_40_work/raw/smart_watch.json`
  - 新 enriched：`datasets_40_work/enriched/smart_watch.json`

## 3. 前后对比
| 数据文件 | 商品数 | 配件疑似数 | 含评论 | 含多图 |
|---|---:|---:|---:|---:|
| 旧 raw | 161 | 128 (79.5%) | 11/161 | 161/161 |
| 旧 enriched | 94 | 63 (67.0%) | 94/94 | 94/94 |
| 新 raw | 107 | 0 (0.0%) | 0/107 | 107/107 |
| 新 enriched | 105 | 0 (0.0%) | 105/105 | 105/105 |

## 4. 结果结论
- 新 `enriched/smart_watch.json` 共 105 条，配件疑似条目从 67.0% 降到 0.0%。
- 新 `raw/smart_watch.json` 共 107 条，配件疑似条目从 79.5% 降到 0.0%。
- 新 enriched 数据 100% 保留评论与图片字段（105/105）。
- 已完成“清洗 + 重扩充 + 覆盖替换”。

## 5. 产物
- JSON 报告：`datasets_40_work/enriched/smart_watch_cleanup_report_2026-04-01.json`
- 本文档：`docs/reports/SMART_WATCH_DATA_CLEANUP_2026-04-01.md`
