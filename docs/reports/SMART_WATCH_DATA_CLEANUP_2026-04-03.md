# Smart Watch 商品库清洗与扩充报告（2026-04-03）

## 1. 背景
- 发现 `datasets_expand_stage/smart_watch.json` 混入大量表带/表壳/贴膜等配件条目。
- 目标：清理配件并扩充 `smart_watch` 商品库。

## 2. 执行动作
- 重新执行 `smart_watch` 专项流程：`expand-by-keyword -> enrich -> filter-quality`。
- 扩充参数：`max_products=240`、`scan_limit=260000`。
- 扩充阶段自动拦截配件标题：`19072` 条。
- 替换正式文件前先备份旧数据。

## 3. 备份位置
- `datasets_40_work/archive_20260403_215118_smartwatch_refresh/`

## 4. 前后对比
| 文件 | 清洗前 | 清洗后 | 配件疑似比例（前 -> 后） |
|---|---:|---:|---:|
| `datasets_40_work/raw/smart_watch.json` | 107 | 240 | 0.0% -> 0.0% |
| `datasets_40_work/enriched/smart_watch.json` | 105 | 163 | 0.0% -> 0.0% |
| `datasets_expand_stage/smart_watch.json` | 160 | 240 | 80.0% -> 0.0% |

## 5. 结果
- `datasets_expand_stage/smart_watch.json` 的配件混入问题已清空。
- 正式库（`datasets_40_work/enriched/smart_watch.json`）从 105 扩充到 163。
- raw 库从 107 扩充到 240。

## 6. 产物
- 统计 JSON：`datasets_40_work/enriched/smart_watch_cleanup_report_2026-04-03.json`
- 本报告：`docs/reports/SMART_WATCH_DATA_CLEANUP_2026-04-03.md`
