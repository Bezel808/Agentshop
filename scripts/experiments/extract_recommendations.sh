#!/usr/bin/env bash
set -euo pipefail
# 从批处理日志中提取各轮推荐的商品 ID
# 用法: ./scripts/experiments/extract_recommendations.sh [日志文件]
#       不传参数则使用 logs/ 下最新的日志

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
LOG="${1:-}"
if [[ -z "$LOG" ]]; then
  LOG=$(ls -t "$LOG_DIR"/batch_*.log 2>/dev/null | head -1)
  if [[ -z "$LOG" ]]; then
    echo "未找到日志文件。用法: $0 <logs/batch_YYYYMMDD_HHMMSS.log>"
    exit 1
  fi
  echo "使用最新日志: $LOG"
  echo ""
fi

if [[ "$LOG" != /* ]]; then
  LOG="$REPO_ROOT/$LOG"
fi

if [[ ! -f "$LOG" ]]; then
  echo "文件不存在: $LOG"
  exit 1
fi

awk '
  /========== (kimi|qwen)-[a-z]+-r[0-9]+ ==========/ {
    match($0, /(kimi|qwen)-[a-z]+-r[0-9]+/)
    if (RSTART) cond = substr($0, RSTART, RLENGTH)
  }
  /推荐商品 ID:/ {
    line = $0
    sub(/.*推荐商品 ID: */, "", line)
    sub(/ *=.*/, "", line)
    gsub(/^ +| +$/, "", line)
    pid = line
    if (cond != "" && pid != "") print cond, pid
  }
' "$LOG" | while read -r cond pid; do
  printf "%-25s %s\n" "$cond" "$pid"
done

echo ""
echo "Markdown 表格:"
echo "| 条件 | 推荐商品ID |"
echo "|------|------------|"
awk '
  /========== (kimi|qwen)-[a-z]+-r[0-9]+ ==========/ {
    match($0, /(kimi|qwen)-[a-z]+-r[0-9]+/)
    if (RSTART) cond = substr($0, RSTART, RLENGTH)
  }
  /推荐商品 ID:/ {
    line = $0
    sub(/.*推荐商品 ID: */, "", line)
    sub(/ *=.*/, "", line)
    gsub(/^ +| +$/, "", line)
    pid = line
    if (cond != "" && pid != "") printf "| %s | %s |\n", cond, pid
  }
' "$LOG"
