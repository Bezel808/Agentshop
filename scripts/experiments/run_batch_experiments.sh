#!/usr/bin/env bash
set -euo pipefail
# 2 LLM * 2 perception * 3 rounds = 12 轮
# 不设置 temperature，使用默认
# 日志保存到 logs/batch_YYYYMMDD_HHMMSS.log，结束后自动提取推荐商品汇总

QUERY="我需要一个纯色的好看的鼠标垫"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/batch_$(date +%Y%m%d_%H%M%S).log"
echo "日志文件: $LOG"
echo ""

{
for llm in kimi qwen; do
  for perception in visual verbal; do
    for round in 1 2 3; do
      cond="${llm}-${perception}-r${round}"
      echo "========== $cond =========="
      python "$REPO_ROOT/run_browser_agent.py" --llm "$llm" --perception "$perception" \
        --query "$QUERY" --once --condition-name "$cond"
      echo ""
      sleep 2
    done
  done
done
} 2>&1 | tee "$LOG"

echo "========== 全部 12 轮完成 =========="

# 提取推荐商品汇总（与 extract_recommendations.sh 相同逻辑）
echo ""
echo "========== 推荐商品汇总 =========="
awk '
  /========== (kimi|qwen)-[a-z]+-r[0-9]+ ==========/ {
    match($0, /(kimi|qwen)-[a-z]+-r[0-9]+/)
    if (RSTART) cond = substr($0, RSTART, RLENGTH)
  }
  /推荐商品 ID:|Recommended Product ID:/ {
    line = $0
    sub(/.*(推荐商品 ID:|Recommended Product ID:) */, "", line)
    sub(/ *=.*/, "", line)
    gsub(/^ +| +$/, "", line)
    pid = line
    if (cond != "" && pid != "") printf "%-25s %s\n", cond, pid
  }
' "$LOG"

echo ""
echo "| 条件 | 推荐商品ID |"
echo "|------|------------|"
awk '
  /========== (kimi|qwen)-[a-z]+-r[0-9]+ ==========/ {
    match($0, /(kimi|qwen)-[a-z]+-r[0-9]+/)
    if (RSTART) cond = substr($0, RSTART, RLENGTH)
  }
  /推荐商品 ID:|Recommended Product ID:/ {
    line = $0
    sub(/.*(推荐商品 ID:|Recommended Product ID:) */, "", line)
    sub(/ *=.*/, "", line)
    gsub(/^ +| +$/, "", line)
    pid = line
    if (cond != "" && pid != "") printf "| %s | %s |\n", cond, pid
  }
' "$LOG"

echo ""
echo "完整日志: $LOG"
echo "提取汇总: ./scripts/experiments/extract_recommendations.sh $LOG"
