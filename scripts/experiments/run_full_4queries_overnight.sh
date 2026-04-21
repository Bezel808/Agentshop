#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$ROOT/experiment_results/full_4queries_100x2_vlmverbal_${TS}"
LOG_DIR="$ROOT/experiment_results/nightly_logs"
mkdir -p "$RUN_ROOT" "$LOG_DIR"

MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"
SLEEP_BETWEEN_ATTEMPTS="${SLEEP_BETWEEN_ATTEMPTS:-30}"

echo "[INFO] root=$ROOT"
echo "[INFO] run_root=$RUN_ROOT"
echo "[INFO] max_attempts=$MAX_ATTEMPTS"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  OUT_DIR="$RUN_ROOT/attempt_${attempt}"
  mkdir -p "$OUT_DIR"
  echo "[INFO] attempt=$attempt out_dir=$OUT_DIR"

  python3 scripts/experiments/run_query_mode_distribution.py \
    --llm qwen \
    --server http://127.0.0.1:5000 \
    --runs-per-combo 100 \
    --workers 4 \
    --max-steps 45 \
    --timeout-verbal 220 \
    --timeout-visual 240 \
    --retries 2 \
    --verbal-use-vlm \
    --query "bluetooth_speaker_35_40=bluetooth speaker price 35 and 40" \
    --query "usb_flash_drive_720_1220=usb_flash_drive price between 7.20 and 12.20" \
    --query "vase_1499_20=vase price between 14.99 and 20" \
    --query "smart_watch_55_5999=smart watch price between 55 and 59.99" \
    --output-dir "$OUT_DIR"

  RECORDS_FILE="$OUT_DIR/records.jsonl"
  SUMMARY_FILE="$OUT_DIR/summary.json"
  lines=0
  if [[ -f "$RECORDS_FILE" ]]; then
    lines="$(wc -l < "$RECORDS_FILE" | tr -d '[:space:]')"
  fi

  ok_count="$(python3 - "$SUMMARY_FILE" <<'PY'
import json, sys
path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(int((data.get("status_counts") or {}).get("ok", 0)))
except Exception:
    print(0)
PY
)"

  echo "[INFO] attempt=$attempt records=$lines ok_count=$ok_count"
  if [[ "$lines" == "800" && "$ok_count" == "800" ]]; then
    ln -sfn "$OUT_DIR" "$ROOT/experiment_results/full_4queries_latest"
    echo "[DONE] full run completed at $OUT_DIR"
    exit 0
  fi

  echo "[WARN] attempt=$attempt incomplete, sleep ${SLEEP_BETWEEN_ATTEMPTS}s then retry..."
  sleep "$SLEEP_BETWEEN_ATTEMPTS"
done

echo "[ERROR] all attempts failed to produce 800/800 complete records"
exit 1

