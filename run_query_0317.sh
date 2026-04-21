#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv_aces/bin/python}"
QUERY_FILE="${QUERY_FILE:-$ROOT_DIR/query.md}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/logs/query_0317}"
LLM_BACKEND="${LLM_BACKEND:-qwen}"
TIMEOUT_SEC="${TIMEOUT_SEC:-240}"
MAX_QUERIES="${MAX_QUERIES:-0}"   # 0 means all

mkdir -p "$OUT_DIR"
SUMMARY_CSV="$OUT_DIR/summary.csv"
RESULT_JSON="$OUT_DIR/insights.json"

echo "id,category,code,mode,status,reason,steps,log_path,trace_path" > "$SUMMARY_CSV"

echo "[INFO] parsing queries from $QUERY_FILE"
mapfile -t QUERY_ROWS < <("$PYTHON_BIN" - <<'PY' "$QUERY_FILE"
import re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding='utf-8')
cat = 'unknown'
rows = []
for line in text.splitlines():
    mcat = re.match(r'^##\s+\d+\.\s+(.+?)\s*$', line.strip())
    if mcat:
        cat = re.sub(r'\s+', '_', mcat.group(1).strip().lower())
        continue
    m = re.match(r'^-\s+\*\*([A-Za-z0-9_]+)\*\*:\s*(.+)$', line.strip())
    if not m:
        continue
    code = m.group(1).strip()
    query = m.group(2).strip()
    qid = f"{cat}_{code}_{len(rows)+1:03d}"
    rows.append((qid, cat, code, query))
for qid, cat, code, query in rows:
    print("\t".join([qid, cat, code, query]))
PY
)

if [[ "${#QUERY_ROWS[@]}" -eq 0 ]]; then
  echo "[ERROR] no query parsed from $QUERY_FILE"
  exit 1
fi

TOTAL="${#QUERY_ROWS[@]}"
if [[ "$MAX_QUERIES" -gt 0 && "$MAX_QUERIES" -lt "$TOTAL" ]]; then
  TOTAL="$MAX_QUERIES"
fi

echo "[INFO] total queries to run: $TOTAL"

run_one() {
  local qid="$1"; local cat="$2"; local code="$3"; local query="$4"; local mode="$5"
  local tag="${qid}_${mode}"
  local log_path="$OUT_DIR/${tag}.log"
  local trace_path="$OUT_DIR/${tag}.jsonl"

  set +e
  timeout "$TIMEOUT_SEC" "$PYTHON_BIN" run_browser_agent.py \
    --llm "$LLM_BACKEND" \
    --perception "$mode" \
    --query "$query" \
    --once \
    --trace-output "$trace_path" \
    > "$log_path" 2>&1
  local rc=$?
  set -e

  local status="ok"
  if [[ $rc -eq 124 ]]; then
    status="timeout"
  elif [[ $rc -ne 0 ]]; then
    status="failed"
  fi

  local reason="unknown"
  local steps="0"
  if [[ -s "$trace_path" ]]; then
    local parsed
    parsed="$("$PYTHON_BIN" - <<'PY' "$trace_path"
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
lines = []
for ln in p.read_text(encoding='utf-8').splitlines():
    ln = ln.strip()
    if not ln:
        continue
    try:
        lines.append(json.loads(ln))
    except Exception:
        pass
reason = 'unknown'
for e in reversed(lines):
    if isinstance(e, dict):
        if e.get('event_type') == 'termination':
            payload = e.get('payload') or {}
            if isinstance(payload, dict) and payload.get('reason'):
                reason = str(payload['reason'])
                break
        term = e.get('termination') or {}
        if isinstance(term, dict) and term.get('reason'):
            reason = str(term['reason'])
            break
steps = 0
for e in lines:
    if isinstance(e, dict) and e.get('event_type') == 'action':
        steps += 1
print(reason)
print(steps)
PY
)"
    reason="$(echo "$parsed" | sed -n '1p')"
    steps="$(echo "$parsed" | sed -n '2p')"
  fi

  if [[ "$reason" == "unknown" && -s "$log_path" ]]; then
    if rg -q "APIConnectionError|Connection error|ConnectError" "$log_path"; then
      reason="llm_connection_error"
    elif rg -q "Operation not permitted" "$log_path"; then
      reason="sandbox_or_network_denied"
    elif rg -q "TargetClosedError|BrowserType.launch" "$log_path"; then
      reason="playwright_launch_error"
    elif rg -q "No valid action found" "$log_path"; then
      reason="no_valid_action"
    elif rg -q "未知工具: noop|unknown tool: noop" "$log_path"; then
      reason="unknown_tool_noop"
    fi
  fi

  echo "${qid},${cat},${code},${mode},${status},${reason},${steps},${log_path},${trace_path}" >> "$SUMMARY_CSV"
  echo "[DONE] $tag status=$status reason=$reason steps=$steps"
}

idx=0
for row in "${QUERY_ROWS[@]}"; do
  idx=$((idx+1))
  if [[ "$idx" -gt "$TOTAL" ]]; then
    break
  fi
  IFS=$'\t' read -r qid cat code query <<< "$row"
  echo "[RUN] ($idx/$TOTAL) $qid"
  run_one "$qid" "$cat" "$code" "$query" verbal
  run_one "$qid" "$cat" "$code" "$query" visual
 done

"$PYTHON_BIN" - <<'PY' "$SUMMARY_CSV" "$RESULT_JSON"
import csv, json, sys
from collections import Counter, defaultdict
summary_csv, out_json = sys.argv[1], sys.argv[2]
rows = []
with open(summary_csv, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

by_mode = defaultdict(list)
for r in rows:
    by_mode[r['mode']].append(r)

report = {'total_runs': len(rows), 'by_mode': {}, 'top_failure_reasons': {}, 'notes': []}
for mode, rs in by_mode.items():
    total = len(rs)
    ok = sum(1 for r in rs if r['status'] == 'ok')
    recommend_done = sum(1 for r in rs if r['reason'] == 'recommend_done')
    avg_steps = round(sum(int(r['steps'] or 0) for r in rs)/max(total,1), 2)
    reasons = Counter(r['reason'] for r in rs)
    report['by_mode'][mode] = {
        'total': total,
        'ok': ok,
        'ok_rate': round(ok/max(total,1), 4),
        'recommend_done': recommend_done,
        'recommend_done_rate': round(recommend_done/max(total,1), 4),
        'avg_steps': avg_steps,
        'top_reasons': reasons.most_common(5),
    }

all_fail = Counter(r['reason'] for r in rows if r['status'] != 'ok' or r['reason'] != 'recommend_done')
report['top_failure_reasons'] = all_fail.most_common(10)

if 'visual' in report['by_mode'] and 'verbal' in report['by_mode']:
    v = report['by_mode']['visual']
    t = report['by_mode']['verbal']
    report['notes'].append(
        f"visual recommend_done_rate={v['recommend_done_rate']}, verbal recommend_done_rate={t['recommend_done_rate']}"
    )
    report['notes'].append(
        f"visual avg_steps={v['avg_steps']}, verbal avg_steps={t['avg_steps']}"
    )

with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(json.dumps(report, ensure_ascii=False, indent=2))
PY

echo "[INFO] done. summary=$SUMMARY_CSV insights=$RESULT_JSON"
