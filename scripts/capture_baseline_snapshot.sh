#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/docs/baseline}"
mkdir -p "$OUT_DIR"

echo "[baseline] writing to $OUT_DIR"

timeout 20 python "$ROOT_DIR/run_browser_agent.py" --help > "$OUT_DIR/run_browser_agent_help.txt"
timeout 20 python "$ROOT_DIR/run_agent_dual_mode.py" --help > "$OUT_DIR/run_agent_dual_mode_help.txt"
timeout 20 python "$ROOT_DIR/start_web_server.py" --help > "$OUT_DIR/start_web_server_help.txt" || true
if [[ ! -s "$OUT_DIR/start_web_server_help.txt" ]]; then
  echo "start_web_server.py does not expose a fast --help output in this environment." > "$OUT_DIR/start_web_server_help.txt"
fi

cat > "$OUT_DIR/README.md" <<'MD'
# Baseline Snapshot

This folder stores CLI help snapshots for compatibility regression checks.

- `run_browser_agent_help.txt`
- `run_agent_dual_mode_help.txt`
- `start_web_server_help.txt`

Regenerate:

```bash
bash scripts/capture_baseline_snapshot.sh
```
MD

echo "[baseline] done"
