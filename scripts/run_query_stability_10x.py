#!/usr/bin/env python3
"""
Run the same query repeatedly for verbal/visual modes and summarize stability.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Optional


def parse_recommendation(trace_path: Path, log_path: Path) -> tuple[Optional[str], Optional[int]]:
    rec_pid: Optional[str] = None
    rec_step: Optional[int] = None

    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("event_type") != "tool_result":
                continue
            payload = obj.get("payload") or {}
            action = payload.get("action") or {}
            result = payload.get("result") or {}
            if action.get("tool_name") == "recommend" and result.get("success"):
                data = result.get("data") or {}
                rec_pid = data.get("product_id")
                rec_step = obj.get("step")

    if rec_pid:
        return rec_pid, rec_step

    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"recommend:([A-Za-z0-9_\\-]+)", text)
        if m:
            rec_pid = m.group(1)
    return rec_pid, rec_step


def entropy_norm(counter: Counter[str]) -> float:
    n = sum(counter.values())
    if n <= 1:
        return 0.0
    probs = [v / n for v in counter.values() if v > 0]
    h = -sum(p * math.log(p, 2) for p in probs)
    hmax = math.log(len(counter), 2) if len(counter) > 1 else 1.0
    return h / hmax if hmax > 0 else 0.0


def summarize(mode_rows: list[dict]) -> dict:
    valid = [r for r in mode_rows if r["recommended_product_id"]]
    recs = Counter(r["recommended_product_id"] for r in valid)
    top1_pid = recs.most_common(1)[0][0] if recs else None
    top1_count = recs.most_common(1)[0][1] if recs else 0
    steps = [r["recommend_step"] for r in valid if r["recommend_step"] is not None]

    return {
        "runs": len(mode_rows),
        "valid_recommendations": len(valid),
        "success_rate": len(valid) / len(mode_rows) if mode_rows else 0.0,
        "unique_recommendations": len(recs),
        "top1_product_id": top1_pid,
        "top1_ratio": top1_count / len(valid) if valid else 0.0,
        "normalized_entropy": entropy_norm(recs),
        "avg_recommend_step": mean(steps) if steps else None,
        "step_std": pstdev(steps) if len(steps) > 1 else 0.0,
        "distribution": dict(recs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Repeat one query 10x and compare verbal/visual stability.")
    parser.add_argument("--query", required=True, help="Query text to run repeatedly")
    parser.add_argument("--runs", type=int, default=10, help="Number of repeats per mode")
    parser.add_argument("--llm", default="qwen", help="LLM backend for run_browser_agent.py")
    parser.add_argument("--server", default="http://localhost:5000", help="Web server URL")
    parser.add_argument("--timeout", type=int, default=360, help="Timeout seconds per run")
    parser.add_argument("--output-dir", default="logs/stability_runs", help="Base output directory")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    run_py = root / "run_browser_agent.py"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / args.output_dir / f"query_stability_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    total = args.runs * 2
    idx = 0
    for mode in ("verbal", "visual"):
        for run_idx in range(1, args.runs + 1):
            idx += 1
            trace = out_dir / f"{mode}_run{run_idx}.jsonl"
            log = out_dir / f"{mode}_run{run_idx}.log"

            cmd = [
                sys.executable,
                str(run_py),
                "--llm",
                args.llm,
                "--perception",
                mode,
                "--query",
                args.query,
                "--server",
                args.server,
                "--once",
                "--trace-output",
                str(trace),
            ]

            print(f"[{idx}/{total}] {mode} run {run_idx}/{args.runs}")
            proc = subprocess.run(
                cmd,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=args.timeout,
                env=None,
            )
            log.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
            rec_pid, rec_step = parse_recommendation(trace, log)

            records.append(
                {
                    "mode": mode,
                    "run": run_idx,
                    "exit_code": proc.returncode,
                    "recommended_product_id": rec_pid,
                    "recommend_step": rec_step,
                }
            )

    by_mode = {
        "verbal": summarize([r for r in records if r["mode"] == "verbal"]),
        "visual": summarize([r for r in records if r["mode"] == "visual"]),
    }

    summary = {
        "query": args.query,
        "runs_per_mode": args.runs,
        "llm": args.llm,
        "server": args.server,
        "timestamp": ts,
        "by_mode": by_mode,
    }

    with (out_dir / "records.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "run", "exit_code", "recommended_product_id", "recommend_step"])
        w.writeheader()
        w.writerows(records)

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Stability Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

