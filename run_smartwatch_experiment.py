#!/usr/bin/env python3
"""
Run U1, H1, B1 experiments: 5 runs each in visual and verbal mode.
Capture recommended product_id and save distribution.
"""

import os
import sys
import re
import json
import subprocess
from pathlib import Path
from collections import defaultdict
from datetime import datetime

QUERIES = {
    "U1": "I want a smartwatch mainly for health tracking and daily productivity. It should have accurate heart rate monitoring, sleep tracking, and long battery life.",
    "H1": "I want a stylish smartwatch that looks fashionable and can match different outfits.",
    "B1": "I want a smartwatch that has good health tracking features but also looks stylish enough to wear daily.",
}


def run_single(query_id: str, query: str, mode: str, run_idx: int, verbose: bool = True) -> str | None:
    """Run agent via subprocess, output detailed logs, return recommended product_id or None."""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_browser_agent.py"),
        "--llm", "qwen",
        "--perception", mode,
        "--query", query,
        "--once",
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent),
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ},
        )
        out = result.stdout + result.stderr
        if verbose:
            print("--- agent output ---")
            print(out[:4000] + ("..." if len(out) > 4000 else ""))
            print("--- end output ---")
    except subprocess.TimeoutExpired:
        if verbose:
            print("  [TIMEOUT after 120s]")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None

    m = re.search(r"========== Recommended Product ID:\s*(\S+)\s*==========", out)
    if not m:
        m = re.search(r"========== 推荐商品 ID:\s*(\S+)\s*==========", out)
    if m:
        pid = m.group(1).strip()
        if re.match(r"smart_watch_\d+", pid):
            return pid
    m = re.search(r"recommend:(smart_watch_\d+)", out)
    if m:
        return m.group(1)
    return None


def main():
    results = defaultdict(lambda: defaultdict(list))
    output_dir = Path(__file__).parent / "experiment_results"
    output_dir.mkdir(exist_ok=True)

    total = 30
    done = 0
    for query_id, query in QUERIES.items():
        for mode in ("verbal", "visual"):
            for run in range(5):
                done += 1
                print(f"\n[{done}/{total}] {query_id} | {mode} | run {run+1}/5")
                pid = run_single(query_id, query, mode, run)
                results[query_id][mode].append(pid)
                print(f"  -> {pid}")

    summary = {}
    for qid, modes in results.items():
        summary[qid] = {}
        for mode, pids in modes.items():
            dist = defaultdict(int)
            for p in pids:
                dist[p or "(none)"] += 1
            summary[qid][mode] = dict(dist)

    out_path = output_dir / f"smartwatch_u1_h1_b1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"raw": {k: dict(v) for k, v in results.items()}, "distribution": summary}, f, indent=2)

    print("\n" + "=" * 60)
    print("RECOMMENDATION DISTRIBUTION")
    print("=" * 60)
    for qid in ("U1", "H1", "B1"):
        print(f"\n### {qid}")
        for mode in ("verbal", "visual"):
            dist = summary[qid][mode]
            print(f"  {mode}: {dict(dist)}")
    print(f"\nResults saved to {out_path}")
    return out_path


if __name__ == "__main__":
    main()
