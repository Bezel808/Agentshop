#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_QUERIES = {
    "vase_1499_20": "vase price between 14.99 and 20",
    "smart_watch_55_5999": "smart watch price between 55 and 59.99",
}


def parse_recommendation(output: str) -> Optional[str]:
    patterns = [
        r"Recommended Product ID:\s*([A-Za-z0-9_-]+)",
        r"推荐商品 ID:\s*([A-Za-z0-9_-]+)",  # backward compatibility
        r"recommend:([A-Za-z0-9_-]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, output)
        if m:
            return m.group(1).strip()
    return None


def run_once(
    repo_root: Path,
    llm: str,
    server: str,
    query_name: str,
    query_text: str,
    mode: str,
    run_idx: int,
    max_steps: int,
    timeout_sec: int,
    verbal_use_vlm: bool,
    retries: int,
) -> Dict:
    t0 = time.time()
    last_status = "error"
    last_output = ""
    for attempt in range(1, retries + 2):
        cmd = [
            sys.executable,
            str(repo_root / "run_browser_agent.py"),
            "--llm",
            llm,
            "--perception",
            mode,
            "--query",
            query_text,
            "--server",
            server,
            "--once",
            "--max-steps",
            str(max_steps),
        ]
        if verbal_use_vlm and mode == "verbal":
            cmd.append("--verbal-use-vlm")

        status = "ok"
        output = ""
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env={**os.environ},
            )
            output = (proc.stdout or "") + "\n" + (proc.stderr or "")
            if proc.returncode != 0:
                status = f"exit_{proc.returncode}"
        except subprocess.TimeoutExpired as e:
            status = "timeout"
            output = ((e.stdout or "") if isinstance(e.stdout, str) else "") + "\n" + (
                (e.stderr or "") if isinstance(e.stderr, str) else ""
            )
        except Exception as e:
            status = "error"
            output = f"{type(e).__name__}: {e}"

        rec_id = parse_recommendation(output)
        if rec_id is not None:
            elapsed_sec = round(time.time() - t0, 3)
            status_out = "ok" if status == "ok" else f"{status}_with_rec"
            return {
                "query_name": query_name,
                "query_text": query_text,
                "mode": mode,
                "run_idx": run_idx,
                "status": status_out,
                "recommendation": rec_id,
                "elapsed_sec": elapsed_sec,
                "attempts": attempt,
            }

        last_status = status
        last_output = output
        if attempt <= retries:
            time.sleep(min(8.0, (2 ** (attempt - 1)) + random.random()))

    elapsed_sec = round(time.time() - t0, 3)
    tail = ""
    if last_output:
        lines = [ln.strip() for ln in last_output.splitlines() if ln.strip()]
        if lines:
            tail = lines[-1][:300]
    return {
        "query_name": query_name,
        "query_text": query_text,
        "mode": mode,
        "run_idx": run_idx,
        "status": last_status,
        "recommendation": "(none)",
        "elapsed_sec": elapsed_sec,
        "attempts": retries + 1,
        "error_tail": tail,
    }


def summarize(records: List[Dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    grouped: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in records:
        grouped[(r["query_name"], r["mode"])].append(r)

    result: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for (query_name, mode), rows in grouped.items():
        counter = Counter(r["recommendation"] for r in rows)
        n = len(rows)
        dist = {}
        for pid, cnt in counter.most_common():
            dist[pid] = {
                "count": cnt,
                "probability": round(cnt / n, 4) if n else 0.0,
            }
        result[query_name][mode] = dist
    return dict(result)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repeated recommendation experiments for verbal/visual modes.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--llm", default="qwen", choices=["qwen", "kimi", "openai"])
    parser.add_argument("--server", default="http://127.0.0.1:5000")
    parser.add_argument("--runs-per-combo", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--timeout-verbal", type=int, default=120)
    parser.add_argument("--timeout-visual", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--query", action="append", default=None, help="Custom query in form name=text")
    parser.add_argument("--verbal-use-vlm", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    queries = dict(DEFAULT_QUERIES)
    if args.query:
        queries = {}
        for item in args.query:
            if "=" not in item:
                raise SystemExit(f"Invalid --query format: {item}. Expected name=text")
            name, text = item.split("=", 1)
            name = name.strip()
            text = text.strip()
            if not name or not text:
                raise SystemExit(f"Invalid --query format: {item}. Expected non-empty name=text")
            queries[name] = text

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else (repo_root / "experiment_results" / f"dist_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    records_jsonl = out_dir / "records.jsonl"
    summary_json = out_dir / "summary.json"
    summary_csv = out_dir / "summary.csv"

    tasks = []
    for query_name, query_text in queries.items():
        for mode in ("verbal", "visual"):
            for i in range(1, args.runs_per_combo + 1):
                timeout = args.timeout_visual if mode == "visual" else args.timeout_verbal
                tasks.append((query_name, query_text, mode, i, timeout))

    total = len(tasks)
    done = 0
    records: List[Dict] = []

    print(f"[INFO] repo_root={repo_root}")
    print(f"[INFO] output_dir={out_dir}")
    print(f"[INFO] total_runs={total} (queries={len(queries)}, modes=2, runs_per_combo={args.runs_per_combo})")
    print(
        f"[INFO] workers={args.workers}, llm={args.llm}, server={args.server}, "
        f"verbal_use_vlm={args.verbal_use_vlm}, retries={args.retries}"
    )
    print("")

    with records_jsonl.open("w", encoding="utf-8") as fw, ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut_map = {}
        for query_name, query_text, mode, run_idx, timeout in tasks:
            fut = ex.submit(
                run_once,
                repo_root,
                args.llm,
                args.server,
                query_name,
                query_text,
                mode,
                run_idx,
                args.max_steps,
                timeout,
                args.verbal_use_vlm,
                args.retries,
            )
            fut_map[fut] = (query_name, mode, run_idx)

        for fut in as_completed(fut_map):
            query_name, mode, run_idx = fut_map[fut]
            rec = fut.result()
            records.append(rec)
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fw.flush()
            done += 1
            print(
                f"[{done}/{total}] {query_name} | {mode} | run={run_idx} | "
                f"status={rec['status']} | rec={rec['recommendation']} | "
                f"attempts={rec.get('attempts', 1)} | {rec['elapsed_sec']}s"
            )

    dist = summarize(records)
    status_counter = Counter(r["status"] for r in records)
    summary_payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "llm": args.llm,
            "server": args.server,
            "runs_per_combo": args.runs_per_combo,
            "workers": args.workers,
            "max_steps": args.max_steps,
            "timeouts": {
                "verbal": args.timeout_verbal,
                "visual": args.timeout_visual,
            },
            "retries": args.retries,
            "queries": queries,
            "verbal_use_vlm": bool(args.verbal_use_vlm),
        },
        "status_counts": dict(status_counter),
        "distribution": dist,
    }
    summary_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with summary_csv.open("w", encoding="utf-8", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["query_name", "mode", "recommendation", "count", "probability"])
        for query_name, modes in dist.items():
            for mode, items in modes.items():
                for rec_id, stats in items.items():
                    writer.writerow([query_name, mode, rec_id, stats["count"], stats["probability"]])

    print("")
    print("==== STATUS COUNTS ====")
    for k, v in sorted(status_counter.items()):
        print(f"{k}: {v}")
    print("")
    print("==== TOP RECOMMENDATIONS ====")
    for query_name, modes in dist.items():
        for mode, items in modes.items():
            top = list(items.items())[:10]
            print(f"{query_name} | {mode}")
            for rec_id, stats in top:
                print(f"  {rec_id}: {stats['count']} ({stats['probability']:.2%})")
            if not top:
                print("  (none)")
            print("")

    print(f"[DONE] records: {records_jsonl}")
    print(f"[DONE] summary: {summary_json}")
    print(f"[DONE] csv: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
