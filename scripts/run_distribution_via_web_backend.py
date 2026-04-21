#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


DEFAULT_QUERIES = {
    "bluetooth_speaker_35_40": "bluetooth speaker price 35 and 40",
    "usb_flash_drive_720_1220": "usb_flash_drive price between 7.20 and 12.20",
    "vase_1499_20": "vase price between 14.99 and 20",
    "smart_watch_55_5999": "smart watch price between 55 and 59.99",
}


def _load_token(explicit_token: Optional[str], env_path: Path) -> str:
    if explicit_token:
        return explicit_token.strip()
    if os.getenv("ACES_VIEWER_TOKEN"):
        return os.getenv("ACES_VIEWER_TOKEN", "").strip()
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("ACES_VIEWER_TOKEN="):
                return line.split("=", 1)[1].strip()
    return ""


def _wait_idle(session: requests.Session, server: str, token: str, timeout_sec: int) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        try:
            resp = session.get(
                f"{server}/api/agent-status",
                params={"token": token},
                timeout=20,
            )
            data = resp.json()
            if not data.get("running", False):
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def _parse_rec_id(log_chunk: str) -> str:
    m = re.findall(r"Recommended Product ID:\s*([A-Za-z0-9_-]+)", log_chunk)
    if m:
        return m[-1]
    m2 = re.findall(r"recommend:([A-Za-z0-9_-]+)", log_chunk)
    if m2:
        return m2[-1]
    return "(none)"


def _run_once_via_web(
    session: requests.Session,
    server: str,
    token: str,
    query: str,
    mode: str,
    log_path: Path,
    run_timeout_sec: int,
    start_retries: int = 3,
) -> Tuple[str, str]:
    # Ensure no active run.
    if not _wait_idle(session, server, token, timeout_sec=run_timeout_sec):
        return "timeout_wait_idle", "(none)"

    start_size = log_path.stat().st_size if log_path.exists() else 0
    payload = {"query": query, "perception": mode, "headless": True}

    start_ok = False
    last_err = ""
    for _ in range(start_retries):
        try:
            resp = session.post(
                f"{server}/api/run-agent",
                params={"token": token},
                json=payload,
                timeout=30,
            )
            data = resp.json()
            if data.get("ok"):
                start_ok = True
                break
            last_err = str(data.get("error", "start_failed"))
            if "正在运行中" in last_err:
                time.sleep(1.5)
                continue
            time.sleep(1.0)
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(1.0)

    if not start_ok:
        return f"start_failed:{last_err}", "(none)"

    finished = _wait_idle(session, server, token, timeout_sec=run_timeout_sec)
    if not finished:
        return "timeout_running", "(none)"

    chunk = ""
    if log_path.exists():
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            f.seek(start_size)
            chunk = f.read()
    rec = _parse_rec_id(chunk)
    return "ok", rec


def _summarize(records: List[dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["query_name"], r["mode"])].append(r)

    out: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for (qname, mode), rows in grouped.items():
        cnt = Counter(row["recommendation"] for row in rows)
        n = len(rows)
        dist = {}
        for pid, c in cnt.most_common():
            dist[pid] = {"count": c, "probability": round(c / n, 4) if n else 0.0}
        out[qname][mode] = dist
    return dict(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run repeated recommendation distribution via web backend API (/api/run-agent)."
    )
    parser.add_argument("--server", default="http://127.0.0.1:5000")
    parser.add_argument("--token", default=None, help="ACES_VIEWER_TOKEN, defaults to env or .env")
    parser.add_argument("--runs-per-combo", type=int, default=100)
    parser.add_argument("--run-timeout-sec", type=int, default=420)
    parser.add_argument("--log-path", default="logs/web_server.log")
    parser.add_argument("--query", action="append", default=None, help="name=text")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    token = _load_token(args.token, repo_root / ".env")
    if not token:
        raise SystemExit("ACES_VIEWER_TOKEN not found. Pass --token or set env/.env.")

    queries = dict(DEFAULT_QUERIES)
    if args.query:
        queries = {}
        for item in args.query:
            if "=" not in item:
                raise SystemExit(f"Invalid --query format: {item} (expected name=text)")
            name, text = item.split("=", 1)
            queries[name.strip()] = text.strip()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else repo_root / "experiment_results" / f"web_dist_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    summary_path = out_dir / "summary.json"
    csv_path = out_dir / "summary.csv"

    records: List[dict] = []
    sess = requests.Session()
    log_path = (repo_root / args.log_path) if not Path(args.log_path).is_absolute() else Path(args.log_path)

    total = len(queries) * 2 * args.runs_per_combo
    done = 0
    print(f"[INFO] server={args.server}")
    print(f"[INFO] output_dir={out_dir}")
    print(f"[INFO] total_runs={total}")

    with records_path.open("w", encoding="utf-8") as fw:
        for qname, qtext in queries.items():
            for mode in ("verbal", "visual"):
                for i in range(1, args.runs_per_combo + 1):
                    t0 = time.time()
                    status, rec = _run_once_via_web(
                        session=sess,
                        server=args.server,
                        token=token,
                        query=qtext,
                        mode=mode,
                        log_path=log_path,
                        run_timeout_sec=args.run_timeout_sec,
                    )
                    row = {
                        "query_name": qname,
                        "query_text": qtext,
                        "mode": mode,
                        "run_idx": i,
                        "status": status,
                        "recommendation": rec,
                        "elapsed_sec": round(time.time() - t0, 3),
                    }
                    records.append(row)
                    fw.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fw.flush()
                    done += 1
                    print(
                        f"[{done}/{total}] {qname} | {mode} | run={i} | status={status} | rec={rec} | {row['elapsed_sec']}s"
                    )

    dist = _summarize(records)
    status_counts = Counter(r["status"] for r in records)
    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "server": args.server,
            "runs_per_combo": args.runs_per_combo,
            "queries": queries,
            "mode_source": "web_backend_api",
        },
        "status_counts": dict(status_counts),
        "distribution": dist,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_name", "mode", "recommendation", "count", "probability"])
        for qname, modes in dist.items():
            for mode, recs in modes.items():
                for pid, info in recs.items():
                    writer.writerow([qname, mode, pid, info["count"], info["probability"]])

    print(f"\nDone. records={records_path}")
    print(f"summary={summary_path}")
    print(f"csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

