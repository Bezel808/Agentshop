#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import requests

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_browser_agent.py"
SERVER = ROOT / "start_web_server.py"

SMARTWATCH_QUERIES = {
    "U2S": "I want reliable health monitoring functions such as sleep tracking, blood oxygen measurement, and heart rate alerts.",
    "B1A": "I want the smartwatch to be both fashionable and highly functional.",
    "H2S": "I want customizable watch faces, straps, and interface themes so that the watch reflects my personal taste.",
    "H3A": "I want using the smartwatch to feel exciting and novel, like owning a trendy piece of technology.",
    "U1S": "I want the smartwatch to accurately track steps, heart rate, calories burned, and workout duration.",
}


@dataclass
class ServerHandle:
    proc: subprocess.Popen
    port: int


def wait_health(server_url: str, timeout_sec: int = 60) -> None:
    deadline = time.time() + timeout_sec
    last_err = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{server_url}/health", timeout=2)
            if r.ok:
                return
        except Exception as e:
            last_err = e
        time.sleep(1)
    raise RuntimeError(f"Server health check failed for {server_url}: {last_err}")


def start_server(port: int, condition_file: Optional[str]) -> ServerHandle:
    cmd = [
        sys.executable,
        str(SERVER),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--datasets-dir",
        "datasets_unified",
        "--simple-search",
    ]
    if condition_file:
        cmd.extend(["--condition-file", condition_file])

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    handle = ServerHandle(proc=proc, port=port)
    wait_health(f"http://127.0.0.1:{port}")
    return handle


def stop_server(handle: Optional[ServerHandle]) -> None:
    if not handle:
        return
    try:
        os.killpg(os.getpgid(handle.proc.pid), signal.SIGTERM)
    except Exception:
        pass


def run_agent_once(
    api_key: str,
    llm: str,
    server_url: str,
    query: str,
    mode: str,
    trace_path: Path,
    log_path: Path,
    *,
    condition_name: Optional[str] = None,
    max_steps: int = 40,
    max_repeated_actions: int = 6,
    max_errors: int = 8,
    max_history: int = 12,
    temperature: float = 0.7,
    confidence_threshold: Optional[int] = None,
    min_viewed_before_recommend: Optional[int] = None,
) -> int:
    env = os.environ.copy()
    if llm == "qwen":
        env["QWEN_API_KEY"] = api_key
    elif llm == "kimi":
        env["KIMI_API_KEY"] = api_key
    else:
        env["OPENAI_API_KEY"] = api_key

    cmd = [
        sys.executable,
        str(RUNNER),
        "--llm",
        llm,
        "--perception",
        mode,
        "--query",
        query,
        "--server",
        server_url,
        "--once",
        "--trace-output",
        str(trace_path),
        "--max-steps",
        str(max_steps),
        "--max-repeated-actions",
        str(max_repeated_actions),
        "--max-errors",
        str(max_errors),
        "--max-history",
        str(max_history),
        "--temperature",
        str(temperature),
    ]
    if confidence_threshold is not None:
        cmd.extend(["--confidence-threshold", str(confidence_threshold)])
    if min_viewed_before_recommend is not None:
        cmd.extend(["--min-viewed-before-recommend", str(min_viewed_before_recommend)])
    if condition_name:
        cmd.extend(["--condition-name", condition_name])

    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, env=env)
    log_path.write_text((p.stdout or "") + "\n" + (p.stderr or ""), encoding="utf-8")
    return p.returncode


def parse_trace(trace_path: Path) -> Dict[str, Any]:
    actions: List[Tuple[str, str]] = []
    by_tool = Counter()
    success = False
    reason = "unknown"
    rec_pid: Optional[str] = None
    rec_step: Optional[int] = None
    first_selected_index: Optional[int] = None

    if not trace_path.exists():
        return {
            "steps": 0,
            "toolcall_total": 0,
            "toolcall_by_tool": {},
            "noop_rate": 1.0,
            "repeat_action_rate": 0.0,
            "success": False,
            "reason": "no_trace",
            "recommended_product_id": None,
            "recommend_step": None,
            "selected_product_rank": None,
        }

    for line in trace_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        et = e.get("event_type")
        payload = e.get("payload") or {}
        if et == "action":
            action = payload.get("action") or {}
            tool = str(action.get("tool_name") or "")
            params = action.get("parameters") or {}
            by_tool[tool] += 1
            actions.append((tool, json.dumps(params, sort_keys=True, ensure_ascii=False)))
            if tool == "select_product" and first_selected_index is None:
                try:
                    first_selected_index = int(params.get("index"))
                except Exception:
                    pass
        elif et == "tool_result":
            action = payload.get("action") or {}
            result = payload.get("result") or {}
            if action.get("tool_name") == "recommend" and result.get("success"):
                data = result.get("data") or {}
                success = True
                rec_pid = data.get("product_id")
                rec_step = e.get("step")
        elif et == "termination":
            reason = str((payload.get("reason") or reason))

    repeats = 0
    for i in range(1, len(actions)):
        if actions[i] == actions[i - 1]:
            repeats += 1

    total = len(actions)
    noop = by_tool.get("noop", 0)
    return {
        "steps": total,
        "toolcall_total": total,
        "toolcall_by_tool": dict(by_tool),
        "noop_rate": (noop / total) if total else 0.0,
        "repeat_action_rate": (repeats / total) if total else 0.0,
        "success": success,
        "reason": reason,
        "recommended_product_id": rec_pid,
        "recommend_step": rec_step,
        "selected_product_rank": first_selected_index,
    }


def entropy_norm(counter: Counter[str]) -> float:
    n = sum(counter.values())
    if n <= 1:
        return 0.0
    probs = [v / n for v in counter.values() if v > 0]
    h = -sum(p * math.log(p, 2) for p in probs)
    hmax = math.log(len(counter), 2) if len(counter) > 1 else 1.0
    return h / hmax if hmax > 0 else 0.0


def summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    ok = [r for r in rows if r.get("success")]
    steps = [r["steps"] for r in rows]
    rec_steps = [r["recommend_step"] for r in ok if r.get("recommend_step") is not None]
    recs = Counter(r["recommended_product_id"] for r in ok if r.get("recommended_product_id"))
    top_count = recs.most_common(1)[0][1] if recs else 0
    top_ratio = (top_count / len(ok)) if ok else 0.0
    return {
        "n": n,
        "success_rate": (len(ok) / n) if n else 0.0,
        "avg_steps": mean(steps) if steps else None,
        "avg_recommend_step": mean(rec_steps) if rec_steps else None,
        "avg_noop_rate": mean([r["noop_rate"] for r in rows]) if rows else None,
        "avg_repeat_action_rate": mean([r["repeat_action_rate"] for r in rows]) if rows else None,
        "top1_ratio": top_ratio,
        "distribution_entropy": entropy_norm(recs),
        "product_switch_rate": (1 - top_ratio) if ok else None,
        "distribution": dict(recs),
    }


def fetch_search_products(server_url: str, query: str, condition_name: Optional[str] = None) -> List[Dict[str, Any]]:
    params = {"q": query, "page": 1, "page_size": 8}
    if condition_name:
        params["condition_name"] = condition_name
    try:
        r = requests.get(f"{server_url}/api/search", params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("products", []) or []
    except Exception:
        return []


def compute_price_sensitivity_z(products: List[Dict[str, Any]], product_id: Optional[str]) -> Optional[float]:
    if not product_id or not products:
        return None
    prices = []
    selected = None
    for p in products:
        try:
            pr = float(p.get("price"))
        except Exception:
            continue
        prices.append(pr)
        if str(p.get("id")) == str(product_id):
            selected = pr
    if selected is None or len(prices) < 2:
        return None
    mu = sum(prices) / len(prices)
    var = sum((x - mu) ** 2 for x in prices) / len(prices)
    std = math.sqrt(var)
    return 0.0 if std == 0 else (selected - mu) / std


def build_sensitivity_configs(config_set: str) -> List[Dict[str, Any]]:
    full = [
        {"config": "baseline", "param": "baseline", "value": "default", "kwargs": {}},
        {"config": "max_steps_20", "param": "max_steps", "value": "20", "kwargs": {"max_steps": 20}},
        {"config": "max_steps_60", "param": "max_steps", "value": "60", "kwargs": {"max_steps": 60}},
        {"config": "temperature_0.2", "param": "temperature", "value": "0.2", "kwargs": {"temperature": 0.2}},
        {"config": "temperature_1.0", "param": "temperature", "value": "1.0", "kwargs": {"temperature": 1.0}},
        {"config": "max_history_8", "param": "max_history", "value": "8", "kwargs": {"max_history": 8}},
        {"config": "max_history_24", "param": "max_history", "value": "24", "kwargs": {"max_history": 24}},
        {"config": "min_viewed_1", "param": "min_viewed_before_recommend", "value": "1", "kwargs": {"min_viewed_before_recommend": 1}},
        {"config": "min_viewed_3", "param": "min_viewed_before_recommend", "value": "3", "kwargs": {"min_viewed_before_recommend": 3}},
        {"config": "confidence_75", "param": "confidence_threshold", "value": "75", "kwargs": {"confidence_threshold": 75}},
        {"config": "confidence_95", "param": "confidence_threshold", "value": "95", "kwargs": {"confidence_threshold": 95}},
    ]
    if config_set == "full":
        return full
    # quick subset: baseline + one stress point per parameter group.
    return [
        x for x in full
        if x["config"] in {"baseline", "max_steps_20", "temperature_1.0", "max_history_8", "min_viewed_3", "confidence_95"}
    ]


def run_sensitivity(
    out_dir: Path,
    api_key: str,
    llm: str,
    server_url: str,
    runs: int,
    modes: List[str],
    query_items: List[Tuple[str, str]],
    configs: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    sensitivity_dir = out_dir / "sensitivity"
    sensitivity_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    total = len(configs) * len(query_items) * len(modes) * runs
    cur = 0
    for cfg in configs:
        for q_code, query in query_items:
            for mode in modes:
                for run_idx in range(1, runs + 1):
                    cur += 1
                    run_id = f"{cfg['config']}__{q_code}__{mode}__r{run_idx}"
                    trace = sensitivity_dir / f"{run_id}.jsonl"
                    log = sensitivity_dir / f"{run_id}.log"
                    print(f"[SENS {cur}/{total}] {run_id}")
                    rc = run_agent_once(
                        api_key,
                        llm,
                        server_url,
                        query,
                        mode,
                        trace,
                        log,
                        **cfg["kwargs"],
                    )
                    parsed = parse_trace(trace)
                    rows.append(
                        {
                            "phase": "sensitivity",
                            "config": cfg["config"],
                            "param": cfg["param"],
                            "value": cfg["value"],
                            "query_code": q_code,
                            "query": query,
                            "mode": mode,
                            "run": run_idx,
                            "exit_code": rc,
                            **parsed,
                            "toolcall_by_tool": json.dumps(parsed["toolcall_by_tool"], ensure_ascii=False),
                        }
                    )

    by_cfg_mode: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["config"], r["mode"])].append(r)

    for (config, mode), group_rows in grouped.items():
        meta = next(c for c in configs if c["config"] == config)
        s = summarize_group(group_rows)
        by_cfg_mode.append({"config": config, "param": meta["param"], "value": meta["value"], "mode": mode, **s})

    return rows, by_cfg_mode


def run_highlight_ab(
    out_dir: Path,
    api_key: str,
    llm: str,
    server_url: str,
    runs: int,
    query_items: List[Tuple[str, str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ab_dir = out_dir / "highlight_ab"
    ab_dir.mkdir(parents=True, exist_ok=True)

    conditions = ["no_label", "best_seller_on_target"]
    rows: List[Dict[str, Any]] = []

    # visual full, verbal sampled (first 2 queries)
    verbal_codes = [x[0] for x in query_items[: min(2, len(query_items))]]
    total = (len(conditions) * len(query_items) * runs) + (len(conditions) * len(verbal_codes) * runs)
    cur = 0

    for cond in conditions:
        for q_code, query in query_items:
            for run_idx in range(1, runs + 1):
                cur += 1
                run_id = f"{cond}__{q_code}__visual__r{run_idx}"
                trace = ab_dir / f"{run_id}.jsonl"
                log = ab_dir / f"{run_id}.log"
                print(f"[AB {cur}/{total}] {run_id}")
                rc = run_agent_once(
                    api_key,
                    llm,
                    server_url,
                    query,
                    "visual",
                    trace,
                    log,
                    condition_name=cond,
                )
                parsed = parse_trace(trace)
                products = fetch_search_products(server_url, query, cond)
                pz = compute_price_sensitivity_z(products, parsed.get("recommended_product_id"))
                rows.append(
                    {
                        "phase": "highlight_ab",
                        "condition": cond,
                        "query_code": q_code,
                        "query": query,
                        "mode": "visual",
                        "run": run_idx,
                        "exit_code": rc,
                        **parsed,
                        "price_sensitivity": pz,
                        "toolcall_by_tool": json.dumps(parsed["toolcall_by_tool"], ensure_ascii=False),
                    }
                )

    for cond in conditions:
        for q_code in verbal_codes:
            query = dict(query_items)[q_code]
            for run_idx in range(1, runs + 1):
                cur += 1
                run_id = f"{cond}__{q_code}__verbal__r{run_idx}"
                trace = ab_dir / f"{run_id}.jsonl"
                log = ab_dir / f"{run_id}.log"
                print(f"[AB {cur}/{total}] {run_id}")
                rc = run_agent_once(
                    api_key,
                    llm,
                    server_url,
                    query,
                    "verbal",
                    trace,
                    log,
                    condition_name=cond,
                )
                parsed = parse_trace(trace)
                products = fetch_search_products(server_url, query, cond)
                pz = compute_price_sensitivity_z(products, parsed.get("recommended_product_id"))
                rows.append(
                    {
                        "phase": "highlight_ab",
                        "condition": cond,
                        "query_code": q_code,
                        "query": query,
                        "mode": "verbal",
                        "run": run_idx,
                        "exit_code": rc,
                        **parsed,
                        "price_sensitivity": pz,
                        "toolcall_by_tool": json.dumps(parsed["toolcall_by_tool"], ensure_ascii=False),
                    }
                )

    summary = {}
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["condition"], r["mode"])].append(r)
    for key, group_rows in grouped.items():
        summary[f"{key[0]}|{key[1]}"] = summarize_group(group_rows)
        sr = [x for x in group_rows if x.get("selected_product_rank") is not None]
        pz = [x["price_sensitivity"] for x in group_rows if x.get("price_sensitivity") is not None]
        summary[f"{key[0]}|{key[1]}"]["avg_selected_product_rank"] = mean([x["selected_product_rank"] for x in sr]) if sr else None
        summary[f"{key[0]}|{key[1]}"]["avg_price_sensitivity"] = mean(pz) if pz else None

    return rows, summary


def compute_sensitivity_rank(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    base = {}
    for r in summary_rows:
        if r["config"] == "baseline":
            base[r["mode"]] = r

    out = []
    for r in summary_rows:
        if r["config"] == "baseline":
            continue
        b = base.get(r["mode"])
        if not b:
            continue
        delta_success = abs((r.get("success_rate") or 0) - (b.get("success_rate") or 0))
        delta_steps = abs((r.get("avg_steps") or 0) - (b.get("avg_steps") or 0))
        delta_top1 = abs((r.get("top1_ratio") or 0) - (b.get("top1_ratio") or 0))
        delta_noop = abs((r.get("avg_noop_rate") or 0) - (b.get("avg_noop_rate") or 0))
        score = delta_success * 3 + delta_top1 * 2 + delta_steps * 0.1 + delta_noop * 1.5
        out.append({
            "config": r["config"],
            "param": r["param"],
            "value": r["value"],
            "mode": r["mode"],
            "delta_success": delta_success,
            "delta_steps": delta_steps,
            "delta_top1": delta_top1,
            "delta_noop": delta_noop,
            "sensitivity_score": score,
        })
    out.sort(key=lambda x: x["sensitivity_score"], reverse=True)
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def render_report(
    out_dir: Path,
    llm: str,
    runs: int,
    sens_summary: List[Dict[str, Any]],
    sens_rank: List[Dict[str, Any]],
    ab_summary: Dict[str, Any],
) -> None:
    highs = sens_rank[:3]
    lows = sorted(sens_rank, key=lambda x: x["sensitivity_score"])[:2]

    def _fmt(v: Any) -> str:
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = []
    lines.append("# Visual Agent 框架评估与价格高亮实验报告")
    lines.append("")
    lines.append("## 1) 实验设计与控制变量")
    lines.append(f"- LLM: `{llm}`")
    lines.append(f"- 查询集: `U2S, B1A, H2S, H3A, U1S`（smartwatch）")
    lines.append(f"- 重复次数: 每设置每模式 `{runs}` 轮")
    lines.append("- 固定设置: 同一数据集、同一工具集合、同一运行入口 `run_browser_agent.py`，仅改单一因素")
    lines.append("")
    lines.append("## 2) 超参数敏感性结果")
    lines.append("- 统计口径: `success_rate / avg_steps / top1_ratio / distribution_entropy / avg_noop_rate / avg_repeat_action_rate`")
    lines.append("")
    lines.append("### 2.1 各配置汇总（按 mode）")
    lines.append("")
    lines.append("| config | param | value | mode | success_rate | avg_steps | top1_ratio | entropy | noop_rate | repeat_rate |")
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in sens_summary:
        lines.append(
            f"| {r['config']} | {r['param']} | {r['value']} | {r['mode']} | {_fmt(r.get('success_rate'))} | {_fmt(r.get('avg_steps'))} | {_fmt(r.get('top1_ratio'))} | {_fmt(r.get('distribution_entropy'))} | {_fmt(r.get('avg_noop_rate'))} | {_fmt(r.get('avg_repeat_action_rate'))} |"
        )

    lines.append("")
    lines.append("### 2.2 高敏感参数（Top 3）")
    for i, x in enumerate(highs, 1):
        lines.append(
            f"{i}. `{x['config']}` ({x['mode']}): score={_fmt(x['sensitivity_score'])}, Δsuccess={_fmt(x['delta_success'])}, Δsteps={_fmt(x['delta_steps'])}, Δtop1={_fmt(x['delta_top1'])}, Δnoop={_fmt(x['delta_noop'])}"
        )

    lines.append("")
    lines.append("### 2.3 低敏感参数（Bottom 2）")
    for i, x in enumerate(lows, 1):
        lines.append(
            f"{i}. `{x['config']}` ({x['mode']}): score={_fmt(x['sensitivity_score'])}, Δsuccess={_fmt(x['delta_success'])}, Δsteps={_fmt(x['delta_steps'])}, Δtop1={_fmt(x['delta_top1'])}, Δnoop={_fmt(x['delta_noop'])}"
        )

    lines.append("")
    lines.append("## 3) 价格高亮 A/B 结果")
    lines.append("- 对照组: `no_label`")
    lines.append("- 实验组: `best_seller_on_target` + 价格视觉强调（加粗/轻背景）")
    lines.append("")
    lines.append("| condition|mode | n | success_rate | avg_steps | top1_ratio | product_switch_rate | avg_selected_product_rank | avg_price_sensitivity |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for k in sorted(ab_summary.keys()):
        s = ab_summary[k]
        cond, mode = k.split("|")
        lines.append(
            f"| {cond} | {mode} | {_fmt(s.get('n'))} | {_fmt(s.get('success_rate'))} | {_fmt(s.get('avg_steps'))} | {_fmt(s.get('top1_ratio'))} | {_fmt(s.get('product_switch_rate'))} | {_fmt(s.get('avg_selected_product_rank'))} | {_fmt(s.get('avg_price_sensitivity'))} |"
        )

    lines.append("")
    lines.append("## 4) 风险与解释")
    lines.append("- 小样本（3轮）下，结论以方向性为主，建议后续扩到5-10轮复核。")
    lines.append("- LLM接口波动、页面渲染随机性会引入噪声，已通过同配置重复运行部分对冲。")
    lines.append("- `price_sensitivity` 使用首屏候选价格 z-score 近似，便于快速比较，不等价于完整离线评测指标。")

    lines.append("")
    lines.append("## 5) 可执行建议")
    lines.append("1. 将高敏感参数纳入默认配置基准，优先收敛 `max_steps/temperature` 组合。")
    lines.append("2. 若 A/B 出现 top1_ratio 或 steps 明显变化，建议对价格高亮进行分层试验（功利/享乐任务分开）。")
    lines.append("3. 下一轮把 `runs` 提升到 5+ 并加 bootstrap 置信区间，提升统计稳健性。")

    (out_dir / "visual_framework_and_price_highlight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run visual-framework sensitivity + price highlight AB experiments")
    parser.add_argument("--llm", default="qwen", choices=["qwen", "openai", "kimi"])
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--base-port", type=int, default=5010)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sensitivity-modes", default="visual,verbal", help="comma-separated modes")
    parser.add_argument("--config-set", choices=["full", "quick"], default="full", help="sensitivity config set")
    parser.add_argument(
        "--query-codes",
        default="U2S,B1A,H2S,H3A,U1S",
        help="comma-separated query codes from {U2S,B1A,H2S,H3A,U1S}",
    )
    args = parser.parse_args()

    api_key = args.api_key
    if not api_key:
        if args.llm == "qwen":
            api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        elif args.llm == "kimi":
            api_key = os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key. Pass --api-key or set env var.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else (ROOT / "logs" / "visual_framework_eval" / ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = [x.strip() for x in args.sensitivity_modes.split(",") if x.strip()]
    selected_codes = [x.strip() for x in args.query_codes.split(",") if x.strip()]
    query_items = [(k, SMARTWATCH_QUERIES[k]) for k in selected_codes if k in SMARTWATCH_QUERIES]
    if not query_items:
        raise RuntimeError("No valid query codes selected.")
    configs = build_sensitivity_configs(args.config_set)

    # 1) Sensitivity on baseline server (no condition file)
    base_handle = None
    ab_handle = None
    try:
        print("[INFO] Starting baseline server...")
        base_handle = start_server(args.base_port, condition_file=None)
        base_url = f"http://127.0.0.1:{args.base_port}"
        sens_rows, sens_summary = run_sensitivity(out_dir, api_key, args.llm, base_url, args.runs, modes, query_items, configs)

        # 2) Highlight AB on condition-aware server
        print("[INFO] Starting condition server...")
        ab_handle = start_server(args.base_port + 1, condition_file="configs/experiments/example_label_framing.yaml")
        ab_url = f"http://127.0.0.1:{args.base_port + 1}"
        ab_rows, ab_summary = run_highlight_ab(out_dir, api_key, args.llm, ab_url, args.runs, query_items)

    finally:
        stop_server(base_handle)
        stop_server(ab_handle)

    all_rows = sens_rows + ab_rows
    write_csv(out_dir / "records_plus_toolcalls.csv", all_rows)
    write_csv(out_dir / "param_sensitivity_summary.csv", sens_summary)
    (out_dir / "summary_by_condition_mode.json").write_text(json.dumps(ab_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    sens_rank = compute_sensitivity_rank(sens_summary)
    write_csv(out_dir / "param_sensitivity_rank.csv", sens_rank)
    render_report(out_dir, args.llm, args.runs, sens_summary, sens_rank, ab_summary)

    print(f"[DONE] output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
