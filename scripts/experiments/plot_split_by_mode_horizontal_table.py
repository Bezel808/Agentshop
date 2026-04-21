#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _query_title(meta_queries: dict, query_key: str) -> str:
    human = meta_queries.get(query_key, query_key)
    return f"{query_key}\n{human}"


def _plot_mode(ax, dist_map: dict, mode: str, color: str):
    items = sorted(
        ((pid, info["count"], info["probability"]) for pid, info in dist_map.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    labels = [x[0] for x in items]
    probs = [x[2] * 100 for x in items]
    counts = [x[1] for x in items]

    y = list(range(len(labels)))
    ax.barh(y, probs, color=color, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_xlabel("Probability (%)")
    ax.set_title(f"{mode} (unique={len(labels)})", fontsize=11)

    for yi, p, c in zip(y, probs, counts):
        ax.text(min(p + 1.2, 98), yi, f"{p:.0f}% ({c})", va="center", fontsize=8)


def main():
    parser = argparse.ArgumentParser(
        description="Merge 8 split charts (4 queries x 2 modes) into one horizontal-bar table figure."
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path(
            "experiment_results/full_4queries_100x2_vlmverbal_20260420_200650/attempt_1/summary.json"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "analysis_outputs/full_4queries_split_by_mode_20260421/full_4queries_split_by_mode_horizontal_table.png"
        ),
    )
    args = parser.parse_args()

    data = _load_summary(args.summary_json)
    dist = data["distribution"]
    meta_queries = data.get("meta", {}).get("queries", {})

    query_order = list(meta_queries.keys()) or list(dist.keys())
    modes = ["verbal", "visual"]
    mode_colors = {"verbal": "#d55e00", "visual": "#0072b2"}

    fig, axes = plt.subplots(
        nrows=len(query_order),
        ncols=2,
        figsize=(16, max(14, 3.4 * len(query_order))),
        constrained_layout=True,
    )
    if len(query_order) == 1:
        axes = [axes]

    for r, q in enumerate(query_order):
        for c, m in enumerate(modes):
            ax = axes[r][c] if len(query_order) > 1 else axes[c]
            _plot_mode(ax, dist[q][m], m, mode_colors[m])
            if c == 0:
                ax.set_ylabel(_query_title(meta_queries, q), fontsize=10)
            else:
                ax.set_ylabel("")

    fig.suptitle(
        "Recommendation Distribution by Query and Mode (Horizontal Bars)\n"
        "Rows = Query, Columns = Mode (verbal / visual), labels = probability and count",
        fontsize=14,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220)
    print(args.out)


if __name__ == "__main__":
    main()

