#!/usr/bin/env python3
"""Create paper-ready quality/latency plots from the unified comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "v1": "#46647f",
    "v2_1": "#d98017",
    "v2_2": "#16877b",
    "wait_k1": "#8c55a2",
    "wait_k3": "#4c78a8",
    "wait_k5": "#54a24b",
    "wait_k7": "#e45756",
}


def display_name(name: str) -> str:
    return name.replace("v2_", "v2.").replace("wait_k", "Wait-")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.comparison.read_text(encoding="utf-8"))
    names = list(report["systems"])
    means = {name: report["systems"][name]["means"] for name in names}
    colors = [COLORS.get(name, "#777777") for name in names]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 180,
    })
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True)

    ax = axes[0]
    for name, color in zip(names, colors):
        ax.scatter(means[name]["laal"], means[name]["sequence_similarity"], s=70, color=color, zorder=3)
        ax.annotate(display_name(name), (means[name]["laal"], means[name]["sequence_similarity"]), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("LAAL (lower is better)")
    ax.set_ylabel("Sequence similarity (higher is better)")
    ax.set_title("Quality-latency frontier")
    ax.grid(alpha=0.25)

    x = np.arange(len(names))
    ax = axes[1]
    width = 0.36
    ax.bar(x - width / 2, [means[name]["sequence_similarity"] for name in names], width, label="Sequence similarity", color="#4c78a8")
    ax.bar(x + width / 2, [means[name]["token_f1"] for name in names], width, label="Token F1", color="#54a24b")
    ax.set_xticks(x, [display_name(name) for name in names], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Pseudo-gloss quality")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[2]
    ax.bar(x - width / 2, [means[name]["completed_after_full_source"] for name in names], width, label="Completion", color="#16877b")
    ax.bar(x + width / 2, [means[name]["invalid_ratio"] for name in names], width, label="Invalid ratio", color="#e45756")
    ax.set_xticks(x, [display_name(name) for name in names], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Rollout reliability")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Locked-500 comparison on identical pseudo-gloss references", fontsize=13, fontweight="bold")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    print(args.output_prefix.with_suffix(".png"))
    print(args.output_prefix.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
