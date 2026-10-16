#!/usr/bin/env python3
"""Create publication-style plots for the unified locked-500 comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"v1": "#60758A", "v2_1": "#D47A19", "v2_2": "#16867D"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def style(axis, title: str) -> None:
    axis.set_title(title, loc="left", fontsize=12, fontweight="bold")
    axis.grid(axis="y", color="#D9E0E7", linewidth=0.8, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    args = parser.parse_args()

    report = read_json(args.report)
    versions = report["versions"]
    names = ["v1", "v2_1", "v2_2"]
    labels = ["v1", "v2.1", "v2.2"]
    comparison = report["paired_comparisons"]["v2_2_vs_v1"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelcolor": "#243447",
        "xtick.color": "#4A5568",
        "ytick.color": "#4A5568",
        "text.color": "#1A2B3C",
    })
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    figure.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.13, hspace=0.48, wspace=0.28)
    figure.suptitle("Unified locked-500 comparison", fontsize=17, fontweight="bold")

    quality_metrics = ["sequence_similarity", "token_f1", "completed_after_full_source"]
    quality_labels = ["Sequence\nsimilarity", "Token F1", "Valid\ncompletion"]
    x = np.arange(len(quality_metrics))
    width = 0.24
    for index, name in enumerate(names):
        values = [versions[name]["means"][metric] for metric in quality_metrics]
        axes[0, 0].bar(x + (index - 1) * width, values, width, color=COLORS[name], label=labels[index])
    axes[0, 0].set_xticks(x, quality_labels)
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_ylabel("Ratio; higher is better")
    axes[0, 0].legend(frameon=False, ncols=3)
    style(axes[0, 0], "A  Pseudo-gloss quality and completion")

    latency_metrics = ["reference_length_al", "laal"]
    latency_counts = "/".join(
        str(versions[name]["valid_counts"]["laal"]) for name in names
    )
    latency_labels = [
        f"Reference-length AL\n(defined n={latency_counts})",
        f"LAAL\n(defined n={latency_counts})",
    ]
    x = np.arange(len(latency_metrics))
    for index, name in enumerate(names):
        values = [versions[name]["means"][metric] for metric in latency_metrics]
        axes[0, 1].bar(x + (index - 1) * width, values, width, color=COLORS[name], label=labels[index])
    axes[0, 1].set_xticks(x, latency_labels)
    axes[0, 1].set_ylabel("Source tokens; lower is better")
    axes[0, 1].legend(frameon=False, ncols=3)
    style(axes[0, 1], "B  Standard latency")

    quality_delta_metrics = ["sequence_similarity", "token_f1", "completed_after_full_source"]
    quality_delta_labels = ["Sequence similarity", "Token F1", "Completion"]
    y = np.arange(len(quality_delta_metrics))
    means = [comparison[metric]["mean_delta"] for metric in quality_delta_metrics]
    low = [mean - comparison[metric]["ci95_lower"] for mean, metric in zip(means, quality_delta_metrics)]
    high = [comparison[metric]["ci95_upper"] - mean for mean, metric in zip(means, quality_delta_metrics)]
    axes[1, 0].errorbar(means, y, xerr=[low, high], fmt="o", color=COLORS["v2_2"], capsize=4, linewidth=2)
    axes[1, 0].axvline(0, color="#7A8795", linewidth=1)
    axes[1, 0].set_yticks(y, quality_delta_labels)
    axes[1, 0].set_xlabel("v2.2 minus v1; positive is better")
    style(axes[1, 0], "C  Paired quality deltas with 95% CI")

    latency_delta_metrics = ["reference_length_al", "laal"]
    latency_delta_labels = ["Reference-length AL", "LAAL"]
    y = np.arange(len(latency_delta_metrics))
    means = [comparison[metric]["mean_delta"] for metric in latency_delta_metrics]
    low = [mean - comparison[metric]["ci95_lower"] for mean, metric in zip(means, latency_delta_metrics)]
    high = [comparison[metric]["ci95_upper"] - mean for mean, metric in zip(means, latency_delta_metrics)]
    axes[1, 1].errorbar(means, y, xerr=[low, high], fmt="o", color="#2E6FAD", capsize=4, linewidth=2)
    axes[1, 1].axvline(0, color="#7A8795", linewidth=1)
    axes[1, 1].set_yticks(y, latency_delta_labels)
    axes[1, 1].set_xlabel("v2.2 minus v1; negative is better")
    style(axes[1, 1], "D  Paired latency deltas with 95% CI")

    figure.text(
        0.01,
        0.025,
        "All systems use the same 500 English inputs and fixed Qwen pseudo-gloss references; references are not expert gold ASL.",
        fontsize=9,
        color="#657382",
    )
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_png, dpi=240, bbox_inches="tight")
    figure.savefig(args.output_pdf, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps({"png": str(args.output_png), "pdf": str(args.output_pdf)}, indent=2))


if __name__ == "__main__":
    main()
