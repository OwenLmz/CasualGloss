#!/usr/bin/env python3
"""Plot final v2.2 training and locked-test latency-quality results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


COLORS = {
    "v1": "#5B7083",
    "v2_1": "#B7791F",
    "v2_2_1": "#167D77",
    "v2_2_2": "#2F6DB0",
    "primary": "#167D77",
    "recovery": "#2F6DB0",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def history_points(path: Path, metric: str) -> tuple[list[int], list[float]]:
    steps, values = [], []
    if not path.is_file():
        return steps, values
    for row in read_jsonl(path):
        if row.get("event") != "validation":
            continue
        validation = row.get("validation", {})
        if validation.get(metric) is not None:
            steps.append(int(row["global_step"]))
            values.append(float(validation[metric]))
    return steps, values


def style_axis(axis, title: str, ylabel: str) -> None:
    axis.set_title(title, loc="left", fontsize=12, fontweight="bold")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#D8DEE6", linewidth=0.8, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--comparison-report", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    args = parser.parse_args()

    primary = args.train_root / "runs/qwen2_5_7b_lora_semantic_causal_v2_2_1_treebank_full/training_history.jsonl"
    recovery = args.train_root / "runs/qwen2_5_7b_lora_semantic_causal_v2_2_2_recovery_full/training_history.jsonl"
    comparison = read_json(args.comparison_report)
    decision = read_json(args.decision)
    selected = decision.get("selected_candidate") or decision.get("fallback")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": "#243447",
            "xtick.color": "#4A5568",
            "ytick.color": "#4A5568",
            "text.color": "#1A2B3C",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    figure.suptitle(
        "Semantic Causal Alignment: training and locked-test evaluation",
        fontsize=16,
        fontweight="bold",
    )

    for name, path in (("primary", primary), ("recovery", recovery)):
        steps, values = history_points(path, "pseudo_action_loss")
        if steps:
            axes[0, 0].plot(
                steps, values, marker="o", linewidth=2, markersize=4,
                color=COLORS[name], label=name,
            )
    style_axis(axes[0, 0], "A  Validation loss", "Cross-entropy")
    axes[0, 0].set_xlabel("Optimizer updates")
    axes[0, 0].legend(frameon=False)

    for name, path in (("primary", primary), ("recovery", recovery)):
        steps, values = history_points(
            path, "pseudo_action_sequence_exact_match_teacher_forced"
        )
        if steps:
            axes[0, 1].plot(
                steps, values, marker="o", linewidth=2, markersize=4,
                color=COLORS[name], label=name,
            )
    style_axis(axes[0, 1], "B  Teacher-forced action sequence exact", "Ratio")
    axes[0, 1].set_xlabel("Optimizer updates")
    axes[0, 1].set_ylim(0, 1.0)
    axes[0, 1].legend(frameon=False)

    versions = comparison["versions"]
    names = [name for name in ("v1", "v2_1", "v2_2_1", "v2_2_2") if name in versions]
    labels = [name.replace("_", ".") for name in names]
    x = list(range(len(names)))
    width = 0.36
    sequence = [versions[name]["means"]["sequence_similarity"] for name in names]
    token_f1 = [versions[name]["means"]["token_f1"] for name in names]
    axes[1, 0].bar(
        [value - width / 2 for value in x], sequence, width,
        color="#167D77", label="Sequence similarity",
    )
    axes[1, 0].bar(
        [value + width / 2 for value in x], token_f1, width,
        color="#D9821B", label="Token F1",
    )
    style_axis(axes[1, 0], "C  Locked pseudo-gloss quality", "Score")
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_ylim(0, 1.0)
    axes[1, 0].legend(frameon=False, ncols=2, fontsize=9)

    laal = [versions[name]["means"]["laal"] for name in names]
    reference_al = [versions[name]["means"]["reference_length_al"] for name in names]
    axes[1, 1].bar(
        [value - width / 2 for value in x], laal, width,
        color="#2F6DB0", label="LAAL",
    )
    axes[1, 1].bar(
        [value + width / 2 for value in x], reference_al, width,
        color="#A8553A", label="Reference-length AL",
    )
    style_axis(axes[1, 1], "D  Locked latency", "Source tokens; lower is better")
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].legend(frameon=False, ncols=2, fontsize=9)

    if selected in names:
        selected_index = names.index(selected)
        for axis in (axes[1, 0], axes[1, 1]):
            axis.text(
                selected_index, axis.get_ylim()[1] * 0.96, "selected",
                ha="center", va="top", fontsize=8, color="#167D77",
                fontweight="bold",
            )

    figure.text(
        0.01,
        0.005,
        "Quality is measured against weak pseudo-gloss references, not expert gold ASL.",
        fontsize=9,
        color="#6B7280",
    )
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_png, dpi=220, bbox_inches="tight")
    figure.savefig(args.output_pdf, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps({"png": str(args.output_png), "pdf": str(args.output_pdf)}, indent=2))


if __name__ == "__main__":
    main()
