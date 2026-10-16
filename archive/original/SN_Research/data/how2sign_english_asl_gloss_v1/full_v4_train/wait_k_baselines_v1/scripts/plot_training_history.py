#!/usr/bin/env python3
"""Plot train/validation pseudo-action metrics for a Wait-k run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.history.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    train = [row for row in rows if row.get("event") == "train_update"]
    valid = [row for row in rows if row.get("event") == "validation"]
    if not train or not valid:
        raise ValueError("Training history does not yet contain train and validation events")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with args.output_prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "loss", "token_accuracy", "sequence_exact_match", "stage"])
        for row in valid:
            metrics = row["validation"]
            writer.writerow([
                row["global_step"],
                metrics["pseudo_action_loss"],
                metrics["pseudo_action_token_accuracy_teacher_forced"],
                metrics["pseudo_action_sequence_exact_match_teacher_forced"],
                row.get("stage", ""),
            ])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    axes[0].plot(
        [row["global_step"] for row in train],
        [row["pseudo_action_train_loss"] for row in train],
        color="#4c78a8",
        linewidth=1.4,
        label="Train loss",
    )
    axes[0].plot(
        [row["global_step"] for row in valid],
        [row["validation"]["pseudo_action_loss"] for row in valid],
        color="#e45756",
        marker="o",
        linewidth=1.8,
        label="Validation loss",
    )
    axes[0].set(xlabel="Optimizer step", ylabel="Cross-entropy loss", title="Pseudo-action loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(
        [row["global_step"] for row in valid],
        [row["validation"]["pseudo_action_token_accuracy_teacher_forced"] for row in valid],
        color="#16877b",
        marker="o",
        linewidth=1.8,
        label="Token accuracy",
    )
    axes[1].plot(
        [row["global_step"] for row in valid],
        [row["validation"]["pseudo_action_sequence_exact_match_teacher_forced"] for row in valid],
        color="#f2a541",
        marker="s",
        linewidth=1.8,
        label="Sequence exact match",
    )
    axes[1].set(xlabel="Optimizer step", ylabel="Pseudo-label agreement", ylim=(0, 1), title="Held-out validation")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle(args.title, fontsize=13, fontweight="bold")
    fig.text(0.5, -0.02, "Metrics use pseudo causal-action labels, not expert gold ASL.", ha="center", fontsize=9, color="#555555")
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
