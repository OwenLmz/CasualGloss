#!/usr/bin/env python3
"""Create a compact research plot for the Semantic v2.1 Full run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.history.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    train = [row for row in rows if row.get("event") == "train_update"]
    valid = [row for row in rows if row.get("event") == "validation"]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "loss", "token_accuracy", "sequence_exact_match"])
        for row in valid:
            metrics = row["validation"]
            writer.writerow([
                row["global_step"], metrics["pseudo_action_loss"],
                metrics["pseudo_action_token_accuracy_teacher_forced"],
                metrics["pseudo_action_sequence_exact_match_teacher_forced"],
            ])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot([row["global_step"] for row in train], [row["pseudo_action_train_loss"] for row in train], color="#6c8ebf", alpha=0.55, linewidth=1.2, label="Train loss (logged)")
    axes[0].plot([row["global_step"] for row in valid], [row["validation"]["pseudo_action_loss"] for row in valid], color="#c84b31", marker="o", linewidth=2.0, label="Validation loss")
    axes[0].set(title="Pseudo-action loss", xlabel="Optimizer step", ylabel="Cross-entropy loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].plot([row["global_step"] for row in valid], [row["validation"]["pseudo_action_token_accuracy_teacher_forced"] for row in valid], color="#168c82", marker="o", linewidth=2.0, label="Token accuracy")
    axes[1].plot([row["global_step"] for row in valid], [row["validation"]["pseudo_action_sequence_exact_match_teacher_forced"] for row in valid], color="#e39a28", marker="s", linewidth=2.0, label="Sequence exact match")
    axes[1].set(title="Held-out validation agreement", xlabel="Optimizer step", ylabel="Pseudo-label metric", ylim=(0.3, 1.0))
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.suptitle("Semantic Causal Alignment v2.1 - Full Fine-tune 2", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01, "Metrics measure agreement with pseudo causal actions, not gold ASL accuracy.", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
