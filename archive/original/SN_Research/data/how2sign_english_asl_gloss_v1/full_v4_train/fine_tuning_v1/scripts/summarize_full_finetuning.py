#!/usr/bin/env python3
"""Compare full base/LoRA evaluations and create figures and report text."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from finetune_common import read_json, read_jsonl, write_json


def metric_delta(after: float, before: float) -> float:
    return round(after - before, 8)


def bootstrap_delta(
    base: np.ndarray, best: np.ndarray, seed: int, samples: int = 10000
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        chosen = rng.integers(0, len(base), size=len(base))
        deltas[index] = float(np.mean(best[chosen] - base[chosen]))
    return {
        "mean_delta": round(float(np.mean(best - base)), 8),
        "ci95_low": round(float(np.quantile(deltas, 0.025)), 8),
        "ci95_high": round(float(np.quantile(deltas, 0.975)), 8),
        "bootstrap_probability_delta_gt_0": round(float(np.mean(deltas > 0)), 8),
        "bootstrap_samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-history", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--split-report", type=Path, required=True)
    parser.add_argument("--base-evaluation", type=Path, required=True)
    parser.add_argument("--best-evaluation", type=Path, required=True)
    parser.add_argument("--base-test-predictions", type=Path, required=True)
    parser.add_argument("--best-test-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    history = read_jsonl(args.training_history)
    training_summary = read_json(args.training_summary)
    split_report = read_json(args.split_report)
    base = read_json(args.base_evaluation)
    best = read_json(args.best_evaluation)
    base_predictions = {
        row["id"]: row for row in read_jsonl(args.base_test_predictions)
    }
    best_predictions = {
        row["id"]: row for row in read_jsonl(args.best_test_predictions)
    }
    if set(base_predictions) != set(best_predictions):
        raise SystemExit("Base and best prediction IDs differ")

    comparison: dict[str, Any] = {
        "scientific_status": "pseudo_label_comparison_not_gold_asl",
        "split_counts": split_report["counts"],
        "best_checkpoint": training_summary["best_checkpoint"],
        "metrics": {},
    }
    for split in ("validation", "test"):
        base_tf = base["teacher_forced"][split]
        best_tf = best["teacher_forced"][split]
        base_gen = base["generation"][split]
        best_gen = best["generation"][split]
        comparison["metrics"][split] = {
            "base": {"teacher_forced": base_tf, "generation": base_gen},
            "best_lora": {"teacher_forced": best_tf, "generation": best_gen},
            "delta_best_minus_base": {
                "pseudo_label_loss": metric_delta(
                    float(best_tf["pseudo_label_loss"]),
                    float(base_tf["pseudo_label_loss"]),
                ),
                "teacher_forced_token_accuracy": metric_delta(
                    float(best_tf["pseudo_label_token_accuracy_teacher_forced"]),
                    float(base_tf["pseudo_label_token_accuracy_teacher_forced"]),
                ),
                "generation_exact_match": metric_delta(
                    float(best_gen["pseudo_label_generation_exact_match"]),
                    float(base_gen["pseudo_label_generation_exact_match"]),
                ),
                "generation_sequence_similarity": metric_delta(
                    float(best_gen["pseudo_label_generation_token_sequence_similarity"]),
                    float(base_gen["pseudo_label_generation_token_sequence_similarity"]),
                ),
            },
        }

    ids = sorted(base_predictions)
    base_similarity = np.array(
        [base_predictions[row_id]["pseudo_label_token_sequence_similarity"] for row_id in ids],
        dtype=np.float64,
    )
    best_similarity = np.array(
        [best_predictions[row_id]["pseudo_label_token_sequence_similarity"] for row_id in ids],
        dtype=np.float64,
    )
    base_exact = np.array(
        [float(base_predictions[row_id]["pseudo_label_exact_match"]) for row_id in ids],
        dtype=np.float64,
    )
    best_exact = np.array(
        [float(best_predictions[row_id]["pseudo_label_exact_match"]) for row_id in ids],
        dtype=np.float64,
    )
    comparison["paired_bootstrap_test_generation"] = {
        "sequence_similarity": bootstrap_delta(
            base_similarity, best_similarity, args.seed
        ),
        "exact_match": bootstrap_delta(base_exact, best_exact, args.seed + 1),
    }
    comparison["important_limitation"] = (
        "Improvements indicate closer reproduction of generated pseudo labels, "
        "not verified ASL correctness."
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "full_finetuning_comparison.json", comparison)

    train_events = [row for row in history if row["event"] == "train_update"]
    validation_events = [row for row in history if row["event"] == "validation"]
    train_steps = np.array([row["global_step"] for row in train_events])
    train_losses = np.array([row["pseudo_label_train_loss"] for row in train_events])
    if len(train_losses) >= 5:
        smooth = np.convolve(train_losses, np.ones(5) / 5, mode="same")
        smooth[:2] = train_losses[:2]
        smooth[-2:] = train_losses[-2:]
    else:
        smooth = train_losses
    validation_steps = np.array([row["global_step"] for row in validation_events])
    validation_losses = np.array(
        [row["validation"]["pseudo_label_loss"] for row in validation_events]
    )
    validation_accuracy = np.array(
        [
            row["validation"]["pseudo_label_token_accuracy_teacher_forced"]
            for row in validation_events
        ]
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    axes[0].plot(train_steps, train_losses, color="#9CB8C5", linewidth=1, alpha=0.55, label="Train loss")
    axes[0].plot(train_steps, smooth, color="#176B87", linewidth=2.2, label="Train loss (smoothed)")
    axes[0].plot(validation_steps, validation_losses, color="#B44B4B", marker="o", linewidth=2, label="Validation loss")
    axes[0].set_title("Pseudo-label loss")
    axes[0].set_xlabel("Optimizer update step")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(validation_steps, validation_accuracy, color="#287D5A", marker="o", linewidth=2.2)
    axes[1].set_title("Validation token accuracy")
    axes[1].set_xlabel("Optimizer update step")
    axes[1].set_ylabel("Teacher-forced pseudo-label accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)
    fig.suptitle("Full How2Sign English-to-pseudo-ASL-gloss LoRA fine-tuning", fontsize=15, fontweight="bold")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(args.output_dir / f"full_finetuning_curves.{suffix}", dpi=220)
    plt.close(fig)

    with (args.output_dir / "curve_source_data.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["series", "global_step", "value"])
        for step, value in zip(train_steps, train_losses):
            writer.writerow(["train_loss", int(step), float(value)])
        for step, value in zip(validation_steps, validation_losses):
            writer.writerow(["validation_loss", int(step), float(value)])
        for step, value in zip(validation_steps, validation_accuracy):
            writer.writerow(["validation_token_accuracy", int(step), float(value)])

    test_metrics = comparison["metrics"]["test"]
    test_delta = test_metrics["delta_best_minus_base"]
    results_md = f"""# Full Fine-tune 1 results

## Setup

- Train/validation/test: **{split_report['counts']['train']} / {split_report['counts']['validation']} / {split_report['counts']['test']}**.
- Splits are request-disjoint and How2Sign source-video-disjoint.
- Model: `Qwen/Qwen2.5-7B-Instruct` with rank-16 LoRA on `q_proj` and `v_proj`.
- Best checkpoint: step **{training_summary['best_checkpoint']['global_step']}**, selected only by validation pseudo-label loss.

## Held-out test comparison

| Metric | Base model | Best LoRA | Delta |
|---|---:|---:|---:|
| Pseudo-label loss | {test_metrics['base']['teacher_forced']['pseudo_label_loss']:.4f} | {test_metrics['best_lora']['teacher_forced']['pseudo_label_loss']:.4f} | {test_delta['pseudo_label_loss']:+.4f} |
| Teacher-forced token accuracy | {test_metrics['base']['teacher_forced']['pseudo_label_token_accuracy_teacher_forced']:.4f} | {test_metrics['best_lora']['teacher_forced']['pseudo_label_token_accuracy_teacher_forced']:.4f} | {test_delta['teacher_forced_token_accuracy']:+.4f} |
| Greedy generation exact match | {test_metrics['base']['generation']['pseudo_label_generation_exact_match']:.4f} | {test_metrics['best_lora']['generation']['pseudo_label_generation_exact_match']:.4f} | {test_delta['generation_exact_match']:+.4f} |
| Greedy generation sequence similarity | {test_metrics['base']['generation']['pseudo_label_generation_token_sequence_similarity']:.4f} | {test_metrics['best_lora']['generation']['pseudo_label_generation_token_sequence_similarity']:.4f} | {test_delta['generation_sequence_similarity']:+.4f} |

## Interpretation

These metrics test reproduction of deterministic pseudo glosses on unseen
How2Sign video groups. They are not gold ASL accuracy and do not evaluate sign
video generation or qualified linguistic correctness.
"""
    (args.output_dir / "RESULTS.md").write_text(results_md, encoding="utf-8")

    professor_md = f"""# Professor update: Fine-tune 1

- Constructed leakage-controlled splits from 30,102 unique How2Sign English/pseudo-gloss pairs: {split_report['counts']['train']} train, {split_report['counts']['validation']} validation, and {split_report['counts']['test']} test.
- Fine-tuned the fixed Qwen2.5-7B-Instruct checkpoint with rank-16 LoRA using four RTX 5090 GPUs.
- Selected checkpoint step {training_summary['best_checkpoint']['global_step']} using validation pseudo-label loss only.
- On the unseen test split, loss changed from {test_metrics['base']['teacher_forced']['pseudo_label_loss']:.4f} to {test_metrics['best_lora']['teacher_forced']['pseudo_label_loss']:.4f}, token accuracy from {test_metrics['base']['teacher_forced']['pseudo_label_token_accuracy_teacher_forced']:.4f} to {test_metrics['best_lora']['teacher_forced']['pseudo_label_token_accuracy_teacher_forced']:.4f}, and greedy sequence similarity from {test_metrics['base']['generation']['pseudo_label_generation_token_sequence_similarity']:.4f} to {test_metrics['best_lora']['generation']['pseudo_label_generation_token_sequence_similarity']:.4f}.
- These are pseudo-label agreement metrics, not gold ASL accuracy. Causal-alignment fine-tuning remains the next stage.
"""
    (args.output_dir / "PROFESSOR_UPDATE.md").write_text(
        professor_md, encoding="utf-8"
    )


if __name__ == "__main__":
    main()

