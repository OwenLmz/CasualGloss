#!/usr/bin/env python3
"""Compare full-data Seed 2 with the locked full-data Seed 1 result."""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def safe_mean(values: list[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    return round(mean(filtered), 8) if filtered else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "parent_count": len(rows),
        "end_completion_rate": safe_mean([float(row["completed_with_end"]) for row in rows]),
        "premature_end_rate": safe_mean([float(row["premature_end"]) for row in rows]),
        "maximum_actions_reached_rate": safe_mean(
            [float(row["termination_reason"] == "maximum_actions_reached") for row in rows]
        ),
        "pseudo_gloss_exact_match_rate": safe_mean(
            [float(row["pseudo_gloss_exact_match"]) for row in rows]
        ),
        "mean_pseudo_gloss_sequence_similarity": safe_mean(
            [row["pseudo_gloss_sequence_similarity"] for row in rows]
        ),
        "mean_pseudo_gloss_token_precision": safe_mean(
            [row["pseudo_gloss_token_precision"] for row in rows]
        ),
        "mean_pseudo_gloss_token_recall": safe_mean(
            [row["pseudo_gloss_token_recall"] for row in rows]
        ),
        "mean_pseudo_gloss_token_f1": safe_mean([row["pseudo_gloss_token_f1"] for row in rows]),
        "mean_hallucinated_token_ratio": safe_mean(
            [row["hallucinated_token_ratio"] for row in rows]
        ),
        "mean_invalid_action_ratio": safe_mean([row["invalid_action_ratio"] for row in rows]),
        "mean_wait_action_ratio": safe_mean([row["wait_action_ratio"] for row in rows]),
        "mean_word_position_normalized_average_lagging": safe_mean(
            [row["word_position_normalized_average_lagging"] for row in rows]
        ),
        "mean_action_count": safe_mean([row["action_count"] for row in rows]),
    }


def paired_bootstrap_delta(
    seed1: dict[str, dict[str, Any]],
    seed2: dict[str, dict[str, Any]],
    ids: list[str],
    key: str,
    seed: int,
    resamples: int = 5000,
) -> dict[str, Any]:
    paired_ids = [
        row_id
        for row_id in ids
        if seed1[row_id].get(key) is not None and seed2[row_id].get(key) is not None
    ]
    observed = mean(
        float(seed2[row_id][key]) - float(seed1[row_id][key]) for row_id in paired_ids
    )
    rng = random.Random(seed)
    deltas = []
    for _ in range(resamples):
        sample = [paired_ids[rng.randrange(len(paired_ids))] for _ in paired_ids]
        deltas.append(
            mean(float(seed2[row_id][key]) - float(seed1[row_id][key]) for row_id in sample)
        )
    deltas.sort()
    return {
        "paired_parent_count": len(paired_ids),
        "delta_seed2_minus_seed1": round(observed, 8),
        "paired_bootstrap_95_ci": [
            round(deltas[int(0.025 * resamples)], 8),
            round(deltas[min(resamples - 1, int(0.975 * resamples))], 8),
        ],
        "resamples": resamples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--seed1-action", type=Path, required=True)
    parser.add_argument("--seed2-action", type=Path, required=True)
    parser.add_argument("--seed1-rollouts", type=Path, required=True)
    parser.add_argument("--seed2-rollouts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    history = read_jsonl(args.history)
    training = read_json(args.training_summary)
    preflight = read_json(args.preflight_report)
    seed1_action = read_json(args.seed1_action)["greedy_generation_on_gold_states"]
    seed2_action = read_json(args.seed2_action)["greedy_generation_on_gold_states"]
    seed1_rows = read_jsonl(args.seed1_rollouts)
    seed2_rows = read_jsonl(args.seed2_rollouts)
    seed1_map = {row["id"]: row for row in seed1_rows}
    seed2_map = {row["id"]: row for row in seed2_rows}
    if set(seed1_map) != set(seed2_map) or len(seed1_map) != 500:
        raise SystemExit("Seed 1 and Seed 2 must use the same locked 500 parents")
    ids = sorted(seed1_map)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    free_running = {
        "seed1_full": aggregate([seed1_map[row_id] for row_id in ids]),
        "seed2_full": aggregate([seed2_map[row_id] for row_id in ids]),
    }
    bootstrap = {
        "sequence_similarity": paired_bootstrap_delta(
            seed1_map, seed2_map, ids, "pseudo_gloss_sequence_similarity", 20260720
        ),
        "token_f1": paired_bootstrap_delta(
            seed1_map, seed2_map, ids, "pseudo_gloss_token_f1", 20260721
        ),
        "end_completion": paired_bootstrap_delta(
            seed1_map, seed2_map, ids, "completed_with_end", 20260722
        ),
        "invalid_action_ratio": paired_bootstrap_delta(
            seed1_map, seed2_map, ids, "invalid_action_ratio", 20260723
        ),
    }
    action_keys = (
        "pseudo_action_exact_accuracy",
        "action_type_accuracy",
        "write_token_exact_accuracy",
        "premature_write_on_wait_rate",
        "over_wait_on_write_rate",
        "invalid_output_rate",
        "parent_macro_action_accuracy",
    )
    action_delta = {
        key: round(float(seed2_action[key]) - float(seed1_action[key]), 8)
        for key in action_keys
    }
    seed2_free = free_running["seed2_full"]
    gates = {
        "preflight_ok": bool(preflight.get("ok")),
        "validation_loss_improved_over_finetune1": (
            training["best_checkpoint"]["validation_pseudo_action_loss"]
            < training["baseline_validation"]["pseudo_action_loss"]
        ),
        "end_completion_at_least_0_98": seed2_free["end_completion_rate"] >= 0.98,
        "timeout_at_most_0_02": seed2_free["maximum_actions_reached_rate"] <= 0.02,
        "invalid_action_at_most_0_01": seed2_free["mean_invalid_action_ratio"] <= 0.01,
    }
    decision = (
        "full_seed2_replication_completed_ready_for_interpretation"
        if all(gates.values())
        else "full_seed2_operational_regression_requires_inspection"
    )
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "full_data_seed_replication_pseudo_labels_not_gold_asl",
        "controlled_design": {
            "training_seed_pair": [20260720, 20260721],
            "same_finetune1_initialization": True,
            "same_397700_training_actions": True,
            "same_locked_500_parent_test": True,
            "preflight": preflight,
        },
        "training_seed2": training,
        "gold_state_test": {
            "seed1_full": seed1_action,
            "seed2_full": seed2_action,
            "delta_seed2_minus_seed1": action_delta,
        },
        "free_running_locked_500_parent_test": free_running,
        "paired_bootstrap_seed2_minus_seed1": bootstrap,
        "operational_gates": gates,
        "operational_gates_all_passed": all(gates.values()),
        "recommended_decision": decision,
        "interpretation_boundary": (
            "This compares two full-data random seeds on the same locked test parents. "
            "Targets remain pseudo gloss and pseudo timing, not gold ASL annotations."
        ),
    }
    write_json(args.output_dir / "metrics_comparison.json", result)

    train_points = [row for row in history if row["event"] == "train_update"]
    validation_points = [row for row in history if row["event"] == "validation"]
    with (args.output_dir / "training_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["event", "global_step", "train_loss", "validation_loss"])
        for row in train_points:
            writer.writerow(["train", row["global_step"], row["pseudo_action_train_loss"], ""])
        for row in validation_points:
            writer.writerow([
                "validation",
                row["global_step"],
                "",
                row["validation"]["pseudo_action_loss"],
            ])

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12})
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    axes[0].plot(
        [row["global_step"] for row in train_points],
        [row["pseudo_action_train_loss"] for row in train_points],
        color="#176B87",
        linewidth=1.5,
        label="Train pseudo-action loss",
    )
    axes[0].scatter(
        [row["global_step"] for row in validation_points],
        [row["validation"]["pseudo_action_loss"] for row in validation_points],
        color="#C4473A",
        s=34,
        label="Validation pseudo-action loss",
        zorder=3,
    )
    axes[0].set(title="Full-data Seed 2 loss", xlabel="Optimizer update", ylabel="Cross-entropy loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    labels = ["END completion", "Sequence similarity", "Token F1"]
    keys = ["end_completion_rate", "mean_pseudo_gloss_sequence_similarity", "mean_pseudo_gloss_token_f1"]
    positions = list(range(len(keys)))
    width = 0.32
    for offset, name, color in (
        (-width / 2, "seed1_full", "#7895B2"),
        (width / 2, "seed2_full", "#2A7F62"),
    ):
        axes[1].bar(
            [position + offset for position in positions],
            [free_running[name][key] for key in keys],
            width,
            label=name.replace("_", " "),
            color=color,
        )
    axes[1].set_xticks(positions, labels, rotation=8)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("Pseudo-label metric")
    axes[1].set_title("Locked 500-parent seed replication")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    figure.savefig(args.output_dir / "full_seed_stability.png", dpi=180)
    plt.close(figure)

    seed1_free = free_running["seed1_full"]
    lines = [
        "# Full-data Seed 2 Replication",
        "",
        f"- Training actions: {training['train_action_count']}",
        f"- Optimizer updates: {training['global_update_steps']}",
        f"- Best validation loss: {training['best_checkpoint']['validation_pseudo_action_loss']:.4f}",
        f"- Sequence similarity, Seed 1 -> Seed 2: {seed1_free['mean_pseudo_gloss_sequence_similarity']:.4f} -> {seed2_free['mean_pseudo_gloss_sequence_similarity']:.4f}",
        f"- Token F1, Seed 1 -> Seed 2: {seed1_free['mean_pseudo_gloss_token_f1']:.4f} -> {seed2_free['mean_pseudo_gloss_token_f1']:.4f}",
        f"- Recommendation: `{decision}`",
        "",
        "The within-Seed-2 5K-to-full comparison is reported separately in controlled_full_seed2.",
        "All targets and timing remain pseudo labels, not gold ASL annotations.",
    ]
    (args.output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    professor = [
        "# Professor Update: Full-data Seed 2 Replication",
        "",
        "- Repeated the full-data causal-policy experiment with a new training seed.",
        "- Kept the model revision, Fine-tune 1 initialization, data, split, and hyperparameters fixed.",
        f"- Seed 2 END completion: {seed2_free['end_completion_rate']:.4f}.",
        f"- Seed 2 sequence similarity: {seed2_free['mean_pseudo_gloss_sequence_similarity']:.4f}.",
        f"- Seed 2 token F1: {seed2_free['mean_pseudo_gloss_token_f1']:.4f}.",
        "- Also compared 5K versus full within Seed 2 to replicate the data-scale effect.",
        "- Metrics use pseudo gloss and pseudo timing, not gold ASL labels.",
    ]
    (args.output_dir / "PROFESSOR_UPDATE.md").write_text(
        "\n".join(professor) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
