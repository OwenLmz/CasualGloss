#!/usr/bin/env python3
"""Compare full-scale causal Fine-tune 2 against the locked 5K control."""

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
    baseline: dict[str, dict[str, Any]],
    final: dict[str, dict[str, Any]],
    ids: list[str],
    key: str,
    seed: int,
    resamples: int = 5000,
) -> dict[str, Any]:
    paired_ids = [
        row_id
        for row_id in ids
        if baseline[row_id].get(key) is not None and final[row_id].get(key) is not None
    ]
    observed = mean(
        float(final[row_id][key]) - float(baseline[row_id][key]) for row_id in paired_ids
    )
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(resamples):
        sample = [paired_ids[rng.randrange(len(paired_ids))] for _ in paired_ids]
        deltas.append(
            mean(float(final[row_id][key]) - float(baseline[row_id][key]) for row_id in sample)
        )
    deltas.sort()
    lower = deltas[int(0.025 * resamples)]
    upper = deltas[min(resamples - 1, int(0.975 * resamples))]
    return {
        "paired_parent_count": len(paired_ids),
        "delta_full_minus_5k": round(observed, 8),
        "paired_bootstrap_95_ci": [round(lower, 8), round(upper, 8)],
        "resamples": resamples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--control-action", type=Path, required=True)
    parser.add_argument("--full-action", type=Path, required=True)
    parser.add_argument("--control-rollouts", type=Path, required=True)
    parser.add_argument("--full-rollouts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    history = read_jsonl(args.history)
    training = read_json(args.training_summary)
    data_report = read_json(args.data_report)
    preflight = read_json(args.preflight_report)
    control_action = read_json(args.control_action)
    full_action = read_json(args.full_action)
    control_rows = read_jsonl(args.control_rollouts)
    full_rows = read_jsonl(args.full_rollouts)
    control_map = {row["id"]: row for row in control_rows}
    full_map = {row["id"]: row for row in full_rows}
    if set(control_map) != set(full_map) or len(control_map) != 500:
        raise SystemExit("Full and 5K rollout IDs must be the same locked 500 parents")
    ids = sorted(control_map)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rollout_metrics = {
        "finetune2_5k": aggregate([control_map[row_id] for row_id in ids]),
        "finetune2_full": aggregate([full_map[row_id] for row_id in ids]),
    }
    bootstrap = {
        "sequence_similarity": paired_bootstrap_delta(
            control_map, full_map, ids, "pseudo_gloss_sequence_similarity", 20260720
        ),
        "token_f1": paired_bootstrap_delta(
            control_map, full_map, ids, "pseudo_gloss_token_f1", 20260721
        ),
        "end_completion": paired_bootstrap_delta(
            control_map, full_map, ids, "completed_with_end", 20260722
        ),
        "invalid_action_ratio": paired_bootstrap_delta(
            control_map, full_map, ids, "invalid_action_ratio", 20260723
        ),
    }

    control_gold = control_action["greedy_generation_on_gold_states"]
    full_gold = full_action["greedy_generation_on_gold_states"]
    gold_delta = {
        key: round(float(full_gold[key]) - float(control_gold[key]), 8)
        for key in (
            "pseudo_action_exact_accuracy",
            "action_type_accuracy",
            "write_token_exact_accuracy",
            "premature_write_on_wait_rate",
            "over_wait_on_write_rate",
            "invalid_output_rate",
            "parent_macro_action_accuracy",
        )
    }

    control_free = rollout_metrics["finetune2_5k"]
    full_free = rollout_metrics["finetune2_full"]
    seq_bootstrap = bootstrap["sequence_similarity"]
    f1_bootstrap = bootstrap["token_f1"]
    gates = {
        "preflight_exact_locked_evaluation_match": bool(preflight.get("ok")),
        "validation_loss_improved_over_finetune1": (
            training["best_checkpoint"]["validation_pseudo_action_loss"]
            < training["baseline_validation"]["pseudo_action_loss"]
        ),
        "full_end_completion_at_least_0_98": full_free["end_completion_rate"] >= 0.98,
        "full_timeout_at_most_0_02": full_free["maximum_actions_reached_rate"] <= 0.02,
        "full_invalid_action_at_most_0_01": full_free["mean_invalid_action_ratio"] <= 0.01,
        "sequence_similarity_delta_positive": seq_bootstrap["delta_full_minus_5k"] > 0,
        "sequence_similarity_ci_lower_above_zero": seq_bootstrap["paired_bootstrap_95_ci"][0] > 0,
        "token_f1_delta_positive": f1_bootstrap["delta_full_minus_5k"] > 0,
        "token_f1_ci_lower_above_zero": f1_bootstrap["paired_bootstrap_95_ci"][0] > 0,
        "normalized_lag_increase_at_most_0_05": (
            full_free["mean_word_position_normalized_average_lagging"]
            <= control_free["mean_word_position_normalized_average_lagging"] + 0.05
        ),
    }
    operational_keys = (
        "preflight_exact_locked_evaluation_match",
        "validation_loss_improved_over_finetune1",
        "full_end_completion_at_least_0_98",
        "full_timeout_at_most_0_02",
        "full_invalid_action_at_most_0_01",
        "normalized_lag_increase_at_most_0_05",
    )
    statistical_keys = (
        "sequence_similarity_delta_positive",
        "sequence_similarity_ci_lower_above_zero",
        "token_f1_delta_positive",
        "token_f1_ci_lower_above_zero",
    )
    if all(gates.values()):
        decision = "full_scale_improvement_supported_proceed_to_replication_and_speech_frontend"
    elif all(gates[key] for key in operational_keys):
        decision = "full_model_operationally_stable_but_scale_gain_not_established_run_second_seed"
    else:
        decision = "full_scale_regressed_inspect_before_speech_frontend"

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "full_scale_pseudo_causal_action_experiment_not_gold_asl",
        "controlled_design": {
            "same_finetune1_initialization": True,
            "same_locked_500_parent_test": True,
            "training_actions_5k": 71882,
            "training_actions_full": data_report["action_split_counts"]["train"],
            "preflight": preflight,
        },
        "training": training,
        "gold_state_test": {
            "finetune2_5k": control_gold,
            "finetune2_full": full_gold,
            "delta_full_minus_5k": gold_delta,
        },
        "free_running_locked_500_parent_test": rollout_metrics,
        "paired_bootstrap_full_minus_5k": bootstrap,
        "decision_gates": gates,
        "operational_gates_all_passed": all(gates[key] for key in operational_keys),
        "statistical_scale_gates_all_passed": all(gates[key] for key in statistical_keys),
        "recommended_decision": decision,
        "interpretation_boundary": (
            "The experiment tests scaling of pseudo causal-policy supervision. It does not measure "
            "gold ASL linguistic correctness or speech-input performance."
        ),
    }
    write_json(args.output_dir / "metrics_comparison.json", result)

    train_points = [row for row in history if row["event"] == "train_update"]
    validation_points = [row for row in history if row["event"] == "validation"]
    with (args.output_dir / "training_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["event", "global_step", "train_loss", "validation_loss", "validation_sequence_accuracy"]
        )
        for row in train_points:
            writer.writerow(["train", row["global_step"], row["pseudo_action_train_loss"], "", ""])
        for row in validation_points:
            writer.writerow(
                [
                    "validation",
                    row["global_step"],
                    "",
                    row["validation"]["pseudo_action_loss"],
                    row["validation"]["pseudo_action_sequence_exact_match_teacher_forced"],
                ]
            )

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
    axes[0].set(title="Full Fine-tune 2 loss", xlabel="Optimizer update", ylabel="Cross-entropy loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    labels = ["END completion", "Sequence similarity", "Token F1"]
    keys = ["end_completion_rate", "mean_pseudo_gloss_sequence_similarity", "mean_pseudo_gloss_token_f1"]
    positions = list(range(len(keys)))
    width = 0.32
    for offset, name, color in (
        (-width / 2, "finetune2_5k", "#7895B2"),
        (width / 2, "finetune2_full", "#2A7F62"),
    ):
        axes[1].bar(
            [position + offset for position in positions],
            [rollout_metrics[name][key] for key in keys],
            width,
            label=name.replace("_", " "),
            color=color,
        )
    axes[1].set_xticks(positions, labels, rotation=8)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("Pseudo-label metric")
    axes[1].set_title("Locked 500-parent free-running test")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    figure.savefig(args.output_dir / "fine_tune2_full_results.png", dpi=180)
    plt.close(figure)

    lines = [
        "# Full-scale Fine-tune 2 Controlled Experiment",
        "",
        "## Training",
        "",
        f"- Training actions: {data_report['action_split_counts']['train']}",
        f"- Optimizer updates: {training['global_update_steps']}",
        f"- Baseline -> best validation loss: {training['baseline_validation']['pseudo_action_loss']:.4f} -> {training['best_checkpoint']['validation_pseudo_action_loss']:.4f}",
        "",
        "## Locked 500-parent free-running test: 5K -> full",
        "",
        f"- END completion: {control_free['end_completion_rate']:.4f} -> {full_free['end_completion_rate']:.4f}",
        f"- Sequence similarity: {control_free['mean_pseudo_gloss_sequence_similarity']:.4f} -> {full_free['mean_pseudo_gloss_sequence_similarity']:.4f}",
        f"- Token F1: {control_free['mean_pseudo_gloss_token_f1']:.4f} -> {full_free['mean_pseudo_gloss_token_f1']:.4f}",
        f"- Invalid-action ratio: {control_free['mean_invalid_action_ratio']:.4f} -> {full_free['mean_invalid_action_ratio']:.4f}",
        f"- Sequence paired bootstrap: {bootstrap['sequence_similarity']}",
        f"- Token-F1 paired bootstrap: {bootstrap['token_f1']}",
        "",
        "## Decision",
        "",
        f"- Gates: {gates}",
        f"- Recommendation: `{decision}`",
        "",
        "All quality targets and timing are pseudo labels, not gold ASL annotations.",
    ]
    (args.output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    professor = [
        "# Professor Update: Full-scale Fine-tune 2",
        "",
        f"- Scaled causal supervision from 71,882 to {data_report['action_split_counts']['train']} actions.",
        "- Restarted from the same Fine-tune 1 adapter used by the 5K control.",
        "- Evaluated both models on the exact same locked 500-parent test set.",
        f"- Sequence similarity: {control_free['mean_pseudo_gloss_sequence_similarity']:.4f} -> {full_free['mean_pseudo_gloss_sequence_similarity']:.4f}.",
        f"- Token F1: {control_free['mean_pseudo_gloss_token_f1']:.4f} -> {full_free['mean_pseudo_gloss_token_f1']:.4f}.",
        f"- Decision: {decision}.",
        "- Metrics use pseudo gloss and pseudo timing, not gold ASL labels.",
    ]
    (args.output_dir / "PROFESSOR_UPDATE.md").write_text(
        "\n".join(professor) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
