#!/usr/bin/env python3
"""Validate full-scale Fine-tune 2 artifacts and hash reproducibility files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--causal-root", type=Path, required=True)
    parser.add_argument("--seed1-full-root", type=Path, required=True)
    parser.add_argument("--seed2-5k-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--hash-report", type=Path, required=True)
    args = parser.parse_args()

    config = read_json(args.config)
    run = args.root / "runs" / config["experiment_name"]
    seed1_run = (
        args.seed1_full_root
        / "runs"
        / "qwen2_5_7b_lora_causal_action_full_corrected_v1"
    )
    seed2_5k_run = (
        args.seed2_5k_root
        / "runs"
        / "qwen2_5_7b_lora_causal_action_5k_seed2_v1"
    )
    scale_results = args.root / "results" / "controlled_full_seed2"
    stability_results = args.root / "results" / "seed_stability_full"

    training = read_json(run / "training_summary.json")
    data_preflight = read_json(args.root / "reports" / "full_experiment_preflight.json")
    seed_preflight = read_json(args.root / "reports" / "full_seed2_preflight.json")
    data_validation = read_json(args.causal_root / "reports" / "validation_report.json")
    action_metrics = read_json(run / "evaluation_full_seed2" / "metrics_summary.json")
    rollout_metrics = read_json(run / "rollout_full_seed2" / "rollout_metrics.json")
    action_predictions = read_jsonl(
        run / "evaluation_full_seed2" / "test_action_predictions.jsonl"
    )
    seed2_full_rollouts = read_jsonl(
        run / "rollout_full_seed2" / "sentence_rollouts.jsonl"
    )
    seed1_full_rollouts = read_jsonl(
        seed1_run / "rollout_full" / "sentence_rollouts.jsonl"
    )
    seed2_5k_rollouts = read_jsonl(
        seed2_5k_run / "rollout_seed2_5k" / "sentence_rollouts.jsonl"
    )
    scale_comparison = read_json(scale_results / "metrics_comparison.json")
    stability_comparison = read_json(stability_results / "metrics_comparison.json")

    seed2_full_ids = {row["id"] for row in seed2_full_rollouts}
    seed1_full_ids = {row["id"] for row in seed1_full_rollouts}
    seed2_5k_ids = {row["id"] for row in seed2_5k_rollouts}
    checks = {
        "full_data_validation_ok": bool(data_validation.get("ok")),
        "locked_data_preflight_ok": bool(data_preflight.get("ok")),
        "seed_replication_preflight_ok": bool(seed_preflight.get("ok")),
        "training_seed_exact": config.get("seed") == 20260721,
        "same_locked_500_parent_ids_across_all_controls": (
            seed2_full_ids == seed1_full_ids == seed2_5k_ids
            and len(seed2_full_ids) == 500
        ),
        "training_counts_exact": (
            training["train_action_count"] == 397700
            and training["validation_action_count"] == 9209
            and training["test_action_count_reserved"] == 8365
        ),
        "global_steps_exact": training["global_update_steps"] == 12429,
        "best_checkpoint_after_training_started": training["best_checkpoint"]["global_step"] > 0,
        "best_validation_loss_finite": math.isfinite(
            float(training["best_checkpoint"]["validation_pseudo_action_loss"])
        ),
        "adapter_best_exists": (run / "adapter_best" / "adapter_model.safetensors").is_file(),
        "action_test_count_exact": action_metrics["record_count"] == 8365,
        "action_prediction_count_exact": len(action_predictions) == 8365,
        "rollout_count_exact": len(seed2_full_rollouts) == 500,
        "rollout_metrics_count_exact": rollout_metrics["parent_count"] == 500,
        "all_action_predictions_non_gold": all(
            row.get("target_is_gold") is False and row.get("alignment_is_gold") is False
            for row in action_predictions
        ),
        "all_rollouts_non_gold": all(
            row.get("target_is_gold") is False
            and row.get("alignment_is_gold") is False
            and row.get("rollout_history_uses_gold_committed_gloss") is False
            for row in seed2_full_rollouts
        ),
        "scale_comparison_has_decision": bool(scale_comparison.get("recommended_decision")),
        "stability_comparison_has_decision": bool(
            stability_comparison.get("recommended_decision")
        ),
        "scale_bootstrap_uses_500_parents": all(
            value["paired_parent_count"] == 500
            for value in scale_comparison["paired_bootstrap_full_minus_5k"].values()
        ),
        "stability_bootstrap_uses_500_parents": all(
            value["paired_parent_count"] == 500
            for value in stability_comparison["paired_bootstrap_seed2_minus_seed1"].values()
        ),
        "scale_curve_exists": (scale_results / "fine_tune2_full_results.png").is_file(),
        "stability_curve_exists": (stability_results / "full_seed_stability.png").is_file(),
    }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "validated_full_seed2_replication_pseudo_labels_not_gold_asl",
        "checks": checks,
        "training": {
            "seed": config["seed"],
            "train_action_count": training["train_action_count"],
            "validation_action_count": training["validation_action_count"],
            "test_action_count_reserved": training["test_action_count_reserved"],
            "global_update_steps": training["global_update_steps"],
            "best_checkpoint": training["best_checkpoint"],
        },
        "scale_decision": scale_comparison["recommended_decision"],
        "seed_stability_decision": stability_comparison["recommended_decision"],
        "qualified_asl_review_complete": False,
        "ok": all(checks.values()),
    }
    write_json(args.output_report, report)

    core_paths = [
        args.config,
        args.root / "scripts" / "causal_action_common.py",
        args.root / "scripts" / "audit_full_experiment_preflight.py",
        args.root / "scripts" / "audit_full_seed2_preflight.py",
        args.root / "scripts" / "train_qwen_lora_causal_ddp.py",
        args.root / "scripts" / "evaluate_causal_action_model.py",
        args.root / "scripts" / "evaluate_free_running_rollout.py",
        args.root / "scripts" / "summarize_full_experiment.py",
        args.root / "scripts" / "summarize_full_seed_stability.py",
        args.root / "scripts" / "validate_full_seed2_experiment.py",
        args.root / "scripts" / "run_finetune2_full_seed2.sh",
        args.root / "reports" / "full_experiment_preflight.json",
        args.root / "reports" / "full_seed2_preflight.json",
        run / "run_metadata.json",
        run / "training_history.jsonl",
        run / "training_summary.json",
        run / "adapter_best" / "adapter_config.json",
        run / "adapter_best" / "adapter_model.safetensors",
        run / "evaluation_full_seed2" / "metrics_summary.json",
        run / "rollout_full_seed2" / "rollout_metrics.json",
        scale_results / "metrics_comparison.json",
        scale_results / "fine_tune2_full_results.png",
        scale_results / "RESULTS.md",
        scale_results / "PROFESSOR_UPDATE.md",
        stability_results / "metrics_comparison.json",
        stability_results / "full_seed_stability.png",
        stability_results / "RESULTS.md",
        stability_results / "PROFESSOR_UPDATE.md",
        args.output_report,
    ]
    missing = [str(path) for path in core_paths if not path.is_file()]
    hashes = {
        str(path.relative_to(args.root)).replace("\\", "/"): sha256_file(path)
        for path in core_paths
        if path.is_file()
    }
    hash_report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "SHA-256",
        "file_count": len(hashes),
        "missing_core_files": missing,
        "files": hashes,
        "ok": not missing,
    }
    write_json(args.hash_report, hash_report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(hash_report, indent=2, ensure_ascii=False))
    if not report["ok"] or not hash_report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
