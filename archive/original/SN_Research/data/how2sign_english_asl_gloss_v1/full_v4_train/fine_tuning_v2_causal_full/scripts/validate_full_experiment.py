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
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--control-5k-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--hash-report", type=Path, required=True)
    args = parser.parse_args()

    config = read_json(args.config)
    run = args.root / "runs" / config["experiment_name"]
    training = read_json(run / "training_summary.json")
    preflight = read_json(args.root / "reports" / "full_experiment_preflight.json")
    action_metrics = read_json(run / "evaluation_full" / "metrics_summary.json")
    full_rollout_metrics = read_json(run / "rollout_full" / "rollout_metrics.json")
    full_rollouts = read_jsonl(run / "rollout_full" / "sentence_rollouts.jsonl")
    control_rollouts = read_jsonl(
        args.control_5k_root
        / "runs"
        / "qwen2_5_7b_lora_causal_action_5k_corrected_v1"
        / "rollout_finetune2_5k"
        / "sentence_rollouts.jsonl"
    )
    comparison = read_json(args.root / "results" / "controlled_full" / "metrics_comparison.json")
    data_validation = read_json(args.data_root / "reports" / "validation_report.json")

    full_ids = {row["id"] for row in full_rollouts}
    control_ids = {row["id"] for row in control_rollouts}
    action_predictions = read_jsonl(run / "evaluation_full" / "test_action_predictions.jsonl")
    checks = {
        "full_data_validation_ok": bool(data_validation.get("ok")),
        "preflight_ok": bool(preflight.get("ok")),
        "same_locked_rollout_parent_ids": full_ids == control_ids and len(full_ids) == 500,
        "training_counts_exact": (
            training["train_action_count"] == 397700
            and training["validation_action_count"] == 9209
            and training["test_action_count_reserved"] == 8365
        ),
        "best_checkpoint_after_training_started": training["best_checkpoint"]["global_step"] > 0,
        "best_validation_loss_finite": math.isfinite(
            float(training["best_checkpoint"]["validation_pseudo_action_loss"])
        ),
        "adapter_best_exists": (run / "adapter_best" / "adapter_model.safetensors").is_file(),
        "action_test_count_exact": action_metrics["record_count"] == 8365,
        "action_prediction_count_exact": len(action_predictions) == 8365,
        "rollout_count_exact": len(full_rollouts) == 500,
        "rollout_metrics_count_exact": full_rollout_metrics["parent_count"] == 500,
        "all_action_predictions_non_gold": all(
            row.get("target_is_gold") is False and row.get("alignment_is_gold") is False
            for row in action_predictions
        ),
        "all_rollouts_non_gold": all(
            row.get("target_is_gold") is False
            and row.get("alignment_is_gold") is False
            and row.get("rollout_history_uses_gold_committed_gloss") is False
            for row in full_rollouts
        ),
        "comparison_has_decision": bool(comparison.get("recommended_decision")),
        "paired_bootstrap_uses_500_parents": all(
            value["paired_parent_count"] == 500
            for value in comparison["paired_bootstrap_full_minus_5k"].values()
        ),
        "result_curve_exists": (
            args.root / "results" / "controlled_full" / "fine_tune2_full_results.png"
        ).is_file(),
    }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "validated_full_scale_pseudo_causal_action_experiment_not_gold_asl",
        "checks": checks,
        "training": {
            "train_action_count": training["train_action_count"],
            "validation_action_count": training["validation_action_count"],
            "test_action_count_reserved": training["test_action_count_reserved"],
            "global_update_steps": training["global_update_steps"],
            "best_checkpoint": training["best_checkpoint"],
        },
        "recommended_decision": comparison["recommended_decision"],
        "qualified_asl_review_complete": False,
        "ok": all(checks.values()),
    }
    write_json(args.output_report, report)

    core_paths = [
        args.config,
        args.root / "scripts" / "causal_action_common.py",
        args.root / "scripts" / "audit_full_experiment_preflight.py",
        args.root / "scripts" / "train_qwen_lora_causal_ddp.py",
        args.root / "scripts" / "evaluate_causal_action_model.py",
        args.root / "scripts" / "evaluate_free_running_rollout.py",
        args.root / "scripts" / "summarize_full_experiment.py",
        args.root / "scripts" / "validate_full_experiment.py",
        args.root / "scripts" / "run_finetune2_full.sh",
        args.root / "reports" / "full_experiment_preflight.json",
        run / "run_metadata.json",
        run / "training_history.jsonl",
        run / "training_summary.json",
        run / "adapter_best" / "adapter_config.json",
        run / "adapter_best" / "adapter_model.safetensors",
        run / "evaluation_full" / "metrics_summary.json",
        run / "rollout_full" / "rollout_metrics.json",
        args.root / "results" / "controlled_full" / "metrics_comparison.json",
        args.root / "results" / "controlled_full" / "fine_tune2_full_results.png",
        args.root / "results" / "controlled_full" / "RESULTS.md",
        args.root / "results" / "controlled_full" / "PROFESSOR_UPDATE.md",
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
