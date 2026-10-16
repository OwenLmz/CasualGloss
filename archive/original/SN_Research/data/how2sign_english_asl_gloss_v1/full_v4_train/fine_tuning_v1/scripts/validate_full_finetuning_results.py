#!/usr/bin/env python3
"""Validate the completed full LoRA experiment and write a release report."""

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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    run = args.root / "runs" / "qwen2_5_7b_lora_how2sign_full_v1"
    paths = {
        "config": args.root / "configs" / "qwen2_5_7b_lora_full_v1.json",
        "split_report": args.root / "reports" / "split_report.json",
        "train": args.root / "manifests" / "train.jsonl",
        "validation": args.root / "manifests" / "validation.jsonl",
        "test": args.root / "manifests" / "test.jsonl",
        "training_history": run / "training_history.jsonl",
        "run_metadata": run / "run_metadata.json",
        "training_summary": run / "training_summary.json",
        "adapter_best": run / "adapter_best" / "adapter_model.safetensors",
        "adapter_selection": run / "adapter_best" / "selection.json",
        "base_metrics": run / "evaluation_base" / "metrics_summary.json",
        "best_metrics": run / "evaluation_best" / "metrics_summary.json",
        "base_test_predictions": run / "evaluation_base" / "test_generation_predictions.jsonl",
        "best_test_predictions": run / "evaluation_best" / "test_generation_predictions.jsonl",
        "comparison": args.root / "results" / "full_finetuning_comparison.json",
        "curve_png": args.root / "results" / "full_finetuning_curves.png",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing required outputs: {missing}")

    split = read_json(paths["split_report"])
    metadata = read_json(paths["run_metadata"])
    training = read_json(paths["training_summary"])
    selection = read_json(paths["adapter_selection"])
    base = read_json(paths["base_metrics"])
    best = read_json(paths["best_metrics"])
    comparison = read_json(paths["comparison"])
    history = read_jsonl(paths["training_history"])
    base_predictions = read_jsonl(paths["base_test_predictions"])
    best_predictions = read_jsonl(paths["best_test_predictions"])

    counts = {
        "train": sum(1 for _ in paths["train"].open(encoding="utf-8")),
        "validation": sum(
            1 for _ in paths["validation"].open(encoding="utf-8")
        ),
        "test": sum(1 for _ in paths["test"].open(encoding="utf-8")),
        "base_test_generation": len(base_predictions),
        "best_test_generation": len(best_predictions),
    }
    validation_events = [row for row in history if row.get("event") == "validation"]
    final_validation_step = max(int(row["global_step"]) for row in validation_events)
    no_test_metrics_during_training = all("test" not in row for row in history)
    base_ids = {row["id"] for row in base_predictions}
    best_ids = {row["id"] for row in best_predictions}
    finite_metrics = all(
        math.isfinite(float(value))
        for value in (
            base["teacher_forced"]["test"]["pseudo_label_loss"],
            best["teacher_forced"]["test"]["pseudo_label_loss"],
            base["generation"]["test"]["pseudo_label_generation_token_sequence_similarity"],
            best["generation"]["test"]["pseudo_label_generation_token_sequence_similarity"],
        )
    )
    checks = {
        "split_pipeline_ok": bool(split["pipeline_ok"]),
        "request_splits_disjoint": bool(split["request_splits_are_disjoint"]),
        "source_video_splits_disjoint": bool(split["source_video_splits_are_disjoint"]),
        "manifest_counts_match_split_report": counts["train"]
        == split["counts"]["train"]
        and counts["validation"] == split["counts"]["validation"]
        and counts["test"] == split["counts"]["test"],
        "training_reached_configured_steps": training["global_update_steps"]
        == metadata["total_updates"],
        "history_reached_final_step": final_validation_step
        == training["global_update_steps"],
        "best_checkpoint_selection_matches": selection["global_step"]
        == training["best_checkpoint"]["global_step"],
        "test_not_used_during_training": no_test_metrics_during_training,
        "base_and_best_test_prediction_ids_match": base_ids == best_ids,
        "test_generation_counts_are_500": counts["base_test_generation"] == 500
        and counts["best_test_generation"] == 500,
        "no_train_or_validation_truncation": metadata["train_dataset_tokenization"][
            "truncated_example_count"
        ]
        == 0
        and metadata["validation_dataset_tokenization"][
            "truncated_example_count"
        ]
        == 0,
        "reported_metrics_are_finite": finite_metrics,
        "comparison_is_marked_non_gold": comparison["scientific_status"]
        == "pseudo_label_comparison_not_gold_asl",
    }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "engineering_validated_pseudo_label_experiment_not_gold_asl",
        "counts": counts,
        "checks": checks,
        "pipeline_ok": all(checks.values()),
        "best_checkpoint": training["best_checkpoint"],
        "test_metrics": comparison["metrics"]["test"],
        "paired_bootstrap_test_generation": comparison[
            "paired_bootstrap_test_generation"
        ],
        "core_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "important_limitation": (
            "Engineering validation and pseudo-label improvements do not establish "
            "ASL linguistic correctness or sign-video alignment."
        ),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pipeline_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
