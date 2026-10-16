#!/usr/bin/env python3
"""Measure timing-policy imitation error against v2.2 pseudo alignment labels."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

from audit_latency import latency_record, read_jsonl, write_json


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * q
    lower = int(location)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap(values: list[float], samples: int, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    count = len(values)
    draws = [mean(values[rng.randrange(count)] for _ in range(count)) for _ in range(samples)]
    return {
        "mean": round(mean(values), 8),
        "ci95_lower": round(percentile(draws, 0.025), 8),
        "ci95_upper": round(percentile(draws, 0.975), 8),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-pairs", type=Path, required=True)
    parser.add_argument("--model-rollout", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    args = parser.parse_args()

    labels = {
        row["id"]: row
        for row in read_jsonl(args.label_pairs)
        if row.get("split") == args.split
    }
    rollouts = {row["id"]: row for row in read_jsonl(args.model_rollout)}
    common_ids = sorted(labels.keys() & rollouts.keys())
    rows: list[dict[str, Any]] = []
    for parent_id in common_ids:
        label = labels[parent_id]
        rollout = rollouts[parent_id]
        expected = [int(value) for value in label["semantic_v2_2_emit_source_positions"]]
        predicted = [int(value) for value in rollout.get("write_source_positions", [])]
        source_count = len(label["source_tokens"])
        target_count = len(label["target_pseudo_asl_gloss_tokens"])
        expected_latency = latency_record(expected, source_count, target_count)
        predicted_latency = latency_record(predicted, source_count, target_count)
        paired_count = min(len(expected), len(predicted))
        position_errors = [
            predicted[index] - expected[index] for index in range(paired_count)
        ]
        rows.append(
            {
                "id": parent_id,
                "source_token_count": source_count,
                "target_token_count": target_count,
                "prediction_token_count": len(predicted),
                "length_difference": len(predicted) - target_count,
                "schedule_exact": predicted == expected,
                "paired_position_count": paired_count,
                "mean_signed_emit_position_error": (
                    mean(position_errors) if position_errors else None
                ),
                "mean_absolute_emit_position_error": (
                    mean(abs(value) for value in position_errors)
                    if position_errors
                    else None
                ),
                "first_write_position_error": (
                    predicted[0] - expected[0] if expected and predicted else None
                ),
                "label_laal": expected_latency["laal"],
                "model_laal": predicted_latency["laal"],
                "model_minus_label_laal": (
                    predicted_latency["laal"] - expected_latency["laal"]
                    if expected_latency["laal"] is not None
                    and predicted_latency["laal"] is not None
                    else None
                ),
                "label_full_source_write_ratio": expected_latency[
                    "full_source_write_ratio"
                ],
                "model_full_source_write_ratio": predicted_latency[
                    "full_source_write_ratio"
                ],
                "completed_after_full_source": bool(
                    rollout.get("completed_after_full_source")
                    if rollout.get("completed_after_full_source") is not None
                    else rollout.get("termination_reason") == "end_after_full_source"
                ),
                "pseudo_gloss_sequence_similarity": float(
                    rollout["pseudo_gloss_sequence_similarity"]
                ),
            }
        )

    available_mae = [
        float(row["mean_absolute_emit_position_error"])
        for row in rows
        if row["mean_absolute_emit_position_error"] is not None
    ]
    available_signed = [
        float(row["mean_signed_emit_position_error"])
        for row in rows
        if row["mean_signed_emit_position_error"] is not None
    ]
    available_laal = [
        float(row["model_minus_label_laal"])
        for row in rows
        if row["model_minus_label_laal"] is not None
    ]
    report = {
        "scientific_status": "model_imitation_of_pseudo_timing_not_gold_alignment",
        "model_name": args.model_name,
        "split": args.split,
        "label_parent_count": len(labels),
        "model_parent_count": len(rollouts),
        "common_parent_count": len(rows),
        "schedule_exact_ratio": round(
            sum(row["schedule_exact"] for row in rows) / max(1, len(rows)), 8
        ),
        "valid_completion_ratio": round(
            sum(row["completed_after_full_source"] for row in rows)
            / max(1, len(rows)),
            8,
        ),
        "mean_length_difference": round(
            mean(float(row["length_difference"]) for row in rows), 8
        ),
        "mean_absolute_emit_position_error": bootstrap(
            available_mae, args.bootstrap_samples, args.seed
        ),
        "mean_signed_emit_position_error": bootstrap(
            available_signed, args.bootstrap_samples, args.seed + 1
        ),
        "model_minus_label_laal": bootstrap(
            available_laal, args.bootstrap_samples, args.seed + 2
        ),
        "interpretation": {
            "positive_signed_position_error": "model writes later than the v2.2 pseudo timing label",
            "negative_signed_position_error": "model writes earlier than the v2.2 pseudo timing label",
            "positive_laal_gap": "model rollout adds latency beyond the label policy",
        },
    }
    write_json(args.output_report, report)
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        rows,
        key=lambda row: (
            abs(float(row["model_minus_label_laal"]))
            if row["model_minus_label_laal"] is not None
            else -1
        ),
        reverse=True,
    )
    with args.output_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranked[0]))
        writer.writeheader()
        writer.writerows(ranked)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
