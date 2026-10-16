#!/usr/bin/env python3
"""Select a checkpoint on validation rollouts before locked-test evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from audit_latency import rollout_rows, write_json


def parse_rollout(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=PATH for --rollout")
    name, path = value.split("=", 1)
    return name, Path(path)


def average(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", action="append", type=parse_rollout, required=True)
    parser.add_argument("--quality-margin", type=float, default=0.02)
    parser.add_argument("--length-error-margin", type=float, default=0.02)
    parser.add_argument("--minimum-completion-rate", type=float, default=0.99)
    parser.add_argument("--invalid-ratio-margin", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    versions = {}
    for name, path in args.rollout:
        rows, _ = rollout_rows(path)
        versions[name] = {
            "path": str(path),
            "parent_count": len(rows),
            "mean_laal": average(rows, "laal"),
            "mean_sequence_similarity": average(rows, "sequence_similarity"),
            "mean_token_f1": average(rows, "token_f1"),
            "mean_prediction_length_absolute_error": average(
                rows, "prediction_reference_length_absolute_error"
            ),
            "completion_rate": average(rows, "completed_after_full_source"),
            "mean_invalid_ratio": average(rows, "invalid_ratio"),
        }

    parent_counts = {row["parent_count"] for row in versions.values()}
    if len(parent_counts) != 1:
        raise RuntimeError(f"Validation rollout size mismatch: {parent_counts}")
    best_sequence = max(row["mean_sequence_similarity"] for row in versions.values())
    best_f1 = max(row["mean_token_f1"] for row in versions.values())
    best_length_error = min(
        row["mean_prediction_length_absolute_error"] for row in versions.values()
    )
    best_invalid_ratio = min(row["mean_invalid_ratio"] for row in versions.values())
    eligible = []
    for name, row in versions.items():
        checks = {
            "completion": row["completion_rate"] >= args.minimum_completion_rate,
            "sequence_similarity": (
                row["mean_sequence_similarity"] >= best_sequence - args.quality_margin
            ),
            "token_f1": row["mean_token_f1"] >= best_f1 - args.quality_margin,
            "prediction_length_error": (
                row["mean_prediction_length_absolute_error"]
                <= best_length_error + args.length_error_margin
            ),
            "invalid_ratio": (
                row["mean_invalid_ratio"]
                <= best_invalid_ratio + args.invalid_ratio_margin
            ),
        }
        row["eligibility_checks"] = checks
        row["eligible"] = all(checks.values())
        if row["eligible"]:
            eligible.append(name)

    fallback_used = not eligible
    if eligible:
        selected = min(eligible, key=lambda name: versions[name]["mean_laal"])
        fallback_reason = None
    else:
        selected = min(
            versions,
            key=lambda name: (
                -versions[name]["completion_rate"],
                -versions[name]["mean_sequence_similarity"],
                -versions[name]["mean_token_f1"],
                versions[name]["mean_prediction_length_absolute_error"],
                versions[name]["mean_invalid_ratio"],
                versions[name]["mean_laal"],
            ),
        )
        fallback_reason = (
            "No checkpoint satisfied all eligibility thresholds; selected by "
            "completion, sequence similarity, token F1, length error, then LAAL."
        )
    output = {
        "scientific_status": "validation_only_checkpoint_selection_over_pseudo_gloss",
        "criteria": {
            "quality_margin": args.quality_margin,
            "length_error_margin": args.length_error_margin,
            "minimum_completion_rate": args.minimum_completion_rate,
            "completion_metric": "completed_after_full_source",
            "invalid_ratio_margin": args.invalid_ratio_margin,
            "selection": "lowest mean LAAL among eligible validation checkpoints",
        },
        "versions": versions,
        "selected_candidate": selected,
        "selection_fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "test_set_used_for_selection": False,
        "important_note": "Validation content metrics use pseudo gloss references, not gold ASL.",
    }
    write_json(args.output, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
