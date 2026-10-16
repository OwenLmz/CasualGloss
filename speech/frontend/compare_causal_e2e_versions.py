#!/usr/bin/env python3
"""Paired comparison of causal policies on the same speech-frontend records."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any


METRICS = {
    "pseudo_gloss_sequence_similarity": 1.0,
    "pseudo_gloss_token_f1": 1.0,
    "pseudo_gloss_exact_match": 1.0,
    "speech_to_gloss_completed_with_end": 1.0,
    "speech_to_gloss_completed_after_full_source": 1.0,
    "speech_to_gloss_premature_end": -1.0,
    "prediction_reference_length_absolute_error": -1.0,
    "speech_to_gloss_first_write_audio_seconds": -1.0,
    "speech_to_gloss_reference_length_al_seconds": -1.0,
    "speech_to_gloss_laal_seconds": -1.0,
    "speech_to_gloss_average_proportion": -1.0,
    "speech_to_gloss_dal_seconds": -1.0,
    "speech_to_gloss_compute_adjusted_laal_seconds": -1.0,
    "speech_to_gloss_processing_realtime_factor": -1.0,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def as_number(row: dict[str, Any], metric: str) -> float | None:
    if metric == "prediction_reference_length_absolute_error":
        reference = len(str(row.get("reference_pseudo_asl_gloss", "")).split())
        prediction = len(row.get("speech_to_gloss_predicted_tokens", []))
        return abs(prediction / max(1, reference) - 1.0)
    value = row.get(metric)
    if value is None:
        return None
    return float(value)


def bootstrap_delta(
    deltas: list[float], samples: int, rng: random.Random
) -> dict[str, float | int]:
    observed = mean(deltas)
    draws = [
        mean(deltas[rng.randrange(len(deltas))] for _ in deltas)
        for _ in range(samples)
    ]
    return {
        "count": len(deltas),
        "mean_candidate_minus_baseline": round(observed, 8),
        "ci95_lower": round(percentile(draws, 0.025), 8),
        "ci95_upper": round(percentile(draws, 0.975), 8),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-name", default="v1")
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--quality-noninferiority-margin", type=float, default=0.02)
    parser.add_argument("--completion-noninferiority-margin", type=float, default=0.01)
    parser.add_argument("--length-error-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--scientific-status",
        default="paired_synthetic_speech_to_pseudo_gloss_not_gold_asl",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = {row["id"]: row for row in read_jsonl(args.baseline)}
    candidate = {row["id"]: row for row in read_jsonl(args.candidate)}
    common_ids = sorted(baseline.keys() & candidate.keys())
    rng = random.Random(args.seed)
    comparisons: dict[str, Any] = {}
    means: dict[str, Any] = {}
    for metric in METRICS:
        paired: list[tuple[float, float]] = []
        for parent_id in common_ids:
            left = as_number(baseline[parent_id], metric)
            right = as_number(candidate[parent_id], metric)
            if left is not None and right is not None:
                paired.append((left, right))
        if not paired:
            continue
        comparisons[metric] = bootstrap_delta(
            [right - left for left, right in paired], args.bootstrap_samples, rng
        )
        means[metric] = {
            args.baseline_name: round(mean(left for left, _ in paired), 8),
            args.candidate_name: round(mean(right for _, right in paired), 8),
            "higher_is_better": METRICS[metric] > 0,
        }

    laal = comparisons.get("speech_to_gloss_laal_seconds")
    compute_adjusted_laal = comparisons.get(
        "speech_to_gloss_compute_adjusted_laal_seconds"
    )
    sequence = comparisons["pseudo_gloss_sequence_similarity"]
    token_f1 = comparisons["pseudo_gloss_token_f1"]
    completion = comparisons["speech_to_gloss_completed_after_full_source"]
    length_error = means["prediction_reference_length_absolute_error"]
    checks = {
        "laal_improved": laal is not None and laal["ci95_upper"] < 0,
        "compute_adjusted_laal_improved": (
            compute_adjusted_laal is not None
            and compute_adjusted_laal["ci95_upper"] < 0
        ),
        "sequence_similarity_noninferior": (
            sequence["ci95_lower"] > -args.quality_noninferiority_margin
        ),
        "token_f1_noninferior": (
            token_f1["ci95_lower"] > -args.quality_noninferiority_margin
        ),
        "completion_noninferior": (
            completion["ci95_lower"] > -args.completion_noninferiority_margin
        ),
        "length_error_within_tolerance": (
            length_error[args.candidate_name]
            <= length_error[args.baseline_name] + args.length_error_tolerance
        ),
    }
    output = {
        "scientific_status": args.scientific_status,
        "baseline": args.baseline_name,
        "candidate": args.candidate_name,
        "common_record_count": len(common_ids),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "means": means,
        "paired_deltas": comparisons,
        "predeclared_checks": checks,
        "completion_metric": "speech_to_gloss_completed_after_full_source",
        "accepted": all(checks.values()),
        "important_scope": [
            "The Mandarin speech is synthetic.",
            "Content metrics use weak pseudo-ASL-gloss references, not gold ASL.",
            "Audio latency excludes model computation time.",
            "Compute-adjusted latency is a serial execution upper bound measured on this server.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
