#!/usr/bin/env python3
"""Compare multiple locked-test rollouts with paired bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

from audit_latency import rollout_rows, write_json


LOWER_IS_BETTER = {
    "project_normalized_al",
    "reference_length_al",
    "laal",
    "ap",
    "dal",
    "wait_ratio",
    "full_source_write_ratio",
    "invalid_ratio",
    "premature_end",
    "maximum_actions_reached",
    "prediction_reference_length_absolute_error",
}
METRICS = [
    "project_normalized_al",
    "reference_length_al",
    "laal",
    "ap",
    "dal",
    "sequence_similarity",
    "token_f1",
    "exact_match",
    "completed_with_end",
    "completed_after_full_source",
    "premature_end",
    "maximum_actions_reached",
    "prediction_reference_length_absolute_error",
    "wait_ratio",
    "invalid_ratio",
    "full_source_write_ratio",
]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * q
    lower = int(location)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap(deltas: list[float], samples: int, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    count = len(deltas)
    estimates = [mean(deltas[rng.randrange(count)] for _ in range(count)) for _ in range(samples)]
    return {
        "mean_delta": round(mean(deltas), 8),
        "ci95_lower": round(percentile(estimates, 0.025), 8),
        "ci95_upper": round(percentile(estimates, 0.975), 8),
    }


def parse_rollout(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=PATH for --rollout")
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", action="append", type=parse_rollout, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-paired-tsv", type=Path, required=True)
    args = parser.parse_args()

    versions: dict[str, dict[str, dict[str, Any]]] = {}
    for name, path in args.rollout:
        rows, _ = rollout_rows(path)
        versions[name] = {row["id"]: row for row in rows}
    if args.baseline not in versions:
        raise ValueError(f"Unknown baseline: {args.baseline}")

    common_ids = sorted(set.intersection(*(set(rows) for rows in versions.values())))
    paired_rows: list[dict[str, Any]] = []
    for parent_id in common_ids:
        output: dict[str, Any] = {"id": parent_id}
        for name, rows in versions.items():
            for metric in METRICS:
                output[f"{name}_{metric}"] = rows[parent_id].get(metric)
        paired_rows.append(output)

    report: dict[str, Any] = {
        "scientific_status": "paired_locked_pseudo_gloss_evaluation_not_gold_asl",
        "baseline": args.baseline,
        "common_parent_count": len(common_ids),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "versions": {},
        "paired_comparisons": {},
        "metric_direction": {
            metric: "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better"
            for metric in METRICS
        },
    }
    for name, rows in versions.items():
        selected = [rows[parent_id] for parent_id in common_ids]
        report["versions"][name] = {
            "input_parent_count": len(rows),
            "paired_parent_count": len(selected),
            "means": {
                metric: round(mean(float(row[metric]) for row in selected if row.get(metric) is not None), 8)
                for metric in METRICS
            },
            "termination_reason_counts": {
                reason: sum(row["termination_reason"] == reason for row in selected)
                for reason in sorted({row["termination_reason"] for row in selected})
            },
        }

    baseline_rows = versions[args.baseline]
    for comparison_index, (name, rows) in enumerate(versions.items()):
        if name == args.baseline:
            continue
        metric_results: dict[str, Any] = {}
        for metric_index, metric in enumerate(METRICS):
            deltas = [
                float(rows[parent_id][metric]) - float(baseline_rows[parent_id][metric])
                for parent_id in common_ids
                if rows[parent_id].get(metric) is not None
                and baseline_rows[parent_id].get(metric) is not None
            ]
            interval = paired_bootstrap(
                deltas,
                args.bootstrap_samples,
                args.seed + comparison_index * 100 + metric_index,
            )
            interval["difference"] = f"{name}_minus_{args.baseline}"
            interval["statistically_clear_improvement"] = (
                interval["ci95_upper"] < 0
                if metric in LOWER_IS_BETTER
                else interval["ci95_lower"] > 0
            )
            metric_results[metric] = interval
        report["paired_comparisons"][f"{name}_vs_{args.baseline}"] = metric_results

    write_json(args.output_report, report)
    args.output_paired_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_paired_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
