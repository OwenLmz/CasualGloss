#!/usr/bin/env python3
"""Paired locked-test comparison for any number of streaming policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


AUDIT_SCRIPTS = Path(__file__).resolve().parents[2] / "causal_alignment_latency_audit_v2_2" / "scripts"
sys.path.insert(0, str(AUDIT_SCRIPTS))

from audit_latency import rollout_rows  # noqa: E402


METRICS = {
    "sequence_similarity": {"label": "Sequence Similarity", "direction": "higher"},
    "token_f1": {"label": "Token F1", "direction": "higher"},
    "exact_match": {"label": "Exact Match", "direction": "higher"},
    "reference_length_al": {"label": "Reference-length AL", "direction": "lower"},
    "laal": {"label": "LAAL", "direction": "lower"},
    "completed_after_full_source": {"label": "Valid Completion Rate", "direction": "higher"},
    "invalid_ratio": {"label": "Invalid Output Ratio", "direction": "lower"},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap(deltas: list[float], repetitions: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    count = len(deltas)
    estimates = [mean(deltas[rng.randrange(count)] for _ in range(count)) for _ in range(repetitions)]
    lower = percentile(estimates, 0.025)
    upper = percentile(estimates, 0.975)
    return {
        "paired_count": count,
        "mean_delta": round(mean(deltas), 8),
        "ci95_lower": round(lower, 8),
        "ci95_upper": round(upper, 8),
        "significant_at_0_05": bool(lower > 0 or upper < 0),
    }


def parse_system(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=PATH for --system")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("System name cannot be empty")
    return name, Path(path)


def as_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    number = float(value)
    return number if math.isfinite(number) else None


def validate_inputs(raw: dict[str, dict[str, dict[str, Any]]], expected_count: int) -> list[str]:
    if len(raw) < 2:
        raise ValueError("At least two systems are required")
    for name, rows in raw.items():
        if len(rows) != expected_count:
            raise ValueError(f"{name} has {len(rows)} parents; expected {expected_count}")
    id_sets = [set(rows) for rows in raw.values()]
    if not all(ids == id_sets[0] for ids in id_sets[1:]):
        raise ValueError("Systems do not contain identical parent IDs")
    fields = ["source_text_en", "source_tokens", "target_pseudo_asl_gloss", "target_pseudo_asl_gloss_tokens"]
    names = list(raw)
    anchor_name = names[0]
    for parent_id in sorted(id_sets[0]):
        anchor = raw[anchor_name][parent_id]
        for name in names[1:]:
            for field in fields:
                if raw[name][parent_id].get(field) != anchor.get(field):
                    raise ValueError(f"{field} mismatch for {parent_id}: {anchor_name} versus {name}")
    return sorted(id_sets[0])


def conclusion(metric: str, result: dict[str, Any]) -> str:
    if not result["significant_at_0_05"]:
        return "no_significant_difference"
    improvement = result["mean_delta"] > 0
    if METRICS[metric]["direction"] == "lower":
        improvement = not improvement
    return "significant_improvement" if improvement else "significant_regression"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", action="append", type=parse_system, required=True)
    parser.add_argument("--baseline", default="v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=500)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    paths = dict(args.system)
    if len(paths) != len(args.system):
        raise ValueError("System names must be unique")
    if args.baseline not in paths:
        raise ValueError(f"Baseline {args.baseline!r} is not among {sorted(paths)}")

    raw: dict[str, dict[str, dict[str, Any]]] = {}
    normalized: dict[str, dict[str, dict[str, Any]]] = {}
    for name, path in paths.items():
        rows = read_jsonl(path)
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate IDs in {name}")
        raw[name] = {row["id"]: row for row in rows}
        metric_rows, _ = rollout_rows(path)
        normalized[name] = {row["id"]: row for row in metric_rows}

    ids = validate_inputs(raw, args.expected_count)
    report: dict[str, Any] = {
        "scientific_status": "paired_locked_test_pseudo_gloss_evaluation_not_gold_asl",
        "baseline": args.baseline,
        "common_parent_count": len(ids),
        "same_parent_ids": True,
        "same_source_and_reference": True,
        "bootstrap": {
            "method": "paired_parent_level_percentile_bootstrap",
            "repetitions": args.bootstrap_repetitions,
            "seed": args.seed,
        },
        "metric_definitions": METRICS,
        "input_files": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
        "systems": {},
        "paired_vs_baseline": {},
        "limitations": [
            "The reference is Qwen-generated pseudo ASL gloss, not expert gold ASL.",
            "The timing supervision is pseudo-aligned rather than video-verified gold timing.",
            "The source is incrementally revealed English text, not raw streaming speech.",
            "Confidence intervals quantify sampling uncertainty but do not remove pseudo-label bias.",
        ],
    }

    for name in paths:
        values: dict[str, list[float]] = {}
        for metric in METRICS:
            values[metric] = [
                number
                for parent_id in ids
                if (number := as_number(normalized[name][parent_id].get(metric))) is not None
            ]
        report["systems"][name] = {
            "parent_count": len(ids),
            "means": {metric: round(mean(metric_values), 8) for metric, metric_values in values.items()},
            "valid_counts": {metric: len(metric_values) for metric, metric_values in values.items()},
            "termination_reason_counts": dict(Counter(normalized[name][parent_id]["termination_reason"] for parent_id in ids)),
        }

    baseline = normalized[args.baseline]
    delta_rows: list[dict[str, Any]] = []
    for parent_id in ids:
        row: dict[str, Any] = {"id": parent_id}
        for name in paths:
            for metric in METRICS:
                row[f"{name}_{metric}"] = normalized[name][parent_id].get(metric)
        delta_rows.append(row)

    for system_index, (name, rows) in enumerate(paths.items()):
        if name == args.baseline:
            continue
        report["paired_vs_baseline"][name] = {}
        for metric_index, metric in enumerate(METRICS):
            deltas = []
            for parent_id in ids:
                left = as_number(baseline[parent_id].get(metric))
                right = as_number(normalized[name][parent_id].get(metric))
                if left is not None and right is not None:
                    deltas.append(right - left)
            result = paired_bootstrap(
                deltas,
                args.bootstrap_repetitions,
                args.seed + 1000 * system_index + metric_index,
            )
            result["direction"] = METRICS[metric]["direction"]
            result["conclusion"] = conclusion(metric, result)
            report["paired_vs_baseline"][name][metric] = result

    output_dir = args.output_dir
    write_json(output_dir / "reports/locked500_comparison.json", report)
    with (output_dir / "reports/locked500_summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["system", *METRICS])
        for name in paths:
            writer.writerow([name, *(report["systems"][name]["means"][metric] for metric in METRICS)])
    with (output_dir / "reports/parent_metrics.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(delta_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(delta_rows)
    write_json(
        output_dir / "reports/reproducibility_manifest.json",
        {
            "script_sha256": sha256(Path(__file__)),
            "input_files": report["input_files"],
            "bootstrap": report["bootstrap"],
            "common_parent_count": len(ids),
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
