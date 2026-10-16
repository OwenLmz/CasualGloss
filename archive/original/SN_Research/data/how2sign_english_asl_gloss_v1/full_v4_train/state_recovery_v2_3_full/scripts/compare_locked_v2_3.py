#!/usr/bin/env python3
"""Paired locked-500 comparison for the frozen v2.3 checkpoint."""

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
    "sequence_similarity": {"direction": "higher"},
    "token_f1": {"direction": "higher"},
    "reference_length_al": {"direction": "lower"},
    "laal": {"direction": "lower"},
    "completed_after_full_source": {"direction": "higher"},
    "invalid_ratio": {"direction": "lower"},
}
EXPECTED_SYSTEMS = ("v1", "v2_1", "v2_2", "v2_3")


def parse_version(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=PATH for --version")
    name, path = value.split("=", 1)
    return name, Path(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap(deltas: list[float], repetitions: int, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    count = len(deltas)
    estimates = [mean(deltas[rng.randrange(count)] for _ in range(count)) for _ in range(repetitions)]
    return {
        "mean_delta": round(mean(deltas), 8),
        "ci95_lower": round(percentile(estimates, 0.025), 8),
        "ci95_upper": round(percentile(estimates, 0.975), 8),
    }


def conclusion(result: dict[str, float], direction: str) -> str:
    if direction == "higher":
        if result["ci95_lower"] > 0:
            return "statistically_clear_improvement"
        if result["ci95_upper"] < 0:
            return "statistically_clear_regression"
    else:
        if result["ci95_upper"] < 0:
            return "statistically_clear_improvement"
        if result["ci95_lower"] > 0:
            return "statistically_clear_regression"
    return "no_statistically_clear_difference"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="append", type=parse_version, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    paths = dict(args.version)
    if tuple(sorted(paths)) != tuple(sorted(EXPECTED_SYSTEMS)):
        raise ValueError(f"Expected {EXPECTED_SYSTEMS}, found {sorted(paths)}")

    raw: dict[str, dict[str, dict[str, Any]]] = {}
    normalized: dict[str, dict[str, dict[str, Any]]] = {}
    for name in EXPECTED_SYSTEMS:
        rows = read_jsonl(paths[name])
        if len(rows) != 500 or len({row["id"] for row in rows}) != 500:
            raise ValueError(f"{name} must contain exactly 500 unique parents")
        raw[name] = {row["id"]: row for row in rows}
        audited, _ = rollout_rows(paths[name])
        normalized[name] = {row["id"]: row for row in audited}

    ids = sorted(raw["v1"])
    if any(set(raw[name]) != set(ids) for name in EXPECTED_SYSTEMS[1:]):
        raise ValueError("Locked rollout parent IDs differ")
    reference_fields = (
        "source_text_en",
        "source_tokens",
        "target_pseudo_asl_gloss",
        "target_pseudo_asl_gloss_tokens",
    )
    for parent_id in ids:
        reference = raw["v1"][parent_id]
        for name in EXPECTED_SYSTEMS[1:]:
            if any(reference.get(field) != raw[name][parent_id].get(field) for field in reference_fields):
                raise ValueError(f"Source/reference mismatch for {parent_id}: v1 versus {name}")

    report: dict[str, Any] = {
        "scientific_status": "one_shot_locked500_pseudo_gloss_evaluation_not_gold_asl",
        "selection_used_locked_test": False,
        "common_parent_count": len(ids),
        "same_parent_ids": True,
        "same_source_and_reference": True,
        "bootstrap": {
            "method": "paired_parent_level_percentile_bootstrap",
            "repetitions": args.bootstrap_repetitions,
            "seed": args.seed,
        },
        "input_files": {name: {"path": str(paths[name]), "sha256": sha256(paths[name])} for name in EXPECTED_SYSTEMS},
        "versions": {},
        "paired_comparisons": {},
        "limitations": [
            "The locked reference is Qwen-generated pseudo ASL rather than expert gold ASL.",
            "Causal timing is pseudo-aligned rather than video-verified gold timing.",
            "The rollout begins from English token prefixes rather than streaming speech chunks.",
            "The locked test was used once after development and confirmation decisions were frozen.",
        ],
    }

    for name in EXPECTED_SYSTEMS:
        values = {
            metric: [
                value
                for parent_id in ids
                if (value := as_number(normalized[name][parent_id][metric])) is not None
            ]
            for metric in METRICS
        }
        report["versions"][name] = {
            "parent_count": len(ids),
            "means": {metric: round(mean(metric_values), 8) for metric, metric_values in values.items()},
            "valid_counts": {metric: len(metric_values) for metric, metric_values in values.items()},
            "termination_reason_counts": dict(Counter(normalized[name][parent_id]["termination_reason"] for parent_id in ids)),
        }

    for baseline_index, baseline in enumerate(("v1", "v2_1", "v2_2")):
        metric_results = {}
        for metric_index, (metric, metadata) in enumerate(METRICS.items()):
            pairs = []
            for parent_id in ids:
                left = as_number(normalized[baseline][parent_id][metric])
                right = as_number(normalized["v2_3"][parent_id][metric])
                if left is not None and right is not None:
                    pairs.append((left, right))
            deltas = [right - left for left, right in pairs]
            result = paired_bootstrap(
                deltas,
                args.bootstrap_repetitions,
                args.seed + baseline_index * 100 + metric_index,
            )
            result.update(
                {
                    "difference": f"v2_3_minus_{baseline}",
                    "direction": metadata["direction"],
                    "paired_count": len(deltas),
                    "excluded_missing_count": len(ids) - len(deltas),
                    "conclusion": conclusion(result, metadata["direction"]),
                }
            )
            metric_results[metric] = result
        report["paired_comparisons"][f"v2_3_vs_{baseline}"] = metric_results

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "locked500_v2_3_comparison.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (args.output_dir / "locked500_v2_3_summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["system", *METRICS]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for name in EXPECTED_SYSTEMS:
            writer.writerow({"system": name, **report["versions"][name]["means"]})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
