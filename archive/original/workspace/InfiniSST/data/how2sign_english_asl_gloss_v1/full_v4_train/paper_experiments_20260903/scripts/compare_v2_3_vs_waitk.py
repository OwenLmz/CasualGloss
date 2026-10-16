#!/usr/bin/env python3
"""Paired locked-500 comparison of v2.3 and the strongest modified Wait-k point."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any


AUDIT_SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "causal_alignment_latency_audit_v2_2"
    / "scripts"
)
sys.path.insert(0, str(AUDIT_SCRIPTS))

from audit_latency import rollout_rows  # noqa: E402


METRICS = {
    "sequence_similarity": "higher",
    "token_f1": "higher",
    "reference_length_al": "lower",
    "laal": "lower",
    "completed_after_full_source": "higher",
    "invalid_ratio": "lower",
}
REFERENCE_FIELDS = (
    "source_text_en",
    "source_tokens",
    "target_pseudo_asl_gloss",
    "target_pseudo_asl_gloss_tokens",
)


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


def paired_bootstrap(
    deltas: list[float], repetitions: int, seed: int
) -> dict[str, float]:
    rng = random.Random(seed)
    count = len(deltas)
    estimates = [
        mean(deltas[rng.randrange(count)] for _ in range(count))
        for _ in range(repetitions)
    ]
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
    parser.add_argument("--waitk-k1", type=Path, required=True)
    parser.add_argument("--v2-3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--quality-noninferiority-margin", type=float, default=0.03)
    args = parser.parse_args()

    raw = {
        "waitk_v1_k1": {row["id"]: row for row in read_jsonl(args.waitk_k1)},
        "v2_3": {row["id"]: row for row in read_jsonl(args.v2_3)},
    }
    for name, rows in raw.items():
        if len(rows) != 500:
            raise ValueError(f"{name} must contain exactly 500 unique rows, found {len(rows)}")
    ids = sorted(raw["waitk_v1_k1"])
    if set(ids) != set(raw["v2_3"]):
        raise ValueError("Locked-500 parent IDs differ")
    for parent_id in ids:
        baseline = raw["waitk_v1_k1"][parent_id]
        candidate = raw["v2_3"][parent_id]
        if any(baseline.get(field) != candidate.get(field) for field in REFERENCE_FIELDS):
            raise ValueError(f"Source/reference mismatch for {parent_id}")

    normalized = {}
    for name, path in (("waitk_v1_k1", args.waitk_k1), ("v2_3", args.v2_3)):
        rows, _ = rollout_rows(path)
        normalized[name] = {row["id"]: row for row in rows}

    report: dict[str, Any] = {
        "scientific_status": "locked500_predeclared_paired_comparison_not_gold_asl",
        "selection_used_locked_test": False,
        "baseline_selection": (
            "v1 k=1 was the highest-Token-F1 point in the complete predeclared "
            "modified Wait-k k=1..8 sweep; no k was tuned after seeing v2.3."
        ),
        "common_parent_count": len(ids),
        "same_parent_ids": True,
        "same_source_and_reference": True,
        "difference_direction": "v2_3_minus_waitk_v1_k1",
        "bootstrap": {
            "method": "paired_parent_level_percentile_bootstrap",
            "repetitions": args.bootstrap_repetitions,
            "seed": args.seed,
        },
        "quality_noninferiority_margin": args.quality_noninferiority_margin,
        "inputs": {
            "waitk_v1_k1": {
                "path": str(args.waitk_k1),
                "sha256": sha256(args.waitk_k1),
            },
            "v2_3": {"path": str(args.v2_3), "sha256": sha256(args.v2_3)},
        },
        "means": {},
        "paired_metrics": {},
        "limitations": [
            "The locked reference is generated pseudo ASL, not expert gold ASL.",
            "Timing is based on English token prefixes, not sign-video boundaries.",
            "The comparison evaluates the frozen models and policies on one locked set.",
        ],
    }

    for name in normalized:
        report["means"][name] = {}
        for metric in METRICS:
            values = [
                value
                for parent_id in ids
                if (value := as_number(normalized[name][parent_id][metric])) is not None
            ]
            report["means"][name][metric] = {
                "mean": round(mean(values), 8),
                "valid_count": len(values),
            }

    for metric_index, (metric, direction) in enumerate(METRICS.items()):
        pairs = []
        for parent_id in ids:
            baseline = as_number(normalized["waitk_v1_k1"][parent_id][metric])
            candidate = as_number(normalized["v2_3"][parent_id][metric])
            if baseline is not None and candidate is not None:
                pairs.append((baseline, candidate))
        result = paired_bootstrap(
            [candidate - baseline for baseline, candidate in pairs],
            args.bootstrap_repetitions,
            args.seed + metric_index,
        )
        result.update(
            {
                "direction": direction,
                "paired_count": len(pairs),
                "excluded_missing_count": len(ids) - len(pairs),
                "superiority_conclusion": conclusion(result, direction),
            }
        )
        if metric in {"sequence_similarity", "token_f1"}:
            result["noninferior_at_margin"] = (
                result["ci95_lower"] > -args.quality_noninferiority_margin
            )
        report["paired_metrics"][metric] = result

    quality_noninferior = all(
        report["paired_metrics"][metric]["noninferior_at_margin"]
        for metric in ("sequence_similarity", "token_f1")
    )
    latency_better = (
        report["paired_metrics"]["laal"]["superiority_conclusion"]
        == "statistically_clear_improvement"
    )
    report["paper_gate"] = {
        "quality_noninferior_at_fixed_margin": quality_noninferior,
        "laal_significantly_lower": latency_better,
        "supports_main_claim": quality_noninferior and latency_better,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
