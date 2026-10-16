#!/usr/bin/env python3
"""Paired confirmation analysis for a fixed v2.3 checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Callable


def read_rows(path: Path) -> dict[str, dict]:
    return {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        )
    }


def bootstrap_ci(values: list[float], repetitions: int, seed: int) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    count = len(values)
    samples = sorted(
        mean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(repetitions)
    )
    lower = samples[int(0.025 * repetitions)]
    upper = samples[min(repetitions - 1, int(0.975 * repetitions))]
    return [lower, upper]


def comparison(
    baseline: dict[str, dict],
    candidate: dict[str, dict],
    *,
    repetitions: int,
    seed: int,
) -> dict:
    if set(baseline) != set(candidate):
        raise RuntimeError("Confirmation parent IDs differ between systems")
    ids = sorted(baseline)
    fields: list[tuple[str, str, Callable[[dict], float | None]]] = [
        (
            "token_f1",
            "candidate_minus_baseline",
            lambda row: row.get("pseudo_gloss_token_f1"),
        ),
        (
            "sequence_similarity",
            "candidate_minus_baseline",
            lambda row: row.get("pseudo_gloss_sequence_similarity"),
        ),
        (
            "laal_source_tokens",
            "candidate_minus_baseline",
            lambda row: row.get("length_adaptive_average_lagging_source_tokens"),
        ),
        (
            "valid_end_completion",
            "candidate_minus_baseline",
            lambda row: 1.0 if row.get("completed_after_full_source") else 0.0,
        ),
        (
            "maximum_actions_reached",
            "candidate_minus_baseline",
            lambda row: 1.0 if row.get("termination_reason") == "maximum_actions_reached" else 0.0,
        ),
    ]
    metrics = {}
    for index, (name, direction, getter) in enumerate(fields):
        paired = []
        baseline_values = []
        candidate_values = []
        for parent_id in ids:
            left = getter(baseline[parent_id])
            right = getter(candidate[parent_id])
            if left is None or right is None:
                continue
            baseline_values.append(float(left))
            candidate_values.append(float(right))
            paired.append(float(right) - float(left))
        metrics[name] = {
            "direction": direction,
            "paired_parent_count": len(paired),
            "baseline_mean": mean(baseline_values) if baseline_values else None,
            "candidate_mean": mean(candidate_values) if candidate_values else None,
            "mean_difference": mean(paired) if paired else None,
            "paired_bootstrap_95_ci": bootstrap_ci(
                paired, repetitions, seed + index
            ),
        }
    return {"parent_count": len(ids), "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-baseline", type=Path, required=True)
    parser.add_argument("--prompt-baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    native = read_rows(args.native_baseline)
    prompt = read_rows(args.prompt_baseline)
    candidate = read_rows(args.candidate)
    report = {
        "scientific_status": "frozen_v2_3_confirmation_over_pseudo_gloss_not_gold_asl",
        "selected_checkpoint": args.selected_checkpoint,
        "selection_used_confirmation": False,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
        "v2_3_vs_v2_2_native": comparison(
            native,
            candidate,
            repetitions=args.bootstrap_repetitions,
            seed=args.seed,
        ),
        "v2_3_vs_v2_2_prompt_guard": comparison(
            prompt,
            candidate,
            repetitions=args.bootstrap_repetitions,
            seed=args.seed + 100,
        ),
        "important_note": (
            "This confirmation set was frozen before v2.3 training and was not used "
            "for checkpoint or hyperparameter selection. Targets remain pseudo glosses."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
