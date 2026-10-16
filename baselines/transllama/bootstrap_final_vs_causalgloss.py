#!/usr/bin/env python3
"""Paired parent-level and hierarchical seed-and-parent bootstrap comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


SEEDS = (20260829, 20260904, 20260905)
FIELDS = {
    "token_f1": "pseudo_gloss_token_f1",
    "sequence_similarity": "pseudo_gloss_sequence_similarity",
    "laal": "length_adaptive_average_lagging_source_tokens",
}


def read_jsonl(path: str) -> dict[str, dict]:
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            item_id = row["id"]
            if item_id in rows:
                raise ValueError(f"duplicate id {item_id} in {path}")
            rows[item_id] = row
    return rows


def as_float_or_nan(value: object) -> float:
    return np.nan if value is None else float(value)


def interval(samples: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "lower": float(low),
        "upper": float(high),
        "bootstrap_mean": float(np.mean(samples)),
    }


def parent_bootstrap(diff: np.ndarray, rng: np.random.Generator, draws: int) -> np.ndarray:
    n = diff.shape[0]
    indices = rng.integers(0, n, size=(draws, n))
    return np.nanmean(diff[indices], axis=1)


def hierarchical_bootstrap(
    diff: np.ndarray,
    rng: np.random.Generator,
    draws: int,
    batch_size: int = 500,
) -> np.ndarray:
    seed_count, parent_count = diff.shape
    output = np.empty(draws, dtype=np.float64)
    offset = 0
    while offset < draws:
        batch = min(batch_size, draws - offset)
        seed_indices = rng.integers(0, seed_count, size=(batch, seed_count))
        parent_indices = rng.integers(
            0, parent_count, size=(batch, seed_count, parent_count)
        )
        sampled = diff[seed_indices[:, :, None], parent_indices]
        output[offset : offset + batch] = np.nanmean(sampled, axis=(1, 2))
        offset += batch
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--causal-rollouts", nargs=3, required=True,
                        help="Frozen CausalGloss rollouts in seed order 20260829 20260904 20260905")
    args = parser.parse_args()

    causal = {seed: read_jsonl(path) for seed, path in zip(SEEDS, args.causal_rollouts)}
    baseline = {}
    for seed in SEEDS:
        baseline[seed] = {}
        for k in range(1, 9):
            path = os.path.join(
                args.experiment_root,
                "runs",
                "final_test",
                f"final_seed_{seed}",
                f"k{k}",
                "sentence_rollouts.jsonl",
            )
            baseline[seed][k] = read_jsonl(path)

    canonical_ids = list(causal[SEEDS[0]])
    canonical_set = set(canonical_ids)
    checks = {
        "three_seed_ids": list(SEEDS),
        "causal_has_500_unique_parents_per_seed": all(
            len(causal[seed]) == 500 for seed in SEEDS
        ),
        "baseline_has_500_unique_parents_per_seed_and_k": all(
            len(baseline[seed][k]) == 500 for seed in SEEDS for k in range(1, 9)
        ),
        "all_parent_id_sets_match": all(
            set(causal[seed]) == canonical_set
            and all(set(baseline[seed][k]) == canonical_set for k in range(1, 9))
            for seed in SEEDS
        ),
        "all_target_tokens_match": all(
            causal[seed][item_id]["target_pseudo_asl_gloss_tokens"]
            == baseline[seed][k][item_id]["target_pseudo_asl_gloss_tokens"]
            for seed in SEEDS
            for k in range(1, 9)
            for item_id in canonical_ids
        ),
        "all_source_tokens_match": all(
            causal[seed][item_id]["source_tokens"]
            == baseline[seed][k][item_id]["source_tokens"]
            for seed in SEEDS
            for k in range(1, 9)
            for item_id in canonical_ids
        ),
        "all_rollouts_are_free_running": all(
            not causal[seed][item_id]["rollout_history_uses_gold_committed_gloss"]
            and not baseline[seed][k][item_id]["rollout_history_uses_gold_committed_gloss"]
            for seed in SEEDS
            for k in range(1, 9)
            for item_id in canonical_ids
        ),
    }
    boolean_checks = [value for value in checks.values() if isinstance(value, bool)]
    if not all(boolean_checks):
        raise SystemExit(f"pairing checks failed: {checks}")

    results = []
    for k in range(1, 9):
        metric_result = {}
        for metric_name, field in FIELDS.items():
            diffs = np.empty((len(SEEDS), len(canonical_ids)), dtype=np.float64)
            per_seed = {}
            for seed_index, seed in enumerate(SEEDS):
                causal_values = np.array(
                    [as_float_or_nan(causal[seed][item_id][field]) for item_id in canonical_ids]
                )
                baseline_values = np.array(
                    [as_float_or_nan(baseline[seed][k][item_id][field]) for item_id in canonical_ids]
                )
                diff = causal_values - baseline_values
                diffs[seed_index] = diff
                seed_rng = np.random.default_rng(2026091900 + 100 * k + seed_index)
                per_seed[str(seed)] = {
                    "valid_paired_parent_count": int(np.sum(~np.isnan(diff))),
                    "causalgloss_mean": float(np.nanmean(causal_values)),
                    "baseline_mean": float(np.nanmean(baseline_values)),
                    "paired_mean_difference_causalgloss_minus_baseline": float(np.nanmean(diff)),
                    "paired_parent_bootstrap_95_ci": interval(
                        parent_bootstrap(diff, seed_rng, args.draws)
                    ),
                }

            aggregate_rng = np.random.default_rng(2026091900 + 100 * k + 99)
            hierarchical_samples = hierarchical_bootstrap(
                diffs, aggregate_rng, args.draws
            )
            aggregate = {
                "valid_paired_seed_parent_count": int(np.sum(~np.isnan(diffs))),
                "causalgloss_mean": float(
                    np.nanmean(
                        [
                            as_float_or_nan(causal[seed][item_id][field])
                            for seed in SEEDS
                            for item_id in canonical_ids
                        ]
                    )
                ),
                "baseline_mean": float(
                    np.nanmean(
                        [
                            as_float_or_nan(baseline[seed][k][item_id][field])
                            for seed in SEEDS
                            for item_id in canonical_ids
                        ]
                    )
                ),
                "paired_mean_difference_causalgloss_minus_baseline": float(
                    np.nanmean(diffs)
                ),
                "hierarchical_seed_and_parent_bootstrap_95_ci": interval(
                    hierarchical_samples
                ),
                "bootstrap_probability_difference_above_zero": float(
                    np.mean(hierarchical_samples > 0)
                ),
            }
            metric_result[metric_name] = {
                "difference_direction": "CausalGloss minus TransLLaMA-style baseline",
                "per_seed": per_seed,
                "aggregate": aggregate,
            }
        results.append({"wait_k": k, "metrics": metric_result})

    report = {
        "report_type": "paired_causalgloss_vs_transllama_style_bootstrap",
        "bootstrap_draws": args.draws,
        "independent_seed_count": 3,
        "paired_parent_count_per_seed": 500,
        "checks": checks,
        "all_checks_pass": all(boolean_checks),
        "results": results,
        "interpretation_note": (
            "Differences are CausalGloss minus baseline. Positive quality differences "
            "and negative LAAL differences favor CausalGloss. Intervals quantify "
            "paired uncertainty; an interval containing zero is not evidence of equivalence."
        ),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
