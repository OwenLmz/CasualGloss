#!/usr/bin/env python3
"""Aggregate and validate the final three-seed TransLLaMA-style test sweep."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
from collections import Counter


METRICS = {
    "token_f1": "mean_pseudo_gloss_token_f1",
    "sequence_similarity": "mean_pseudo_gloss_sequence_similarity",
    "laal": "mean_length_adaptive_average_lagging_source_tokens",
    "bleu4": "bleu_4",
    "length_ratio": "length_ratio",
    "relative_length_error": "mean_relative_length_error",
    "over_generation_rate": "over_generation_rate",
    "completion_rate": "completion_rate",
    "max_actions_rate": "maximum_actions_reached_rate",
    "invalid_action_ratio": "mean_invalid_action_ratio",
}


def mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_glob")
    parser.add_argument("output_json")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    if len(paths) != 3:
        raise SystemExit(f"expected 3 summaries, found {len(paths)}: {paths}")

    summaries = [json.load(open(path, encoding="utf-8")) for path in paths]
    expected_k = list(range(1, 9))
    pair_hashes = {item["sha256"]["pairs"] for item in summaries}
    checks = {
        "summary_count_is_3": len(summaries) == 3,
        "all_test_split": all(item["split"] == "test" for item in summaries),
        "all_test_manifest_accessed": all(item["test_manifest_accessed"] for item in summaries),
        "all_free_running_without_gold_gloss": all(
            not item["rollout_history_uses_gold_committed_gloss"] for item in summaries
        ),
        "all_k_1_to_8": all(item["k_values"] == expected_k for item in summaries),
        "all_500_parents_per_k": all(item["parent_count_per_k"] == 500 for item in summaries),
        "same_locked_test_pairs_hash": len(pair_hashes) == 1,
        "all_native_context_32768": all(
            item["generation"]["native_inference_context_window"] == 32768
            for item in summaries
        ),
        "all_official_loop_safeguard": all(
            "official TransLLaMA" in item["generation"]["loop_safeguard"]
            for item in summaries
        ),
    }

    by_seed = {}
    indexed = []
    for path, summary in zip(paths, summaries):
        seed_name = os.path.basename(os.path.dirname(path))
        rows = {int(row["wait_k"]): row for row in summary["results"]}
        if sorted(rows) != expected_k:
            raise SystemExit(f"missing k rows in {path}: {sorted(rows)}")
        for k, row in rows.items():
            if row["parent_count"] != 500:
                raise SystemExit(f"unexpected parent count in {path}, k={k}")
            if sum(row["termination_reason_counts"].values()) != 500:
                raise SystemExit(f"termination counts do not sum to 500 in {path}, k={k}")
        indexed.append(rows)
        by_seed[seed_name] = {
            "config_sha256": summary["sha256"]["config"],
            "pairs_sha256": summary["sha256"]["pairs"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "max_observed_prompt_tokens": summary["generation"]["max_observed_prompt_tokens"],
            "all_frozen_quality_gates_pass": all(
                row["completion_rate"] >= 0.95
                and row["maximum_actions_reached_rate"] <= 0.05
                and row["mean_invalid_action_ratio"] <= 0.05
                for row in rows.values()
            ),
        }

    aggregate = []
    for k in expected_k:
        rows = [seed_rows[k] for seed_rows in indexed]
        terminations = Counter()
        for row in rows:
            terminations.update(row["termination_reason_counts"])
        item = {
            "wait_k": k,
            "seed_count": 3,
            "parent_count_per_seed": 500,
            "termination_reason_counts_across_1500_rollouts": dict(terminations),
            "metrics": {
                label: mean_sd([float(row[key]) for row in rows])
                for label, key in METRICS.items()
            },
            "all_seed_level_quality_gates_pass": all(
                row["completion_rate"] >= 0.95
                and row["maximum_actions_reached_rate"] <= 0.05
                and row["mean_invalid_action_ratio"] <= 0.05
                for row in rows
            ),
        }
        aggregate.append(item)

    causal_f1 = 0.7047
    causal_laal = 1.7823
    comparisons = []
    for item in aggregate:
        f1 = item["metrics"]["token_f1"]["mean"]
        laal = item["metrics"]["laal"]["mean"]
        comparisons.append(
            {
                "wait_k": item["wait_k"],
                "baseline_mean_token_f1": f1,
                "baseline_mean_laal": laal,
                "causalgloss_minus_baseline_token_f1": causal_f1 - f1,
                "causalgloss_minus_baseline_laal": causal_laal - laal,
                "causalgloss_pareto_dominates_mean_point": causal_f1 > f1 and causal_laal < laal,
            }
        )

    report = {
        "report_type": "three_seed_transllama_style_locked_test_aggregation",
        "inputs": paths,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "locked_test_pairs_sha256": next(iter(pair_hashes)),
        "by_seed": by_seed,
        "aggregate_by_k": aggregate,
        "comparison_reference": {
            "causalgloss_token_f1_mean": causal_f1,
            "causalgloss_laal_mean": causal_laal,
            "note": "Paper-reported CausalGloss three-seed means; descriptive mean-point comparison only.",
        },
        "descriptive_comparison_to_causalgloss": comparisons,
    }

    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
