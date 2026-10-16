#!/usr/bin/env python3
"""Post-hoc completed-case sensitivity analysis for CLERC Epee.

The primary all-prompt result is preserved. This script asks whether the paired
Token F1 advantage remains when each TransLLaMA-style seed is evaluated only on
the prompts for which that seed completed normally. CausalGloss is evaluated on
the identical prompt IDs for each matched seed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np


METRIC = "content_best_token_f1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def percentile_interval(values: np.ndarray) -> list[float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return [float(low), float(high)]


def hierarchical_paired_summary(
    paired_values: list[tuple[np.ndarray, np.ndarray]],
    repetitions: int,
    random_seed: int,
) -> dict[str, Any]:
    if not paired_values or any(len(left) == 0 for left, _ in paired_values):
        raise ValueError("Hierarchical bootstrap received an empty seed stratum")
    if any(len(left) != len(right) for left, right in paired_values):
        raise ValueError("Paired arrays differ in length")

    rng = np.random.default_rng(random_seed)
    seed_count = len(paired_values)
    left_estimates = np.empty(repetitions, dtype=np.float64)
    right_estimates = np.empty(repetitions, dtype=np.float64)

    for repetition in range(repetitions):
        selected_seeds = rng.integers(0, seed_count, seed_count)
        left_seed_means: list[float] = []
        right_seed_means: list[float] = []
        for selected_seed in selected_seeds:
            left, right = paired_values[int(selected_seed)]
            indices = rng.integers(0, len(left), len(left))
            left_seed_means.append(float(left[indices].mean()))
            right_seed_means.append(float(right[indices].mean()))
        left_estimates[repetition] = mean(left_seed_means)
        right_estimates[repetition] = mean(right_seed_means)

    left_point = mean(float(left.mean()) for left, _ in paired_values)
    right_point = mean(float(right.mean()) for _, right in paired_values)
    difference_estimates = left_estimates - right_estimates
    return {
        "causalgloss_mean_token_f1": left_point,
        "causalgloss_ci95": percentile_interval(left_estimates),
        "transllama_mean_token_f1": right_point,
        "transllama_ci95": percentile_interval(right_estimates),
        "causal_minus_transllama": left_point - right_point,
        "difference_ci95": percentile_interval(difference_estimates),
        "difference_ci_excludes_zero": bool(
            np.quantile(difference_estimates, 0.025) > 0
            or np.quantile(difference_estimates, 0.975) < 0
        ),
        "matched_seed_count": seed_count,
        "paired_prompt_count_by_seed": [len(left) for left, _ in paired_values],
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": random_seed,
    }


def index_rows(
    rows: Iterable[dict[str, Any]],
    *,
    system: str,
    k: int | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    selected: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("system") != system:
            continue
        if k is not None and int(row.get("k")) != k:
            continue
        seed = str(row["seed"])
        row_id = str(row["id"])
        if row_id in selected.setdefault(seed, {}):
            raise ValueError(f"Duplicate row for system={system}, seed={seed}, id={row_id}")
        selected[seed][row_id] = row
    return selected


def arrays_for_ids(
    causal: dict[str, dict[str, Any]],
    learned: dict[str, dict[str, Any]],
    ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([float(causal[row_id][METRIC]) for row_id in ids], dtype=np.float64),
        np.asarray([float(learned[row_id][METRIC]) for row_id in ids], dtype=np.float64),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=401)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_921)
    args = parser.parse_args()

    rows = read_jsonl(args.scores)
    causal = index_rows(rows, system="causalgloss", k=None)
    learned = index_rows(rows, system="transllama_style", k=args.k)
    if set(causal) != set(learned):
        raise ValueError(
            f"Matched seed IDs differ: causal={sorted(causal)}, learned={sorted(learned)}"
        )
    seeds = sorted(causal)
    if len(seeds) != 3:
        raise ValueError(f"Expected three matched seeds, found {seeds}")

    all_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    completed_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    incomplete_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    completed_ids_by_seed: dict[str, set[str]] = {}
    per_seed: list[dict[str, Any]] = []

    for seed in seeds:
        causal_ids = set(causal[seed])
        learned_ids = set(learned[seed])
        if causal_ids != learned_ids:
            raise ValueError(f"Prompt IDs differ for seed {seed}")
        if len(causal_ids) != args.expected_count:
            raise ValueError(
                f"Expected {args.expected_count} prompts for seed {seed}, found {len(causal_ids)}"
            )
        if any(float(row["completed"]) != 1.0 for row in causal[seed].values()):
            raise ValueError(f"CausalGloss was not complete for every prompt in seed {seed}")

        all_ids = sorted(causal_ids)
        completed_ids = sorted(
            row_id
            for row_id in all_ids
            if float(learned[seed][row_id]["completed"]) == 1.0
        )
        incomplete_ids = sorted(causal_ids - set(completed_ids))
        completed_ids_by_seed[seed] = set(completed_ids)

        all_pair = arrays_for_ids(causal[seed], learned[seed], all_ids)
        completed_pair = arrays_for_ids(causal[seed], learned[seed], completed_ids)
        incomplete_pair = arrays_for_ids(causal[seed], learned[seed], incomplete_ids)
        all_pairs.append(all_pair)
        completed_pairs.append(completed_pair)
        incomplete_pairs.append(incomplete_pair)

        per_seed.append(
            {
                "seed": seed,
                "total_prompts": len(all_ids),
                "completed_prompts": len(completed_ids),
                "completion_rate": len(completed_ids) / len(all_ids),
                "incomplete_prompts": len(incomplete_ids),
                "completed_causalgloss_f1": float(completed_pair[0].mean()),
                "completed_transllama_f1": float(completed_pair[1].mean()),
                "completed_difference": float(
                    completed_pair[0].mean() - completed_pair[1].mean()
                ),
                "incomplete_causalgloss_f1": float(incomplete_pair[0].mean()),
                "incomplete_transllama_f1": float(incomplete_pair[1].mean()),
                "incomplete_difference": float(
                    incomplete_pair[0].mean() - incomplete_pair[1].mean()
                ),
            }
        )

    common_completed_ids = sorted(set.intersection(*completed_ids_by_seed.values()))
    common_pairs = [
        arrays_for_ids(causal[seed], learned[seed], common_completed_ids)
        for seed in seeds
    ]

    report = {
        "analysis_status": "post_hoc_completed_case_sensitivity_analysis",
        "primary_result_unchanged": True,
        "estimand": (
            "For each matched training seed, mean paired content best-reference Token F1 "
            "difference on prompts where the TransLLaMA-style k=5 rollout emitted END "
            "after consuming the full source. CausalGloss is evaluated on the identical "
            "prompt IDs."
        ),
        "independent_unit": "CLERC Epee English prompt (401 total); signer references remain nested within prompt",
        "selection_note": (
            "Completion is an outcome-dependent post-hoc filter that favors the learned baseline; "
            "this analysis is a sensitivity check and does not replace the all-prompt primary result."
        ),
        "input": {
            "path": str(args.scores.resolve()),
            "sha256": sha256(args.scores),
            "row_count": len(rows),
        },
        "systems": {
            "left": "CausalGloss",
            "right": f"TransLLaMA-style k={args.k}",
            "metric": METRIC,
        },
        "per_seed": per_seed,
        "all_prompts_replication_check": hierarchical_paired_summary(
            all_pairs,
            args.bootstrap_repetitions,
            20_260_920,
        ),
        "seed_specific_completed_cases": hierarchical_paired_summary(
            completed_pairs,
            args.bootstrap_repetitions,
            args.bootstrap_seed,
        ),
        "seed_specific_incomplete_cases_descriptive": hierarchical_paired_summary(
            incomplete_pairs,
            args.bootstrap_repetitions,
            args.bootstrap_seed + 1,
        ),
        "common_completed_subset": {
            "prompt_count": len(common_completed_ids),
            "definition": "Prompts completed by every TransLLaMA-style k=5 seed",
            "summary": hierarchical_paired_summary(
                common_pairs,
                args.bootstrap_repetitions,
                args.bootstrap_seed + 2,
            ),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.output_dir / "completed_only_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    table_path = args.output_dir / "completed_only_per_seed.tsv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(per_seed)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
