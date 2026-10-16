#!/usr/bin/env python3
"""Analyze frozen Fixed Wait-k CLERC Epee rollouts under the shared protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from compare_epee_adaptive import (
    load_exact_rollout,
    paired_source_bootstrap,
    parse_named_path,
    percentile_interval,
    score_rollout,
    summarize_seeded,
)
from epee_common import file_sha256, load_manifest, write_json, write_jsonl


def parse_k_seed_path(value: str) -> tuple[int, str, Path]:
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError("Use K:SEED=PATH")
    key, raw_path = value.split("=", 1)
    raw_k, seed = key.split(":", 1)
    return int(raw_k), seed, Path(raw_path)


def independent_seed_hierarchical_bootstrap(
    left: dict[str, dict[str, dict[str, Any]]],
    right: dict[str, dict[str, dict[str, Any]]],
    ids: list[str],
    metric: str,
    repetitions: int,
    seed: int,
    *,
    finite_only: bool = False,
) -> dict[str, Any]:
    usable_ids = [
        row_id
        for row_id in ids
        if not finite_only
        or (
            all(left_seed[row_id][metric] is not None for left_seed in left.values())
            and all(right_seed[row_id][metric] is not None for right_seed in right.values())
        )
    ]
    if not usable_ids:
        raise ValueError(f"No usable source phrases for {metric}")
    left_matrix = np.asarray(
        [[float(rows[row_id][metric]) for row_id in usable_ids] for rows in left.values()],
        dtype=np.float64,
    )
    right_matrix = np.asarray(
        [[float(rows[row_id][metric]) for row_id in usable_ids] for rows in right.values()],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        source_indices = rng.integers(0, len(usable_ids), len(usable_ids))
        left_seed_indices = rng.integers(0, len(left_matrix), len(left_matrix))
        right_seed_indices = rng.integers(0, len(right_matrix), len(right_matrix))
        left_value = left_matrix[left_seed_indices][:, source_indices].mean()
        right_value = right_matrix[right_seed_indices][:, source_indices].mean()
        estimates[repetition] = left_value - right_value
    ci = percentile_interval(estimates)
    return {
        "mean_difference": float(left_matrix.mean() - right_matrix.mean()),
        "ci95": ci,
        "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
        "left_training_seed_count": int(left_matrix.shape[0]),
        "right_training_seed_count": int(right_matrix.shape[0]),
        "paired_source_count": len(usable_ids),
        "seed_resampling": "independent between systems",
    }


def seed_mean_rows(
    seeded: dict[str, dict[str, dict[str, Any]]], ids: list[str], metrics: tuple[str, ...]
) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for row_id in ids:
        result[row_id] = {}
        for metric in metrics:
            values = [rows[row_id][metric] for rows in seeded.values()]
            result[row_id][metric] = (
                mean(float(value) for value in values)
                if all(value is not None for value in values)
                else None
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--causal", action="append", type=parse_named_path, required=True)
    parser.add_argument("--fixed", action="append", type=parse_k_seed_path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=401)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260921)
    parser.add_argument("--primary-k", type=int, default=7)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite analysis output: {args.output_dir}")

    sources = load_manifest(args.manifest, args.expected_count)
    ids = [str(source["id"]) for source in sources]
    source_by_id = {str(source["id"]): source for source in sources}
    causal_paths = dict(args.causal)
    fixed_paths: dict[int, dict[str, Path]] = {}
    for k, seed, path in args.fixed:
        if seed in fixed_paths.setdefault(k, {}):
            raise ValueError(f"Duplicate Fixed Wait-k seed: k={k}, seed={seed}")
        fixed_paths[k][seed] = path
    if set(fixed_paths) != {1, 3, 5, 7}:
        raise ValueError("Fixed Wait-k frontier must contain exactly k={1,3,5,7}")
    if len(fixed_paths[1]) != 1 or len(fixed_paths[3]) != 1:
        raise ValueError("k=1 and k=3 must each contain the single submitted checkpoint")
    if len(fixed_paths[5]) != 3 or len(fixed_paths[7]) != 3:
        raise ValueError("k=5 and k=7 must each contain three submitted checkpoints")

    raw_causal = {
        seed: load_exact_rollout(path, ids) for seed, path in causal_paths.items()
    }
    raw_fixed = {
        k: {seed: load_exact_rollout(path, ids) for seed, path in paths.items()}
        for k, paths in fixed_paths.items()
    }
    causal = {
        seed: {
            row_id: score_rollout(source_by_id[row_id], rows[row_id]) for row_id in ids
        }
        for seed, rows in raw_causal.items()
    }
    fixed = {
        k: {
            seed: {
                row_id: score_rollout(source_by_id[row_id], rows[row_id])
                for row_id in ids
            }
            for seed, rows in seeded.items()
        }
        for k, seeded in raw_fixed.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    long_rows = []
    for seed, rows in sorted(causal.items()):
        for row_id in ids:
            long_rows.append(
                {"system": "causalgloss", "seed": seed, "k": None, **rows[row_id]}
            )
    for k, seeded in sorted(fixed.items()):
        for seed, rows in sorted(seeded.items()):
            for row_id in ids:
                long_rows.append(
                    {"system": "fixed_wait_k", "seed": seed, "k": k, **rows[row_id]}
                )
    write_jsonl(args.output_dir / "per_source_scores.jsonl", long_rows)

    metrics = (
        "content_best_token_f1",
        "laal",
        "failure_penalized_laal",
        "completed",
    )
    causal_mean = seed_mean_rows(causal, ids, metrics)
    report: dict[str, Any] = {
        "scientific_status": "zero_shot_external_evaluation_against_human_annotated_gloss_references",
        "selection_used_epee": False,
        "independent_evaluation_unit": "401 source phrases; signer references nested within phrase",
        "fixed_frontier": [1, 3, 5, 7],
        "primary_fixed_comparison": f"CausalGloss versus Fixed Wait-k={args.primary_k}",
        "primary_k_selection_reason": "highest-quality fixed point on independent locked-500 benchmark; selected before Fixed Wait-k CLERC rollout",
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "input_hashes": {
            "manifest": {"path": str(args.manifest), "sha256": file_sha256(args.manifest)},
            "causal": {
                seed: {"path": str(path), "sha256": file_sha256(path)}
                for seed, path in causal_paths.items()
            },
            "fixed": {
                str(k): {
                    seed: {"path": str(path), "sha256": file_sha256(path)}
                    for seed, path in paths.items()
                }
                for k, paths in fixed_paths.items()
            },
        },
        "systems": {
            "causalgloss": summarize_seeded(causal, ids),
            "fixed_wait_k": {
                str(k): summarize_seeded(seeded, ids) for k, seeded in fixed.items()
            },
        },
        "comparisons": {},
        "limitations": [
            "This is automatic zero-shot evaluation against human annotations, not expert human preference evaluation.",
            "English source-token latency is reported, not speech-time latency.",
            "Fixed Wait-k k=1 and k=3 have one submitted checkpoint each, so their intervals do not estimate fixed-policy training-seed variance.",
            "CausalGloss and Fixed Wait-k training seeds are independently resampled rather than described as matched.",
        ],
    }

    for index, k in enumerate(sorted(fixed)):
        fixed_seeded = fixed[k]
        if len(fixed_seeded) >= 2:
            quality = independent_seed_hierarchical_bootstrap(
                causal,
                fixed_seeded,
                ids,
                "content_best_token_f1",
                args.bootstrap_repetitions,
                args.bootstrap_seed + index * 10,
            )
            latency = independent_seed_hierarchical_bootstrap(
                causal,
                fixed_seeded,
                ids,
                "laal",
                args.bootstrap_repetitions,
                args.bootstrap_seed + index * 10 + 1,
                finite_only=True,
            )
            failure_latency = independent_seed_hierarchical_bootstrap(
                causal,
                fixed_seeded,
                ids,
                "failure_penalized_laal",
                args.bootstrap_repetitions,
                args.bootstrap_seed + index * 10 + 2,
            )
            completion = independent_seed_hierarchical_bootstrap(
                causal,
                fixed_seeded,
                ids,
                "completed",
                args.bootstrap_repetitions,
                args.bootstrap_seed + index * 10 + 3,
            )
            uncertainty = "independent-seed and paired-source hierarchical bootstrap"
        else:
            fixed_mean = seed_mean_rows(fixed_seeded, ids, metrics)

            def source_delta(metric: str, offset: int, finite_only: bool = False) -> dict[str, Any]:
                usable = [
                    row_id
                    for row_id in ids
                    if not finite_only
                    or (
                        causal_mean[row_id][metric] is not None
                        and fixed_mean[row_id][metric] is not None
                    )
                ]
                values = np.asarray(
                    [
                        float(causal_mean[row_id][metric])
                        - float(fixed_mean[row_id][metric])
                        for row_id in usable
                    ],
                    dtype=np.float64,
                )
                return paired_source_bootstrap(
                    values, args.bootstrap_repetitions, args.bootstrap_seed + index * 10 + offset
                )

            quality = source_delta("content_best_token_f1", 0)
            latency = source_delta("laal", 1, finite_only=True)
            failure_latency = source_delta("failure_penalized_laal", 2)
            completion = source_delta("completed", 3)
            uncertainty = "paired-source bootstrap; fixed-policy seed variance not estimated"

        report["comparisons"][str(k)] = {
            "uncertainty_method": uncertainty,
            "quality_causal_minus_fixed": quality,
            "laal_causal_minus_fixed": latency,
            "failure_penalized_laal_causal_minus_fixed": failure_latency,
            "completion_causal_minus_fixed": completion,
            "quality_ci_supports_superiority": quality["ci95"][0] > 0,
            "laal_ci_supports_lower_latency": latency["ci95"][1] < 0,
            "pareto_dominance_supported": quality["ci95"][0] > 0 and latency["ci95"][1] < 0,
            "primary": k == args.primary_k,
        }

    write_json(args.output_dir / "comparison_report.json", report)
    print(json.dumps(report["comparisons"][str(args.primary_k)], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

