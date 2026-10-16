#!/usr/bin/env python3
"""Frozen multi-seed CLERC Epee comparison for the adaptive-policy study."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from epee_common import file_sha256, load_manifest, read_jsonl, write_json, write_jsonl


VARIANT_SUFFIX_RE = re.compile(r"_(\d+)$")
QUALITY_FIELDS = (
    "best_token_f1",
    "best_precision",
    "best_recall",
    "mean_token_f1",
    "best_sequence_similarity",
    "mean_sequence_similarity",
    "minimum_gloss_error_rate",
    "mean_gloss_error_rate",
    "any_exact_match",
)


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def collapse_exact_period(tokens: list[str]) -> list[str]:
    for period in range(1, len(tokens) // 2 + 1):
        if len(tokens) % period == 0 and tokens == tokens[:period] * (len(tokens) // period):
            return tokens[:period]
    return tokens


def normalize_prediction(tokens: list[str], track: str) -> list[str]:
    strict = [" ".join(str(token).upper().replace("’", "'").split()) for token in tokens]
    if track == "strict":
        return collapse_exact_period(strict)
    content = []
    for token in strict:
        token = token.rstrip("?")
        if token == "QUESTION" or not token:
            continue
        content.append(VARIANT_SUFFIX_RE.sub("", token))
    return collapse_exact_period(content)


def token_overlap(reference: list[str], prediction: list[str]) -> tuple[float, float, float]:
    ref_counter, pred_counter = Counter(reference), Counter(prediction)
    overlap = sum((ref_counter & pred_counter).values())
    precision = overlap / max(1, len(prediction))
    recall = overlap / max(1, len(reference))
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return precision, recall, f1


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, 1):
        current = [i]
        for j, right_token in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def score_references(prediction: list[str], references: list[list[str]]) -> dict[str, float]:
    overlap = [token_overlap(reference, prediction) for reference in references]
    best_index = max(range(len(overlap)), key=lambda index: overlap[index][2])
    similarities = [
        SequenceMatcher(None, reference, prediction, autojunk=False).ratio()
        for reference in references
    ]
    ger = [edit_distance(reference, prediction) / max(1, len(reference)) for reference in references]
    return {
        "best_token_f1": overlap[best_index][2],
        "best_precision": overlap[best_index][0],
        "best_recall": overlap[best_index][1],
        "mean_token_f1": mean(value[2] for value in overlap),
        "best_sequence_similarity": max(similarities),
        "mean_sequence_similarity": mean(similarities),
        "minimum_gloss_error_rate": min(ger),
        "mean_gloss_error_rate": mean(ger),
        "any_exact_match": float(any(reference == prediction for reference in references)),
    }


def load_exact_rollout(path: Path, expected_ids: list[str]) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    by_id = {str(row["id"]): row for row in rows}
    if len(rows) != len(by_id):
        raise ValueError(f"Duplicate rollout IDs in {path}")
    if set(by_id) != set(expected_ids):
        missing = set(expected_ids) - set(by_id)
        extra = set(by_id) - set(expected_ids)
        raise ValueError(f"Rollout ID mismatch in {path}: missing={missing}, extra={extra}")
    return by_id


def score_rollout(source: dict[str, Any], rollout: dict[str, Any]) -> dict[str, Any]:
    scored: dict[str, Any] = {
        "id": str(source["id"]),
        "phrase_id": int(source["phrase_id"]),
        "evaluation_subset": source["evaluation_subset"],
        "completed": float(bool(rollout["completed_after_full_source"])),
        "maximum_actions_reached": float(rollout["termination_reason"] == "maximum_actions_reached"),
        "no_output": float(not rollout["predicted_gloss_tokens"]),
        "invalid_action_ratio": float(rollout["invalid_action_ratio"]),
        "action_count": float(rollout["action_count"]),
        "predicted_length": float(len(rollout["predicted_gloss_tokens"])),
        "reference_length": float(source["median_content_reference_length"]),
        "relative_length_error": abs(
            len(rollout["predicted_gloss_tokens"])
            - int(source["median_content_reference_length"])
        )
        / max(1, int(source["median_content_reference_length"])),
        "over_generated": float(
            len(rollout["predicted_gloss_tokens"])
            > int(source["median_content_reference_length"])
        ),
        "laal": rollout["length_adaptive_average_lagging_source_tokens"],
    }
    source_count = len(source["source_tokens"])
    finite_laal = scored["laal"]
    scored["failure_penalized_laal"] = (
        max(float(finite_laal), float(source_count))
        if finite_laal is not None and not bool(scored["completed"])
        else float(finite_laal)
        if finite_laal is not None
        else float(source_count)
    )
    for track, reference_key in (("strict", "strict_tokens"), ("content", "content_tokens")):
        prediction = normalize_prediction(rollout["predicted_gloss_tokens"], track)
        references = [
            normalize_prediction(reference[reference_key], track)
            for reference in source["references"]
            if reference[reference_key]
        ]
        metrics = score_references(prediction, references)
        for metric, value in metrics.items():
            scored[f"{track}_{metric}"] = value
    return scored


def percentile_interval(estimates: np.ndarray) -> list[float]:
    low, high = np.quantile(estimates, [0.025, 0.975])
    return [float(low), float(high)]


def hierarchical_bootstrap(
    differences: list[np.ndarray], repetitions: int, seed: int
) -> dict[str, Any]:
    if not differences or any(len(values) == 0 for values in differences):
        raise ValueError("Hierarchical bootstrap received an empty seed stratum")
    rng = np.random.default_rng(seed)
    seed_count = len(differences)
    estimates = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        selected_seeds = rng.integers(0, seed_count, seed_count)
        seed_estimates = []
        for selected_seed in selected_seeds:
            values = differences[int(selected_seed)]
            indices = rng.integers(0, len(values), len(values))
            seed_estimates.append(float(values[indices].mean()))
        estimates[repetition] = mean(seed_estimates)
    ci = percentile_interval(estimates)
    point = mean(float(values.mean()) for values in differences)
    return {
        "mean_difference": point,
        "ci95": ci,
        "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
        "matched_seed_count": seed_count,
        "paired_source_count_by_seed": [len(values) for values in differences],
    }


def paired_source_bootstrap(differences: np.ndarray, repetitions: int, seed: int) -> dict[str, Any]:
    if len(differences) == 0:
        raise ValueError("Paired bootstrap received no sources")
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        indices = rng.integers(0, len(differences), len(differences))
        estimates[repetition] = differences[indices].mean()
    ci = percentile_interval(estimates)
    return {
        "mean_difference": float(differences.mean()),
        "ci95": ci,
        "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
        "paired_source_count": len(differences),
    }


def summarize_seeded(
    seed_rows: dict[str, dict[str, dict[str, Any]]], ids: list[str]
) -> dict[str, Any]:
    metric_names = [
        "content_best_token_f1",
        "strict_best_token_f1",
        "content_best_precision",
        "content_best_recall",
        "content_mean_token_f1",
        "content_best_sequence_similarity",
        "content_minimum_gloss_error_rate",
        "predicted_length",
        "relative_length_error",
        "over_generated",
        "completed",
        "maximum_actions_reached",
        "no_output",
        "invalid_action_ratio",
        "action_count",
        "failure_penalized_laal",
    ]
    per_seed: dict[str, Any] = {}
    for seed, rows in seed_rows.items():
        values = {
            metric: mean(float(rows[row_id][metric]) for row_id in ids)
            for metric in metric_names
        }
        finite = [float(rows[row_id]["laal"]) for row_id in ids if rows[row_id]["laal"] is not None]
        values["laal"] = mean(finite) if finite else None
        values["finite_laal_count"] = len(finite)
        per_seed[seed] = values
    aggregate = {
        metric: mean(float(per_seed[seed][metric]) for seed in per_seed)
        for metric in metric_names
    }
    finite_seed_laal = [float(per_seed[seed]["laal"]) for seed in per_seed if per_seed[seed]["laal"] is not None]
    aggregate["laal"] = mean(finite_seed_laal) if finite_seed_laal else None
    aggregate["training_seed_count"] = len(seed_rows)
    return {"per_seed": per_seed, "mean_across_seeds": aggregate}


def summarize_single(rows: dict[str, dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    seeded = summarize_seeded({"single_checkpoint": rows}, ids)
    return seeded["mean_across_seeds"] | {"training_seed_count": 1}


def paired_seed_differences(
    left: dict[str, dict[str, dict[str, Any]]],
    right: dict[str, dict[str, dict[str, Any]]],
    ids: list[str],
    metric: str,
    finite_only: bool = False,
) -> list[np.ndarray]:
    if set(left) != set(right):
        raise ValueError("Matched seed IDs differ between systems")
    result = []
    for seed in sorted(left):
        values = []
        for row_id in ids:
            left_value = left[seed][row_id][metric]
            right_value = right[seed][row_id][metric]
            if finite_only and (left_value is None or right_value is None):
                continue
            values.append(float(left_value) - float(right_value))
        result.append(np.asarray(values, dtype=np.float64))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--causal", action="append", type=parse_named_path, required=True)
    parser.add_argument("--learned", action="append", type=parse_named_path, required=True)
    parser.add_argument("--lexical", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=401)
    parser.add_argument("--k-values", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--confirmatory-k", type=int, default=5)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260920)
    parser.add_argument("--noninferiority-margin", type=float, default=-0.03)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite analysis output: {args.output_dir}")

    sources = load_manifest(args.manifest, args.expected_count)
    ids = [str(source["id"]) for source in sources]
    source_by_id = {str(source["id"]): source for source in sources}
    causal_paths = dict(args.causal)
    learned_dirs = dict(args.learned)
    if set(causal_paths) != set(learned_dirs):
        raise ValueError("CausalGloss and learned baseline must use identical seed labels")
    k_values = [int(value) for value in args.k_values.split(",")]
    if args.confirmatory_k not in k_values:
        raise ValueError("Confirmatory k is absent from the requested sweep")

    raw_causal = {
        seed: load_exact_rollout(path, ids) for seed, path in causal_paths.items()
    }
    raw_lexical = load_exact_rollout(args.lexical, ids)
    raw_learned = {
        seed: {
            k: load_exact_rollout(directory / f"k{k}" / "sentence_rollouts.jsonl", ids)
            for k in k_values
        }
        for seed, directory in learned_dirs.items()
    }

    causal = {
        seed: {row_id: score_rollout(source_by_id[row_id], rows[row_id]) for row_id in ids}
        for seed, rows in raw_causal.items()
    }
    lexical = {
        row_id: score_rollout(source_by_id[row_id], raw_lexical[row_id]) for row_id in ids
    }
    learned = {
        seed: {
            k: {
                row_id: score_rollout(source_by_id[row_id], k_rows[row_id])
                for row_id in ids
            }
            for k, k_rows in seed_rows.items()
        }
        for seed, seed_rows in raw_learned.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    long_rows = []
    for seed in sorted(causal):
        for row_id in ids:
            long_rows.append({"system": "causalgloss", "seed": seed, "k": None, **causal[seed][row_id]})
    for seed in sorted(learned):
        for k in k_values:
            for row_id in ids:
                long_rows.append({"system": "transllama_style", "seed": seed, "k": k, **learned[seed][k][row_id]})
    for row_id in ids:
        long_rows.append({"system": "lexical", "seed": "single_checkpoint", "k": None, **lexical[row_id]})
    write_jsonl(args.output_dir / "per_source_scores.jsonl", long_rows)

    learned_by_k = {
        k: {seed: learned[seed][k] for seed in sorted(learned)} for k in k_values
    }
    report: dict[str, Any] = {
        "scientific_status": "zero_shot_external_evaluation_against_human_annotated_gloss_references",
        "selection_used_epee": False,
        "independent_evaluation_unit": "401 source phrases; signer references nested within phrase",
        "primary_quality_endpoint": "content-normalized best-human-reference Token F1",
        "primary_latency_endpoint": "source-token LAAL with frozen median content-reference length",
        "confirmatory_comparison": f"CausalGloss versus TransLLaMA-style k={args.confirmatory_k}",
        "bootstrap": {
            "method": "matched-seed then paired-source hierarchical percentile bootstrap",
            "repetitions": args.bootstrap_repetitions,
            "seed": args.bootstrap_seed,
        },
        "input_hashes": {
            "manifest": {"path": str(args.manifest), "sha256": file_sha256(args.manifest)},
            "causal": {
                seed: {"path": str(path), "sha256": file_sha256(path)}
                for seed, path in causal_paths.items()
            },
            "lexical": {"path": str(args.lexical), "sha256": file_sha256(args.lexical)},
            "learned": {
                seed: {
                    str(k): {
                        "path": str(directory / f"k{k}" / "sentence_rollouts.jsonl"),
                        "sha256": file_sha256(directory / f"k{k}" / "sentence_rollouts.jsonl"),
                    }
                    for k in k_values
                }
                for seed, directory in learned_dirs.items()
            },
        },
        "systems": {
            "causalgloss": summarize_seeded(causal, ids),
            "lexical_single_checkpoint": summarize_single(lexical, ids),
            "transllama_style": {
                str(k): summarize_seeded(learned_by_k[k], ids) for k in k_values
            },
        },
        "confirmatory": {},
        "descriptive_k_sweep": {},
        "lexical_comparisons": {},
        "limitations": [
            "This is automatic zero-shot evaluation against human annotations, not expert human preference evaluation.",
            "Epee is external and out of domain relative to How2Sign pseudo-gloss training.",
            "Best-reference quality accepts signer variation; mean-reference metrics expose convention sensitivity.",
            "English source-token latency is reported, not speech-time latency.",
            "The lexical policy is one checkpoint, so its interval resamples phrases only and does not estimate training-seed variance.",
        ],
    }

    confirmatory_rows = learned_by_k[args.confirmatory_k]
    quality = hierarchical_bootstrap(
        paired_seed_differences(causal, confirmatory_rows, ids, "content_best_token_f1"),
        args.bootstrap_repetitions,
        args.bootstrap_seed,
    )
    latency = hierarchical_bootstrap(
        paired_seed_differences(causal, confirmatory_rows, ids, "laal", finite_only=True),
        args.bootstrap_repetitions,
        args.bootstrap_seed + 1,
    )
    failure_latency = hierarchical_bootstrap(
        paired_seed_differences(causal, confirmatory_rows, ids, "failure_penalized_laal"),
        args.bootstrap_repetitions,
        args.bootstrap_seed + 2,
    )
    completion = hierarchical_bootstrap(
        paired_seed_differences(causal, confirmatory_rows, ids, "completed"),
        args.bootstrap_repetitions,
        args.bootstrap_seed + 3,
    )
    causal_completion = report["systems"]["causalgloss"]["mean_across_seeds"]["completed"]
    learned_completion = report["systems"]["transllama_style"][str(args.confirmatory_k)]["mean_across_seeds"]["completed"]
    quality_noninferior = quality["ci95"][0] > args.noninferiority_margin
    latency_improved = latency["ci95"][1] < 0
    completion_acceptable = causal_completion >= 0.95 and causal_completion + 0.02 >= learned_completion
    report["confirmatory"] = {
        "quality_causal_minus_learned": quality,
        "quality_noninferiority_margin": args.noninferiority_margin,
        "quality_ci_supports_noninferiority": quality_noninferior,
        "quality_ci_supports_superiority": quality["ci95"][0] > 0,
        "laal_causal_minus_learned": latency,
        "laal_ci_supports_lower_latency": latency_improved,
        "failure_penalized_laal_causal_minus_learned": failure_latency,
        "completion_rate_causal_minus_learned": completion,
        "completion_acceptability_rule": "CausalGloss >=0.95 and no more than 0.02 below learned baseline",
        "completion_acceptable": completion_acceptable,
        "strong_support": quality_noninferior and latency_improved and completion_acceptable,
        "interpretation": "Positive quality and completion deltas favor CausalGloss; negative latency deltas favor CausalGloss.",
    }

    for index, k in enumerate(k_values):
        k_rows = learned_by_k[k]
        report["descriptive_k_sweep"][str(k)] = {
            "quality_causal_minus_learned": hierarchical_bootstrap(
                paired_seed_differences(causal, k_rows, ids, "content_best_token_f1"),
                args.bootstrap_repetitions,
                args.bootstrap_seed + 100 + index * 10,
            ),
            "laal_causal_minus_learned": hierarchical_bootstrap(
                paired_seed_differences(causal, k_rows, ids, "laal", finite_only=True),
                args.bootstrap_repetitions,
                args.bootstrap_seed + 101 + index * 10,
            ),
            "confirmatory": k == args.confirmatory_k,
        }

    causal_seed_mean: dict[str, dict[str, float | None]] = {}
    for row_id in ids:
        causal_seed_mean[row_id] = {}
        for metric in (
            "content_best_token_f1",
            "laal",
            "failure_penalized_laal",
            "completed",
        ):
            values = [causal[seed][row_id][metric] for seed in causal]
            causal_seed_mean[row_id][metric] = (
                mean(float(value) for value in values)
                if all(value is not None for value in values)
                else None
            )
    for offset, metric in enumerate(("content_best_token_f1", "failure_penalized_laal", "completed")):
        differences = np.asarray(
            [causal_seed_mean[row_id][metric] - float(lexical[row_id][metric]) for row_id in ids],
            dtype=np.float64,
        )
        report["lexical_comparisons"][f"causal_seed_mean_minus_lexical_{metric}"] = paired_source_bootstrap(
            differences, args.bootstrap_repetitions, args.bootstrap_seed + 500 + offset
        )
    finite_ids = [
        row_id
        for row_id in ids
        if causal_seed_mean[row_id]["laal"] is not None and lexical[row_id]["laal"] is not None
    ]
    report["lexical_comparisons"]["causal_seed_mean_minus_lexical_laal"] = paired_source_bootstrap(
        np.asarray(
            [
                float(causal_seed_mean[row_id]["laal"]) - float(lexical[row_id]["laal"])
                for row_id in finite_ids
            ],
            dtype=np.float64,
        ),
        args.bootstrap_repetitions,
        args.bootstrap_seed + 504,
    )

    write_json(args.output_dir / "comparison_report.json", report)
    print(json.dumps(report["confirmatory"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
