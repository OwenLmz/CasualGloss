#!/usr/bin/env python3
"""Score paired Epee rollouts against frozen multi-reference glosses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


VARIANT_SUFFIX_RE = re.compile(r"_(\d+)$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collapse_exact_period(tokens: list[str]) -> list[str]:
    for period in range(1, len(tokens) // 2 + 1):
        if len(tokens) % period == 0 and tokens == tokens[:period] * (len(tokens) // period):
            return tokens[:period]
    return tokens


def normalize_prediction(tokens: list[str], track: str) -> list[str]:
    strict = [" ".join(token.upper().replace("’", "'").split()) for token in tokens]
    if track == "strict":
        return collapse_exact_period(strict)
    content = []
    for token in strict:
        token = token.rstrip("?")
        if token == "QUESTION" or not token:
            continue
        content.append(VARIANT_SUFFIX_RE.sub("", token))
    return collapse_exact_period(content)


def token_f1(reference: list[str], prediction: list[str]) -> float:
    ref_counter, pred_counter = Counter(reference), Counter(prediction)
    overlap = sum((ref_counter & pred_counter).values())
    precision = overlap / max(1, len(prediction))
    recall = overlap / max(1, len(reference))
    return 2 * precision * recall / max(1e-12, precision + recall)


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, 1):
        current = [i]
        for j, right_token in enumerate(right, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left_token != right_token))
            )
        previous = current
    return previous[-1]


def score_references(prediction: list[str], references: list[list[str]]) -> dict[str, float]:
    f1_values = [token_f1(reference, prediction) for reference in references]
    sequence_values = [
        SequenceMatcher(None, reference, prediction, autojunk=False).ratio()
        for reference in references
    ]
    ger_values = [edit_distance(reference, prediction) / max(1, len(reference)) for reference in references]
    exact_values = [float(reference == prediction) for reference in references]
    return {
        "best_token_f1": max(f1_values),
        "mean_token_f1": mean(f1_values),
        "best_sequence_similarity": max(sequence_values),
        "mean_sequence_similarity": mean(sequence_values),
        "minimum_gloss_error_rate": min(ger_values),
        "mean_gloss_error_rate": mean(ger_values),
        "any_exact_match": max(exact_values),
    }


def bootstrap_difference(
    differences: np.ndarray, repetitions: int, seed: int, margin: float | None = None
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = rng.integers(0, len(differences), len(differences))
        values[index] = differences[sample].mean()
    low, high = np.quantile(values, [0.025, 0.975])
    result = {
        "mean_difference_v2_2_minus_v1": float(differences.mean()),
        "ci95": [float(low), float(high)],
        "ci_excludes_zero": bool(low > 0 or high < 0),
    }
    if margin is not None:
        result["noninferiority_margin"] = margin
        result["point_estimate_within_margin"] = bool(differences.mean() >= margin)
        result["noninferiority_supported_by_ci"] = bool(low > margin)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--v1-rollouts", type=Path, required=True)
    parser.add_argument("--v2-2-rollouts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--noninferiority-margin", type=float, default=-0.03)
    args = parser.parse_args()

    sources = read_jsonl(args.manifest)
    v1 = {row["id"]: row for row in read_jsonl(args.v1_rollouts)}
    v22 = {row["id"]: row for row in read_jsonl(args.v2_2_rollouts)}
    ids = [row["id"] for row in sources]
    if len(ids) != 401 or set(ids) != set(v1) or set(ids) != set(v22):
        raise ValueError("Manifest and paired rollout IDs do not match exactly")

    per_source = []
    for source in sources:
        row: dict[str, Any] = {
            "id": source["id"],
            "phrase_id": source["phrase_id"],
            "evaluation_subset": source["evaluation_subset"],
            "source_text_en": source["source_text_en"],
            "reference_count": source["reference_count"],
            "v1_prediction": v1[source["id"]]["predicted_gloss"],
            "v2_2_prediction": v22[source["id"]]["predicted_gloss"],
            "v1_laal": v1[source["id"]]["length_adaptive_average_lagging_source_tokens"],
            "v2_2_laal": v22[source["id"]]["length_adaptive_average_lagging_source_tokens"],
            "v1_completed": v1[source["id"]]["completed_after_full_source"],
            "v2_2_completed": v22[source["id"]]["completed_after_full_source"],
        }
        for track, reference_key in (("strict", "strict_tokens"), ("content", "content_tokens")):
            references = [ref[reference_key] for ref in source["references"] if ref[reference_key]]
            for model_name, rollout in (("v1", v1), ("v2_2", v22)):
                prediction = normalize_prediction(rollout[source["id"]]["predicted_gloss_tokens"], track)
                scores = score_references(prediction, references)
                for metric, value in scores.items():
                    row[f"{model_name}_{track}_{metric}"] = value
        per_source.append(row)

    subsets = {
        "all_401": per_source,
        "parallel_multi_reference": [
            row for row in per_source if row["evaluation_subset"] == "parallel_multi_reference"
        ],
        "second_register_single_reference": [
            row for row in per_source if row["evaluation_subset"] == "second_register_single_reference"
        ],
    }
    metric_names = [
        "best_token_f1", "mean_token_f1", "best_sequence_similarity",
        "mean_sequence_similarity", "minimum_gloss_error_rate",
        "mean_gloss_error_rate", "any_exact_match",
    ]
    report: dict[str, Any] = {
        "scientific_status": "paired_external_human_asl_gloss_reference_evaluation",
        "primary_endpoint": "content_best_token_f1 on all_401",
        "noninferiority_margin": args.noninferiority_margin,
        "bootstrap": {
            "method": "paired source-level percentile bootstrap",
            "repetitions": args.bootstrap_repetitions,
            "seed": args.seed,
        },
        "subsets": {},
        "input_sha256": {
            "manifest": sha256(args.manifest),
            "v1_rollouts": sha256(args.v1_rollouts),
            "v2_2_rollouts": sha256(args.v2_2_rollouts),
        },
        "important_limitations": [
            "Epee is an external clip-level corpus rather than a text-to-gloss generation benchmark.",
            "Best-of-reference scores accept signer variation; mean-of-reference scores expose convention sensitivity.",
            "The models were trained on How2Sign pseudo gloss and are evaluated zero-shot on Epee.",
        ],
    }
    summary_rows = []
    for subset_index, (subset_name, rows) in enumerate(subsets.items()):
        if not rows:
            continue
        subset_report: dict[str, Any] = {"source_count": len(rows), "tracks": {}}
        for track in ("strict", "content"):
            track_report = {}
            for metric_index, metric in enumerate(metric_names):
                v1_values = np.array([row[f"v1_{track}_{metric}"] for row in rows], dtype=float)
                v22_values = np.array([row[f"v2_2_{track}_{metric}"] for row in rows], dtype=float)
                margin = args.noninferiority_margin if metric == "best_token_f1" else None
                comparison = bootstrap_difference(
                    v22_values - v1_values,
                    args.bootstrap_repetitions,
                    args.seed + subset_index * 100 + metric_index,
                    margin,
                )
                track_report[metric] = {
                    "v1_mean": float(v1_values.mean()),
                    "v2_2_mean": float(v22_values.mean()),
                    **comparison,
                }
                summary_rows.append(
                    {
                        "subset": subset_name,
                        "track": track,
                        "metric": metric,
                        "source_count": len(rows),
                        "v1_mean": float(v1_values.mean()),
                        "v2_2_mean": float(v22_values.mean()),
                        "difference": comparison["mean_difference_v2_2_minus_v1"],
                        "ci95_low": comparison["ci95"][0],
                        "ci95_high": comparison["ci95"][1],
                    }
                )
            subset_report["tracks"][track] = track_report
        for metric in ("laal", "completed"):
            paired_rows = rows
            if metric == "laal":
                paired_rows = [
                    row
                    for row in rows
                    if row["v1_laal"] is not None and row["v2_2_laal"] is not None
                ]
            if not paired_rows:
                raise ValueError(f"No valid paired rows for {subset_name} {metric}")
            v1_values = np.array(
                [float(row[f"v1_{metric}"]) for row in paired_rows], dtype=float
            )
            v22_values = np.array(
                [float(row[f"v2_2_{metric}"]) for row in paired_rows], dtype=float
            )
            subset_report[metric] = {
                "paired_source_count": len(paired_rows),
                "v1_mean": float(v1_values.mean()),
                "v2_2_mean": float(v22_values.mean()),
                **bootstrap_difference(
                    v22_values - v1_values,
                    args.bootstrap_repetitions,
                    args.seed + subset_index * 100 + 90 + (metric == "completed"),
                ),
            }
        report["subsets"][subset_name] = subset_report

    primary = report["subsets"]["all_401"]["tracks"]["content"]["best_token_f1"]
    report["decision_gate"] = {
        "quality_point_drop_within_0_03": primary["point_estimate_within_margin"],
        "quality_noninferiority_supported_by_95ci": primary["noninferiority_supported_by_ci"],
        "latency_improved": report["subsets"]["all_401"]["laal"][
            "mean_difference_v2_2_minus_v1"
        ] < 0,
        "interpretation_rule": (
            "Strong paper support requires content best Token F1 non-inferiority by 95% CI "
            "and lower LAAL. A point estimate within margin without CI support is preliminary only."
        ),
    }
    write_json(args.output_dir / "external_comparison.json", report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)
    with (args.output_dir / "per_source_scores.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in per_source:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    figure_rows = []
    for subset_name in subsets:
        metric = report["subsets"][subset_name]["tracks"]["content"]["best_token_f1"]
        figure_rows.append((subset_name, metric["mean_difference_v2_2_minus_v1"], *metric["ci95"]))
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    y = np.arange(len(figure_rows))
    centers = np.array([row[1] for row in figure_rows])
    lows = np.array([row[2] for row in figure_rows])
    highs = np.array([row[3] for row in figure_rows])
    ax.errorbar(centers, y, xerr=[centers - lows, highs - centers], fmt="o", color="#15836D", capsize=4)
    ax.axvline(0, color="#303642", linewidth=1)
    ax.axvline(args.noninferiority_margin, color="#D97706", linestyle="--", linewidth=1.4, label="-0.03 margin")
    ax.set_yticks(y, [row[0].replace("_", " ") for row in figure_rows])
    ax.set_xlabel("v2.2 - v1 content best Token F1 (95% paired bootstrap CI)")
    ax.set_title("CLERC Epee external multi-reference quality difference")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "external_quality_difference.png", dpi=220)
    fig.savefig(args.output_dir / "external_quality_difference.pdf")
    plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
