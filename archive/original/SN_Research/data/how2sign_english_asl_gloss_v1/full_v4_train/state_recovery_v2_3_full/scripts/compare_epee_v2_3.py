#!/usr/bin/env python3
"""Zero-shot multi-reference Epee comparison for frozen v2.3."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


VARIANT_SUFFIX_RE = re.compile(r"_(\d+)$")
EXPECTED_SYSTEMS = ("v1", "v2_2", "v2_3")


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
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left_token != right_token)))
        previous = current
    return previous[-1]


def score_references(prediction: list[str], references: list[list[str]]) -> dict[str, float]:
    f1 = [token_f1(reference, prediction) for reference in references]
    similarity = [SequenceMatcher(None, reference, prediction, autojunk=False).ratio() for reference in references]
    ger = [edit_distance(reference, prediction) / max(1, len(reference)) for reference in references]
    exact = [float(reference == prediction) for reference in references]
    return {
        "best_token_f1": max(f1),
        "mean_token_f1": mean(f1),
        "best_sequence_similarity": max(similarity),
        "mean_sequence_similarity": mean(similarity),
        "minimum_gloss_error_rate": min(ger),
        "mean_gloss_error_rate": mean(ger),
        "any_exact_match": max(exact),
    }


def bootstrap(differences: np.ndarray, repetitions: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = rng.integers(0, len(differences), len(differences))
        estimates[index] = differences[sample].mean()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "mean_difference": float(differences.mean()),
        "ci95": [float(low), float(high)],
        "ci_excludes_zero": bool(low > 0 or high < 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--version", action="append", type=parse_version, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--noninferiority-margin", type=float, default=-0.03)
    args = parser.parse_args()

    paths = dict(args.version)
    if tuple(sorted(paths)) != tuple(sorted(EXPECTED_SYSTEMS)):
        raise ValueError(f"Expected {EXPECTED_SYSTEMS}, found {sorted(paths)}")
    sources = read_jsonl(args.manifest)
    ids = [row["id"] for row in sources]
    if len(ids) != 401 or len(set(ids)) != 401:
        raise ValueError("Expected exactly 401 unique Epee sources")
    rollouts = {name: {row["id"]: row for row in read_jsonl(paths[name])} for name in EXPECTED_SYSTEMS}
    if any(set(rows) != set(ids) for rows in rollouts.values()):
        raise ValueError("Manifest and rollout IDs do not match exactly")

    scored = []
    for source in sources:
        row: dict[str, Any] = {
            "id": source["id"],
            "evaluation_subset": source["evaluation_subset"],
        }
        for name in EXPECTED_SYSTEMS:
            rollout = rollouts[name][source["id"]]
            row[f"{name}_laal"] = rollout["length_adaptive_average_lagging_source_tokens"]
            row[f"{name}_completed"] = float(rollout["completed_after_full_source"])
        for track, reference_key in (("strict", "strict_tokens"), ("content", "content_tokens")):
            references = [reference[reference_key] for reference in source["references"] if reference[reference_key]]
            for name in EXPECTED_SYSTEMS:
                prediction = normalize_prediction(rollouts[name][source["id"]]["predicted_gloss_tokens"], track)
                for metric, value in score_references(prediction, references).items():
                    row[f"{name}_{track}_{metric}"] = value
        scored.append(row)

    subsets = {
        "all_401": scored,
        "parallel_multi_reference": [row for row in scored if row["evaluation_subset"] == "parallel_multi_reference"],
        "second_register_single_reference": [row for row in scored if row["evaluation_subset"] == "second_register_single_reference"],
    }
    quality_metrics = (
        "best_token_f1",
        "mean_token_f1",
        "best_sequence_similarity",
        "mean_sequence_similarity",
        "minimum_gloss_error_rate",
        "mean_gloss_error_rate",
        "any_exact_match",
    )
    report: dict[str, Any] = {
        "scientific_status": "one_shot_external_human_asl_gloss_reference_evaluation",
        "selection_used_epee": False,
        "primary_endpoint": "all_401 content best_token_f1",
        "bootstrap": {
            "method": "paired_source_level_percentile_bootstrap",
            "repetitions": args.bootstrap_repetitions,
            "seed": args.seed,
        },
        "input_files": {
            "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)},
            **{name: {"path": str(paths[name]), "sha256": sha256(paths[name])} for name in EXPECTED_SYSTEMS},
        },
        "subsets": {},
        "limitations": [
            "Epee is an external clip-level corpus rather than a text-to-gloss generation benchmark.",
            "Best-of-reference scores accept signer variation; mean scores expose convention sensitivity.",
            "The model was trained on How2Sign pseudo gloss and evaluated zero-shot on Epee human gloss.",
            "Epee was used once after development and confirmation decisions were frozen.",
        ],
    }

    for subset_index, (subset_name, rows) in enumerate(subsets.items()):
        subset_report: dict[str, Any] = {"source_count": len(rows), "tracks": {}}
        for track in ("strict", "content"):
            track_report = {}
            for metric_index, metric in enumerate(quality_metrics):
                means = {
                    name: float(np.mean([row[f"{name}_{track}_{metric}"] for row in rows]))
                    for name in EXPECTED_SYSTEMS
                }
                comparisons = {}
                candidate = np.array([row[f"v2_3_{track}_{metric}"] for row in rows], dtype=float)
                for baseline_index, baseline in enumerate(("v1", "v2_2")):
                    base = np.array([row[f"{baseline}_{track}_{metric}"] for row in rows], dtype=float)
                    comparisons[f"v2_3_vs_{baseline}"] = bootstrap(
                        candidate - base,
                        args.bootstrap_repetitions,
                        args.seed + subset_index * 1000 + metric_index * 10 + baseline_index,
                    )
                track_report[metric] = {"means": means, "comparisons": comparisons}
            subset_report["tracks"][track] = track_report

        for metric_index, metric in enumerate(("laal", "completed")):
            valid_rows = [
                row for row in rows
                if all(row[f"{name}_{metric}"] is not None for name in EXPECTED_SYSTEMS)
            ]
            means = {name: mean(float(row[f"{name}_{metric}"]) for row in valid_rows) for name in EXPECTED_SYSTEMS}
            candidate = np.array([float(row[f"v2_3_{metric}"]) for row in valid_rows])
            comparisons = {}
            for baseline_index, baseline in enumerate(("v1", "v2_2")):
                base = np.array([float(row[f"{baseline}_{metric}"]) for row in valid_rows])
                comparisons[f"v2_3_vs_{baseline}"] = bootstrap(
                    candidate - base,
                    args.bootstrap_repetitions,
                    args.seed + subset_index * 1000 + 900 + metric_index * 10 + baseline_index,
                )
            subset_report[metric] = {
                "paired_source_count": len(valid_rows),
                "means": means,
                "comparisons": comparisons,
            }
        report["subsets"][subset_name] = subset_report

    primary = report["subsets"]["all_401"]["tracks"]["content"]["best_token_f1"]
    delta_v1 = primary["comparisons"]["v2_3_vs_v1"]
    delta_v22 = primary["comparisons"]["v2_3_vs_v2_2"]
    report["decision_summary"] = {
        "v2_3_vs_v1_quality_noninferiority_margin": args.noninferiority_margin,
        "v2_3_vs_v1_point_noninferior": delta_v1["mean_difference"] >= args.noninferiority_margin,
        "v2_3_vs_v1_ci_supports_noninferiority": delta_v1["ci95"][0] > args.noninferiority_margin,
        "v2_3_vs_v2_2_quality_improved": delta_v22["ci95"][0] > 0,
        "interpretation": "Positive quality deltas favor v2.3; negative LAAL deltas favor v2.3.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
