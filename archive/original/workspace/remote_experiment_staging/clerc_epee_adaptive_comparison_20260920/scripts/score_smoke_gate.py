#!/usr/bin/env python3
"""Score fixed smoke rollouts and enforce predeclared FP16/BF16 tolerances."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any

from epee_common import load_manifest, read_jsonl, write_json


VARIANT_SUFFIX_RE = re.compile(r"_(\d+)$")


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def parse_pair(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NEW_NAME=OLD_NAME")
    return tuple(value.split("=", 1))  # type: ignore[return-value]


def collapse_exact_period(tokens: list[str]) -> list[str]:
    for period in range(1, len(tokens) // 2 + 1):
        if len(tokens) % period == 0 and tokens == tokens[:period] * (len(tokens) // period):
            return tokens[:period]
    return tokens


def normalize(tokens: list[str], track: str) -> list[str]:
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


def overlap(reference: list[str], prediction: list[str]) -> tuple[float, float, float]:
    ref_counter, pred_counter = Counter(reference), Counter(prediction)
    common = sum((ref_counter & pred_counter).values())
    precision = common / max(1, len(prediction))
    recall = common / max(1, len(reference))
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return precision, recall, f1


def score_source(source: dict[str, Any], rollout: dict[str, Any], track: str) -> dict[str, float]:
    prediction = normalize(list(rollout.get("predicted_gloss_tokens", [])), track)
    reference_key = "strict_tokens" if track == "strict" else "content_tokens"
    references = [
        normalize(list(reference[reference_key]), track)
        for reference in source["references"]
        if reference[reference_key]
    ]
    triples = [overlap(reference, prediction) for reference in references]
    best_index = max(range(len(triples)), key=lambda index: triples[index][2])
    similarities = [
        SequenceMatcher(None, reference, prediction, autojunk=False).ratio()
        for reference in references
    ]
    return {
        "best_precision": triples[best_index][0],
        "best_recall": triples[best_index][1],
        "best_token_f1": triples[best_index][2],
        "mean_token_f1": mean(value[2] for value in triples),
        "best_sequence_similarity": max(similarities),
        "exact_match": float(any(reference == prediction for reference in references)),
        "predicted_length": float(len(prediction)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--rollout", action="append", type=parse_named_path, required=True)
    parser.add_argument("--precision-pair", action="append", type=parse_pair, default=[])
    parser.add_argument("--f1-tolerance", type=float, default=0.01)
    parser.add_argument("--laal-tolerance", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = load_manifest(args.manifest, args.expected_count)
    source_by_id = {str(source["id"]): source for source in sources}
    expected_ids = set(source_by_id)
    rollouts: dict[str, dict[str, dict[str, Any]]] = {}
    for name, path in args.rollout:
        rows = {str(row["id"]): row for row in read_jsonl(path)}
        missing = expected_ids - set(rows)
        if missing:
            raise ValueError(f"{name} misses smoke IDs: {sorted(missing)}")
        rollouts[name] = {row_id: rows[row_id] for row_id in expected_ids}

    report: dict[str, Any] = {
        "source_count": len(sources),
        "systems": {},
        "precision_compatibility": {},
    }
    per_source: dict[str, dict[str, dict[str, float]]] = {}
    for name, rows in rollouts.items():
        name_scores: dict[str, dict[str, float]] = {}
        for row_id in expected_ids:
            content = score_source(source_by_id[row_id], rows[row_id], "content")
            strict = score_source(source_by_id[row_id], rows[row_id], "strict")
            name_scores[row_id] = {
                "content_best_token_f1": content["best_token_f1"],
                "strict_best_token_f1": strict["best_token_f1"],
            }
        per_source[name] = name_scores
        finite_laal = [
            float(rows[row_id]["length_adaptive_average_lagging_source_tokens"])
            for row_id in expected_ids
            if rows[row_id].get("length_adaptive_average_lagging_source_tokens") is not None
        ]
        report["systems"][name] = {
            "mean_content_best_token_f1": mean(
                score["content_best_token_f1"] for score in name_scores.values()
            ),
            "mean_strict_best_token_f1": mean(
                score["strict_best_token_f1"] for score in name_scores.values()
            ),
            "finite_laal_count": len(finite_laal),
            "mean_laal_source_tokens": mean(finite_laal) if finite_laal else None,
            "completion_rate": mean(
                float(bool(rows[row_id].get("completed_after_full_source")))
                for row_id in expected_ids
            ),
        }

    all_passed = True
    for new_name, old_name in args.precision_pair:
        if new_name not in rollouts or old_name not in rollouts:
            raise ValueError(f"Unknown precision pair: {new_name}={old_name}")
        new_metrics = report["systems"][new_name]
        old_metrics = report["systems"][old_name]
        f1_delta = abs(
            new_metrics["mean_content_best_token_f1"]
            - old_metrics["mean_content_best_token_f1"]
        )
        paired_laal_ids = [
            row_id
            for row_id in expected_ids
            if rollouts[new_name][row_id].get("length_adaptive_average_lagging_source_tokens") is not None
            and rollouts[old_name][row_id].get("length_adaptive_average_lagging_source_tokens") is not None
        ]
        if paired_laal_ids:
            new_laal = mean(
                float(rollouts[new_name][row_id]["length_adaptive_average_lagging_source_tokens"])
                for row_id in paired_laal_ids
            )
            old_laal = mean(
                float(rollouts[old_name][row_id]["length_adaptive_average_lagging_source_tokens"])
                for row_id in paired_laal_ids
            )
            laal_delta = abs(new_laal - old_laal)
        else:
            laal_delta = None
        exact_rate = mean(
            float(
                rollouts[new_name][row_id].get("predicted_gloss_tokens")
                == rollouts[old_name][row_id].get("predicted_gloss_tokens")
            )
            for row_id in expected_ids
        )
        passed = f1_delta <= args.f1_tolerance and (
            laal_delta is not None and laal_delta <= args.laal_tolerance
        )
        all_passed = all_passed and passed
        report["precision_compatibility"][f"{new_name}_vs_{old_name}"] = {
            "absolute_content_f1_delta": f1_delta,
            "absolute_paired_mean_laal_delta": laal_delta,
            "paired_laal_count": len(paired_laal_ids),
            "exact_prediction_rate": exact_rate,
            "f1_tolerance": args.f1_tolerance,
            "laal_tolerance": args.laal_tolerance,
            "passed": passed,
        }
    report["gate_passed"] = all_passed
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.precision_pair and not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
