#!/usr/bin/env python3
"""Select auditable success/failure cases from two locked-test rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != len({row["id"] for row in rows}):
        raise ValueError(f"Duplicate IDs in {path}")
    return {row["id"]: row for row in rows}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def has_write_burst(row: dict[str, Any]) -> bool:
    events = row.get("rollout_events", [])
    return any(
        left.get("predicted_action_class") == "WRITE"
        and right.get("predicted_action_class") == "WRITE"
        and left.get("source_prefix_length") == right.get("source_prefix_length")
        for left, right in zip(events, events[1:])
    )


def compact_case(
    category: str,
    baseline_name: str,
    candidate_name: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "category": category,
        "id": candidate["id"],
        "source_text_en": candidate["source_text_en"],
        "reference_pseudo_gloss": candidate["target_pseudo_asl_gloss"],
        baseline_name: {
            "prediction": baseline["predicted_pseudo_asl_gloss"],
            "sequence_similarity": baseline["pseudo_gloss_sequence_similarity"],
            "token_f1": baseline["pseudo_gloss_token_f1"],
            "reference_length_al": baseline.get("reference_length_average_lagging", baseline.get("word_position_average_lagging_source_tokens")),
            "write_source_positions": baseline.get("write_source_positions"),
            "termination_reason": baseline.get("termination_reason"),
        },
        candidate_name: {
            "prediction": candidate["predicted_pseudo_asl_gloss"],
            "sequence_similarity": candidate["pseudo_gloss_sequence_similarity"],
            "token_f1": candidate["pseudo_gloss_token_f1"],
            "reference_length_al": candidate.get("reference_length_average_lagging", candidate.get("word_position_average_lagging_source_tokens")),
            "write_source_positions": candidate.get("write_source_positions"),
            "termination_reason": candidate.get("termination_reason"),
            "rollout_events": candidate.get("rollout_events"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-name", default="v1")
    parser.add_argument("--candidate-name", default="v2_2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=3)
    args = parser.parse_args()

    baseline = read_jsonl(args.baseline)
    candidate = read_jsonl(args.candidate)
    if set(baseline) != set(candidate):
        raise ValueError("Baseline and candidate must contain identical parent IDs")

    scored = []
    for parent_id in sorted(baseline):
        left = baseline[parent_id]
        right = candidate[parent_id]
        if left.get("source_text_en") != right.get("source_text_en") or left.get(
            "target_pseudo_asl_gloss"
        ) != right.get("target_pseudo_asl_gloss"):
            raise ValueError(f"Source/reference mismatch for {parent_id}")
        quality_delta = float(right["pseudo_gloss_token_f1"]) - float(left["pseudo_gloss_token_f1"])
        latency_left_raw = left.get("word_position_average_lagging_source_tokens")
        latency_right_raw = right.get("word_position_average_lagging_source_tokens")
        if latency_left_raw is None or latency_right_raw is None:
            continue
        latency_left = float(latency_left_raw)
        latency_right = float(latency_right_raw)
        scored.append((parent_id, quality_delta, latency_left - latency_right, left, right))

    categories: dict[str, list[tuple[Any, ...]]] = {
        "quality_preserved_latency_improved": sorted(
            [row for row in scored if row[1] >= 0 and row[2] > 0],
            key=lambda row: (row[2], row[1]),
            reverse=True,
        ),
        "early_output_quality_failure": sorted(
            [row for row in scored if row[2] > 0 and row[1] < -0.10],
            key=lambda row: (row[1], -row[2]),
        ),
        "same_prefix_write_burst": sorted(
            [row for row in scored if has_write_burst(row[4])],
            key=lambda row: (row[1], row[2]),
            reverse=True,
        ),
    }

    selected = []
    used: set[str] = set()
    for category, candidates in categories.items():
        count = 0
        for parent_id, _, _, left, right in candidates:
            if parent_id in used:
                continue
            selected.append(
                compact_case(category, args.baseline_name, args.candidate_name, left, right)
            )
            used.add(parent_id)
            count += 1
            if count == args.per_category:
                break

    write_jsonl(args.output, selected)
    summary = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "common_parent_count": len(scored),
        "selection_counts": {
            category: sum(row["category"] == category for row in selected) for category in categories
        },
        "selection_is_for_qualitative_error_analysis_not_metric_estimation": True,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
