#!/usr/bin/env python3
"""Select reproducible paired examples from the Epee external evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compact(row: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "evaluation_subset": row["evaluation_subset"],
        "source_text_en": row["source_text_en"],
        "human_content_references": [
            " ".join(reference["content_tokens"]) for reference in source["references"]
        ],
        "v1_prediction": row["v1_prediction"],
        "v2_2_prediction": row["v2_2_prediction"],
        "v1_content_best_token_f1": row["v1_content_best_token_f1"],
        "v2_2_content_best_token_f1": row["v2_2_content_best_token_f1"],
        "difference_v2_2_minus_v1": (
            row["v2_2_content_best_token_f1"] - row["v1_content_best_token_f1"]
        ),
        "v1_laal": row["v1_laal"],
        "v2_2_laal": row["v2_2_laal"],
        "v1_completed": row["v1_completed"],
        "v2_2_completed": row["v2_2_completed"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()

    rows = read_jsonl(args.scores)
    sources = {row["id"]: row for row in read_jsonl(args.manifest)}
    if set(sources) != {row["id"] for row in rows}:
        raise ValueError("Score and source IDs differ")

    report: dict[str, Any] = {
        "selection_metric": "content best-of-reference Token F1, v2.2 minus v1",
        "selection_uses_locked_test_for_tuning": False,
        "subsets": {},
    }
    for subset in sorted({row["evaluation_subset"] for row in rows}):
        subset_rows = [row for row in rows if row["evaluation_subset"] == subset]
        ranked = sorted(
            subset_rows,
            key=lambda row: (
                row["v2_2_content_best_token_f1"] - row["v1_content_best_token_f1"],
                row["id"],
            ),
        )
        differences = [
            row["v2_2_content_best_token_f1"] - row["v1_content_best_token_f1"]
            for row in subset_rows
        ]
        report["subsets"][subset] = {
            "source_count": len(subset_rows),
            "outcome_counts": dict(
                Counter(
                    "v2_2_win_gt_0_03"
                    if difference > 0.03
                    else "v1_win_gt_0_03"
                    if difference < -0.03
                    else "within_0_03"
                    for difference in differences
                )
            ),
            "v1_strongest_examples": [
                compact(row, sources[row["id"]]) for row in ranked[: args.count]
            ],
            "v2_2_strongest_examples": [
                compact(row, sources[row["id"]])
                for row in reversed(ranked[-args.count :])
            ],
            "v2_2_completion_failures": [
                compact(row, sources[row["id"]])
                for row in subset_rows
                if not row["v2_2_completed"]
            ],
        }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

