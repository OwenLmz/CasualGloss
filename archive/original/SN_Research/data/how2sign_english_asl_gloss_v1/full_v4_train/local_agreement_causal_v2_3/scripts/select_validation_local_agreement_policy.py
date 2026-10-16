#!/usr/bin/env python3
"""Select a local-agreement policy using validation metrics only."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--maximum-laal", type=float, required=True)
    parser.add_argument("--maximum-conflict-ratio", type=float, default=0.20)
    parser.add_argument("--minimum-valid-completion", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = []
    for path in args.reports:
        report = read_json(path)
        if report.get("split") != "validation":
            raise SystemExit(f"Policy selection received non-validation report: {path}")
        row = {
            "report": str(path),
            "stability_window": int(report["parameters"]["stability_window"]),
            "minimum_prefix_tokens": int(
                report["parameters"]["minimum_prefix_tokens"]
            ),
            "valid_completion": float(report["valid_completion_ratio"]),
            "sequence_similarity": float(
                report["pseudo_label_sequence_similarity"]
            ),
            "token_f1": float(report["pseudo_label_token_f1"]),
            "laal": float(
                report["mean_length_adaptive_average_lagging_source_tokens"]
            ),
            "conflict_ratio": float(report["irreversible_conflict_parent_ratio"]),
        }
        row["passes_constraints"] = (
            row["valid_completion"] >= args.minimum_valid_completion
            and row["laal"] <= args.maximum_laal
            and row["conflict_ratio"] <= args.maximum_conflict_ratio
        )
        candidates.append(row)

    eligible = [row for row in candidates if row["passes_constraints"]]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                row["sequence_similarity"],
                row["token_f1"],
                -row["laal"],
                -row["conflict_ratio"],
            ),
        )
        status = "selected_highest_quality_within_validation_latency_and_stability_constraints"
    else:
        selected = max(
            candidates,
            key=lambda row: (
                row["valid_completion"],
                -row["laal"],
                row["sequence_similarity"],
                row["token_f1"],
            ),
        )
        status = "no_candidate_passed_all_constraints_exploratory_fallback_only"

    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_split": "validation_only",
        "selection_status": status,
        "constraints": {
            "maximum_laal_source_tokens": args.maximum_laal,
            "maximum_irreversible_conflict_ratio": args.maximum_conflict_ratio,
            "minimum_valid_completion": args.minimum_valid_completion,
        },
        "selected": selected,
        "candidates": sorted(
            candidates,
            key=lambda row: (row["stability_window"], row["minimum_prefix_tokens"]),
        ),
        "locked_test_not_used_for_selection": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
