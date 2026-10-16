#!/usr/bin/env python3
"""Apply a predeclared latency/quality acceptance rule to paired comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-report", type=Path, required=True)
    parser.add_argument("--baseline", default="v1")
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--quality-noninferiority-margin", type=float, default=0.02)
    parser.add_argument("--completion-noninferiority-margin", type=float, default=0.01)
    parser.add_argument("--length-error-tolerance", type=float, default=0.02)
    parser.add_argument("--invalid-ratio-margin", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = read_json(args.comparison_report)
    decisions = []
    for candidate in args.candidate:
        key = f"{candidate}_vs_{args.baseline}"
        metrics = report["paired_comparisons"][key]
        means = report["versions"][candidate]["means"]
        baseline_means = report["versions"][args.baseline]["means"]
        laal = metrics["laal"]
        sequence = metrics["sequence_similarity"]
        token_f1 = metrics["token_f1"]
        completion = metrics["completed_after_full_source"]
        invalid = metrics["invalid_ratio"]
        latency_improved = laal["ci95_upper"] < 0
        sequence_noninferior = sequence["ci95_lower"] > -args.quality_noninferiority_margin
        token_f1_noninferior = token_f1["ci95_lower"] > -args.quality_noninferiority_margin
        completion_noninferior = (
            completion["ci95_lower"] > -args.completion_noninferiority_margin
        )
        invalid_noninferior = invalid["ci95_upper"] <= args.invalid_ratio_margin
        length_error_ok = (
            means["prediction_reference_length_absolute_error"]
            <= baseline_means["prediction_reference_length_absolute_error"]
            + args.length_error_tolerance
        )
        accepted = all(
            (
                latency_improved,
                sequence_noninferior,
                token_f1_noninferior,
                completion_noninferior,
                invalid_noninferior,
                length_error_ok,
            )
        )
        decisions.append(
            {
                "candidate": candidate,
                "accepted": accepted,
                "checks": {
                    "laal_ci95_upper_below_zero": latency_improved,
                    "sequence_similarity_noninferior": sequence_noninferior,
                    "token_f1_noninferior": token_f1_noninferior,
                    "completion_noninferior": completion_noninferior,
                    "invalid_ratio_noninferior": invalid_noninferior,
                    "prediction_length_error_within_tolerance": length_error_ok,
                },
                "key_intervals": {
                    "laal_delta": laal,
                    "sequence_similarity_delta": sequence,
                    "token_f1_delta": token_f1,
                    "completed_after_full_source_delta": completion,
                    "invalid_ratio_delta": invalid,
                },
            }
        )

    accepted = [row for row in decisions if row["accepted"]]
    selected = None
    if accepted:
        selected = min(
            accepted,
            key=lambda row: report["versions"][row["candidate"]]["means"]["laal"],
        )["candidate"]
    output = {
        "scientific_status": "predeclared_pseudo_gloss_latency_quality_decision",
        "baseline": args.baseline,
        "common_parent_count": report["common_parent_count"],
        "criteria": {
            "latency": "paired LAAL delta 95% CI upper bound below zero",
            "quality_noninferiority_margin": args.quality_noninferiority_margin,
            "completion_noninferiority_margin": args.completion_noninferiority_margin,
            "completion_metric": "completed_after_full_source",
            "prediction_length_absolute_error_tolerance": args.length_error_tolerance,
            "invalid_ratio_margin": args.invalid_ratio_margin,
        },
        "decisions": decisions,
        "selected_candidate": selected,
        "fallback": args.baseline if selected is None else None,
        "important_note": "All content metrics use pseudo gloss references, not verified gold ASL.",
    }
    write_json(args.output, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
