#!/usr/bin/env python3
"""Export interpretable best/worst locked-test latency cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from audit_latency import rollout_rows, write_json


def trace(row: dict[str, Any]) -> str:
    return " | ".join(
        f"{event['source_prefix_length']}:{event['predicted_action']}"
        for event in row.get("rollout_events", [])
    )


def diagnosis(base_metric: dict[str, Any], candidate_metric: dict[str, Any]) -> list[str]:
    flags = []
    if candidate_metric["wait_ratio"] > base_metric["wait_ratio"] + 0.05:
        flags.append("more_WAIT_actions")
    if (
        candidate_metric["first_write_position"] is not None
        and base_metric["first_write_position"] is not None
        and candidate_metric["first_write_position"] > base_metric["first_write_position"]
    ):
        flags.append("later_first_WRITE")
    if candidate_metric["full_source_write_ratio"] > base_metric["full_source_write_ratio"] + 0.1:
        flags.append("more_WRITEs_after_full_source")
    if candidate_metric["prediction_length"] < base_metric["prediction_length"]:
        flags.append("shorter_prediction_check_under_generation")
    if candidate_metric["prediction_length"] > base_metric["prediction_length"]:
        flags.append("longer_prediction")
    return flags or ["distributed_timing_change"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-rollout", type=Path, required=True)
    parser.add_argument("--candidate-rollout", type=Path, required=True)
    parser.add_argument("--baseline-name", default="v1")
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    args = parser.parse_args()

    base_metrics, base_source = rollout_rows(args.baseline_rollout)
    candidate_metrics, candidate_source = rollout_rows(args.candidate_rollout)
    base_by_id = {row["id"]: row for row in base_metrics}
    candidate_by_id = {row["id"]: row for row in candidate_metrics}
    common_ids = sorted(base_by_id.keys() & candidate_by_id.keys())

    rows = []
    for parent_id in common_ids:
        base = base_by_id[parent_id]
        candidate = candidate_by_id[parent_id]
        base_raw = base_source[parent_id]
        candidate_raw = candidate_source[parent_id]
        row = {
            "id": parent_id,
            "source_text_en": candidate_raw["source_text_en"],
            "reference_pseudo_gloss": candidate_raw["target_pseudo_asl_gloss"],
            "baseline_prediction": base_raw["predicted_pseudo_asl_gloss"],
            "candidate_prediction": candidate_raw["predicted_pseudo_asl_gloss"],
            "baseline_laal": base["laal"],
            "candidate_laal": candidate["laal"],
            "delta_laal": (
                None
                if candidate["laal"] is None or base["laal"] is None
                else candidate["laal"] - base["laal"]
            ),
            "baseline_sequence_similarity": base["sequence_similarity"],
            "candidate_sequence_similarity": candidate["sequence_similarity"],
            "delta_sequence_similarity": (
                candidate["sequence_similarity"] - base["sequence_similarity"]
            ),
            "baseline_token_f1": base["token_f1"],
            "candidate_token_f1": candidate["token_f1"],
            "delta_token_f1": candidate["token_f1"] - base["token_f1"],
            "baseline_wait_ratio": base["wait_ratio"],
            "candidate_wait_ratio": candidate["wait_ratio"],
            "baseline_prediction_reference_length_ratio": (
                base["prediction_length"] / max(1, base["reference_length"])
            ),
            "candidate_prediction_reference_length_ratio": (
                candidate["prediction_length"] / max(1, candidate["reference_length"])
            ),
            "baseline_write_positions": base_raw.get("write_source_positions", []),
            "candidate_write_positions": candidate_raw.get("write_source_positions", []),
            "diagnostic_flags": diagnosis(base, candidate),
            "baseline_trace": trace(base_raw),
            "candidate_trace": trace(candidate_raw),
        }
        rows.append(row)

    scored_rows = [row for row in rows if row["delta_laal"] is not None]
    ordered = sorted(scored_rows, key=lambda row: row["delta_laal"])
    clean_improvements = [
        row
        for row in ordered
        if row["delta_laal"] < 0
        and row["candidate_prediction_reference_length_ratio"] >= 0.8
        and row["delta_sequence_similarity"] >= -0.02
        and row["delta_token_f1"] >= -0.02
    ]
    apparent_improvements = [
        row
        for row in ordered
        if row["delta_laal"] < 0
        and (
            row["candidate_prediction_reference_length_ratio"] < 0.8
            or row["delta_sequence_similarity"] < -0.02
            or row["delta_token_f1"] < -0.02
        )
    ]
    selected = []
    for rank, row in enumerate(clean_improvements[: args.top_k], 1):
        selected.append({"category": "latency_improvement_quality_preserved", "rank": rank, **row})
    for rank, row in enumerate(apparent_improvements[: args.top_k], 1):
        selected.append({"category": "apparent_improvement_quality_or_length_warning", "rank": rank, **row})
    for rank, row in enumerate(reversed(ordered[-args.top_k :]), 1):
        selected.append({"category": "largest_latency_regression", "rank": rank, **row})

    report = {
        "scientific_status": "locked_pseudo_gloss_qualitative_latency_audit_not_gold_asl",
        "baseline": args.baseline_name,
        "candidate": args.candidate_name,
        "common_parent_count": len(common_ids),
        "latency_scored_parent_count": len(scored_rows),
        "quality_preserved_latency_improvement_count": len(clean_improvements),
        "apparent_improvement_quality_or_length_warning_count": len(apparent_improvements),
        "top_k_per_direction": args.top_k,
        "cases": selected,
    }
    write_json(args.output_report, report)

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    flat = []
    for row in selected:
        copied = dict(row)
        for key in ("baseline_write_positions", "candidate_write_positions", "diagnostic_flags"):
            copied[key] = json.dumps(copied[key], ensure_ascii=False)
        flat.append(copied)
    with args.output_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
