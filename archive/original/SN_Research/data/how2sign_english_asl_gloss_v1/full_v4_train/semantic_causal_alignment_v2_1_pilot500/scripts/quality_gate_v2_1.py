#!/usr/bin/env python3
"""Apply predeclared structural and metric gates to Semantic Alignment v2.1."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def structural_errors(row: dict[str, Any]) -> list[str]:
    errors = []
    timeline = row["causal_timeline"]
    gloss = row["target_pseudo_asl_gloss_tokens"]
    writes = [item["action"] for item in timeline if str(item["action_type"]).startswith("WRITE")]
    positions = [int(item["source_prefix_length"]) for item in timeline]
    write_positions = [
        int(item["source_prefix_length"])
        for item in timeline
        if str(item["action_type"]).startswith("WRITE")
    ]
    source_count = len(row["source_tokens"])
    if writes != gloss:
        errors.append("write_sequence_mismatch")
    if positions != sorted(positions):
        errors.append("timeline_source_positions_not_monotonic")
    if write_positions != sorted(write_positions):
        errors.append("write_positions_not_monotonic")
    if any(position < 1 or position > source_count for position in positions):
        errors.append("source_position_out_of_range")
    if not timeline or timeline[-1]["action"] != "<END>" or timeline[-1]["action_type"] != "END":
        errors.append("missing_final_end")
    if sum(item["action_type"] == "END" for item in timeline) != 1:
        errors.append("end_count_not_one")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stress-audit", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--stress-preview", type=Path, required=True)
    parser.add_argument("--random-preview", type=Path, required=True)
    args = parser.parse_args()

    pairs = read_jsonl(args.pairs)
    actions = read_jsonl(args.actions)
    summary = json.loads(args.summary.read_text(encoding="utf-8-sig"))
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    stress_audit = read_jsonl(args.stress_audit)
    stress_ids = {row["id"] for row in stress_audit}
    pair_by_id = {row["id"]: row for row in pairs}
    action_counts = Counter(row["parent_id"] for row in actions)
    errors_by_id = {row["id"]: structural_errors(row) for row in pairs}
    eligible = [row for row in pairs if row["eligible_for_finetune2_v2_1"]]
    structural_failure_ids = [row_id for row_id, errors in errors_by_id.items() if errors]
    action_mismatch_ids = [
        row["id"] for row in eligible if action_counts[row["id"]] != len(row["causal_timeline"])
    ]
    ineligible_action_ids = sorted({row["parent_id"] for row in actions} - {row["id"] for row in eligible})

    stress_rows = [pair_by_id[row_id] for row_id in sorted(stress_ids) if row_id in pair_by_id]
    stress_eligible = [row for row in stress_rows if row["eligible_for_finetune2_v2_1"]]
    stress_mostly_full = [
        row for row in stress_eligible
        if row["metrics"]["full_source_emit_token_count"] / max(1, len(row["target_pseudo_asl_gloss_tokens"])) > 0.5
    ]
    stress_ratio = len(stress_mostly_full) / max(1, len(stress_eligible))

    generator = random.Random(int(config["selection_seed"]) + 910)
    random_rows = generator.sample(eligible, min(50, len(eligible)))
    random_structural_failures = [row["id"] for row in random_rows if errors_by_id[row["id"]]]

    thresholds = config["quality_gate"]
    metrics = summary["v2_1"]
    checks = {
        "semantic_sufficiency": metrics["semantic_sufficiency_ratio"] >= thresholds["minimum_semantic_sufficiency"],
        "normalized_average_lagging": metrics["mean_normalized_average_lagging"] <= thresholds["maximum_mean_normalized_average_lagging"],
        "full_source_emits": metrics["mean_full_source_emit_tokens_per_parent"] <= thresholds["maximum_mean_full_source_emits_per_parent"],
        "semantic_gap": metrics["mean_semantic_gap_nats"] <= thresholds["maximum_mean_semantic_gap_nats"],
        "late_negation": metrics["late_negation_violation_count"] <= thresholds["maximum_late_negation_violation_count"],
        "stress_mostly_full_source": stress_ratio <= thresholds["maximum_stress_mostly_full_source_ratio"],
        "structural_integrity": len(structural_failure_ids) <= thresholds["maximum_structural_failure_count"],
        "action_manifest_integrity": not action_mismatch_ids and not ineligible_action_ids,
        "random50_structural_integrity": not random_structural_failures,
    }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "quality_gate_passed": all(checks.values()),
        "checks": checks,
        "thresholds": thresholds,
        "observed": {
            **metrics,
            "parent_count": len(pairs),
            "eligible_parent_count": len(eligible),
            "excluded_parent_count": len(pairs) - len(eligible),
            "stress_parent_count": len(stress_rows),
            "stress_eligible_parent_count": len(stress_eligible),
            "stress_excluded_parent_count": len(stress_rows) - len(stress_eligible),
            "stress_mostly_full_source_count": len(stress_mostly_full),
            "stress_mostly_full_source_ratio": round(stress_ratio, 6),
            "random50_count": len(random_rows),
            "structural_failure_count": len(structural_failure_ids),
            "action_count_mismatch_count": len(action_mismatch_ids),
            "ineligible_action_parent_count": len(ineligible_action_ids),
        },
        "structural_failure_ids": structural_failure_ids,
        "action_count_mismatch_ids": action_mismatch_ids,
        "ineligible_action_parent_ids": ineligible_action_ids,
        "random50_structural_failure_ids": random_structural_failures,
        "interpretation": (
            "Passing this gate supports engineering-scale expansion of pseudo alignment. "
            "It does not establish gold ASL correctness."
        ),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def write_preview(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "id", "source_text_en", "target_pseudo_asl_gloss", "eligible_for_finetune2_v2_1",
            "v2_1_exclusion_reasons", "scheduled_positions", "normalized_al", "semantic_gap",
            "full_source_emit_count", "bounded_delay_rescue_count", "structural_errors",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "id": row["id"],
                        "source_text_en": row["source_text_en"],
                        "target_pseudo_asl_gloss": row["target_pseudo_asl_gloss"],
                        "eligible_for_finetune2_v2_1": row["eligible_for_finetune2_v2_1"],
                        "v2_1_exclusion_reasons": json.dumps(row["v2_1_exclusion_reasons"]),
                        "scheduled_positions": json.dumps([item["scheduled_emit_source_position"] for item in row["semantic_gloss_token_alignment"]]),
                        "normalized_al": row["metrics"]["word_position_normalized_average_lagging"],
                        "semantic_gap": row["metrics"]["mean_scheduled_to_full_logprob_gap_nats"],
                        "full_source_emit_count": row["metrics"]["full_source_emit_token_count"],
                        "bounded_delay_rescue_count": row["metrics"]["bounded_delay_rescue_token_count"],
                        "structural_errors": json.dumps(errors_by_id[row["id"]]),
                    }
                )

    write_preview(args.stress_preview, stress_rows)
    write_preview(args.random_preview, random_rows)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["quality_gate_passed"] else 2)


if __name__ == "__main__":
    main()
