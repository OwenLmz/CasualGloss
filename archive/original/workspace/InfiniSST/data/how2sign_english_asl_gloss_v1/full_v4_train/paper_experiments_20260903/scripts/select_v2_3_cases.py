#!/usr/bin/env python3
"""Select auditable v2.3 success and failure cases from frozen locked rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


AUDIT_SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "causal_alignment_latency_audit_v2_2"
    / "scripts"
)
sys.path.insert(0, str(AUDIT_SCRIPTS))

from audit_latency import rollout_rows  # noqa: E402


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    indexed = {row["id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate IDs in {path}")
    return indexed


def has_write_burst(row: dict[str, Any]) -> bool:
    positions = row.get("write_source_positions", [])
    return any(left == right for left, right in zip(positions, positions[1:]))


def first_write(row: dict[str, Any]) -> int | None:
    positions = row.get("write_source_positions", [])
    return int(positions[0]) if positions else None


def laal(row: dict[str, Any]) -> float | None:
    value = row.get("laal", row.get("length_adaptive_average_lagging_source_tokens"))
    return None if value is None else float(value)


def compact(
    category: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    baseline_audited: dict[str, Any],
    candidate_audited: dict[str, Any],
) -> dict[str, Any]:
    left_laal = laal(baseline_audited)
    right_laal = laal(candidate_audited)
    return {
        "category": category,
        "id": candidate["id"],
        "source_text_en": candidate["source_text_en"],
        "reference_pseudo_gloss": candidate["target_pseudo_asl_gloss"],
        "v1": {
            "prediction": baseline["predicted_pseudo_asl_gloss"],
            "token_f1": baseline["pseudo_gloss_token_f1"],
            "sequence_similarity": baseline["pseudo_gloss_sequence_similarity"],
            "laal": left_laal,
            "first_write_source_position": first_write(baseline),
            "write_source_positions": baseline.get("write_source_positions", []),
            "termination_reason": baseline.get("termination_reason"),
        },
        "v2_3": {
            "prediction": candidate["predicted_pseudo_asl_gloss"],
            "token_f1": candidate["pseudo_gloss_token_f1"],
            "sequence_similarity": candidate["pseudo_gloss_sequence_similarity"],
            "laal": right_laal,
            "first_write_source_position": first_write(candidate),
            "write_source_positions": candidate.get("write_source_positions", []),
            "termination_reason": candidate.get("termination_reason"),
            "has_same_prefix_write_burst": has_write_burst(candidate),
        },
        "deltas_v2_3_minus_v1": {
            "token_f1": round(
                float(candidate["pseudo_gloss_token_f1"])
                - float(baseline["pseudo_gloss_token_f1"]),
                8,
            ),
            "laal": round(right_laal - left_laal, 8),
        },
        "interpretation_guard": (
            "The category is descriptive. It does not establish that earlier emission "
            "caused a quality change."
        ),
    }


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Frozen Locked-500 Qualitative Cases",
        "",
        "Selected after all aggregate metrics were frozen. These examples support error analysis only.",
        "The reference is pseudo gloss rather than expert gold ASL.",
        "",
    ]
    for index, row in enumerate(rows, 1):
        lines.extend(
            [
                f"## {index}. {row['category']}",
                "",
                f"- ID: `{row['id']}`",
                f"- English: {row['source_text_en']}",
                f"- Reference pseudo gloss: `{row['reference_pseudo_gloss']}`",
                f"- v1: `{row['v1']['prediction']}`",
                f"- v2.3: `{row['v2_3']['prediction']}`",
                (
                    f"- Delta: Token F1 {row['deltas_v2_3_minus_v1']['token_f1']:+.4f}; "
                    f"LAAL {row['deltas_v2_3_minus_v1']['laal']:+.4f} source words"
                ),
                f"- v2.3 burst WRITE: {row['v2_3']['has_same_prefix_write_burst']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2-3", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=2)
    args = parser.parse_args()

    v1 = read_jsonl(args.v1)
    v23 = read_jsonl(args.v2_3)
    if len(v1) != 500 or len(v23) != 500 or set(v1) != set(v23):
        raise ValueError("Both inputs must contain the same 500 unique parent IDs")
    audited_v1_rows, _ = rollout_rows(args.v1)
    audited_v23_rows, _ = rollout_rows(args.v2_3)
    audited_v1 = {row["id"]: row for row in audited_v1_rows}
    audited_v23 = {row["id"]: row for row in audited_v23_rows}

    candidates = []
    for parent_id in sorted(v1):
        left, right = v1[parent_id], v23[parent_id]
        if left.get("source_text_en") != right.get("source_text_en") or left.get(
            "target_pseudo_asl_gloss"
        ) != right.get("target_pseudo_asl_gloss"):
            raise ValueError(f"Source/reference mismatch for {parent_id}")
        left_audited, right_audited = audited_v1[parent_id], audited_v23[parent_id]
        left_laal, right_laal = laal(left_audited), laal(right_audited)
        if left_laal is None or right_laal is None:
            continue
        f1_delta = float(right["pseudo_gloss_token_f1"]) - float(
            left["pseudo_gloss_token_f1"]
        )
        laal_delta = right_laal - left_laal
        first_left, first_right = first_write(left), first_write(right)
        candidates.append(
            {
                "id": parent_id,
                "v1": left,
                "v2_3": right,
                "v1_audited": left_audited,
                "v2_3_audited": right_audited,
                "f1_delta": f1_delta,
                "laal_delta": laal_delta,
                "first_write_delta": (
                    None
                    if first_left is None or first_right is None
                    else first_right - first_left
                ),
            }
        )

    pools = {
        "quality_preserved_latency_improved": sorted(
            [row for row in candidates if row["f1_delta"] >= -0.02 and row["laal_delta"] <= -0.5],
            key=lambda row: (row["laal_delta"], -row["f1_delta"]),
        ),
        "quality_and_latency_improved": sorted(
            [row for row in candidates if row["f1_delta"] >= 0.10 and row["laal_delta"] < 0],
            key=lambda row: (row["f1_delta"], -row["laal_delta"]),
            reverse=True,
        ),
        "earlier_emission_with_quality_loss": sorted(
            [
                row
                for row in candidates
                if row["f1_delta"] <= -0.10
                and row["first_write_delta"] is not None
                and row["first_write_delta"] < 0
            ],
            key=lambda row: (row["f1_delta"], row["first_write_delta"]),
        ),
        "successful_same_prefix_burst": sorted(
            [
                row
                for row in candidates
                if has_write_burst(row["v2_3"])
                and row["f1_delta"] >= 0
                and row["laal_delta"] < 0
            ],
            key=lambda row: (row["f1_delta"], -row["laal_delta"]),
            reverse=True,
        ),
    }

    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    available_counts = {name: len(pool) for name, pool in pools.items()}
    for category, pool in pools.items():
        for row in pool:
            if row["id"] in used:
                continue
            selected.append(
                compact(
                    category,
                    row["v1"],
                    row["v2_3"],
                    row["v1_audited"],
                    row["v2_3_audited"],
                )
            )
            used.add(row["id"])
            if sum(item["category"] == category for item in selected) >= args.per_category:
                break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "selected_v2_3_vs_v1_cases.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_markdown(args.output_dir / "selected_v2_3_vs_v1_cases.md", selected)
    summary = {
        "scientific_status": "post_hoc_qualitative_error_analysis_not_metric_estimation",
        "common_parent_count": len(candidates),
        "available_counts": available_counts,
        "selected_counts": {
            category: sum(row["category"] == category for row in selected)
            for category in pools
        },
        "selection_after_aggregate_metrics_frozen": True,
        "reference_is_expert_gold_asl": False,
    }
    (args.output_dir / "case_selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
