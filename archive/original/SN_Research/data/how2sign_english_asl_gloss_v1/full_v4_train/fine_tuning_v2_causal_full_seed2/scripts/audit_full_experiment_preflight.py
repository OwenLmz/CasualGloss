#!/usr/bin/env python3
"""Audit full data and prove that its locked evaluation actions match the 5K control."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COMPARISON_FIELDS = (
    "id",
    "parent_id",
    "split",
    "state_prompt",
    "target_action",
    "target_action_type",
    "source_prefix_length",
    "source_token_count",
    "source_complete",
    "required_source_position",
    "partial_source_text_en",
    "committed_pseudo_asl_gloss",
    "parent_target_pseudo_asl_gloss",
    "how2sign_video_ids",
    "target_is_gold",
    "alignment_is_gold",
    "timeline_policy",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_projection(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: value["id"]):
        projected = {field: row.get(field) for field in COMPARISON_FIELDS}
        digest.update(json.dumps(projected, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--control-5k-root", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    validation = read_json(args.full_root / "reports" / "validation_report.json")
    full_parents = read_jsonl(args.full_root / "manifests" / "parent_splits_full.jsonl")
    parent_ids = {
        split: {row["id"] for row in full_parents if row["split"] == split}
        for split in ("train", "validation", "test")
    }
    videos = {
        split: {
            video_id
            for row in full_parents
            if row["split"] == split
            for video_id in row.get("how2sign_video_ids", [])
        }
        for split in ("train", "validation", "test")
    }

    split_reports: dict[str, Any] = {}
    all_exact = True
    for split in ("validation", "test"):
        full_rows = read_jsonl(args.full_root / "manifests" / f"{split}_actions.jsonl")
        control_rows = read_jsonl(
            args.control_5k_root / "manifests" / f"{split}_actions.jsonl"
        )
        full_by_id = {row["id"]: row for row in full_rows}
        control_by_id = {row["id"]: row for row in control_rows}
        shared_ids = set(full_by_id) & set(control_by_id)
        mismatched_ids = [
            row_id
            for row_id in sorted(shared_ids)
            if any(
                full_by_id[row_id].get(field) != control_by_id[row_id].get(field)
                for field in COMPARISON_FIELDS
            )
        ]
        exact = (
            set(full_by_id) == set(control_by_id)
            and not mismatched_ids
            and sha256_projection(full_rows) == sha256_projection(control_rows)
        )
        all_exact = all_exact and exact
        split_reports[split] = {
            "full_action_count": len(full_rows),
            "control_5k_action_count": len(control_rows),
            "action_id_set_exact_match": set(full_by_id) == set(control_by_id),
            "projected_content_mismatch_count": len(mismatched_ids),
            "projected_content_mismatch_ids_preview": mismatched_ids[:20],
            "full_projection_sha256": sha256_projection(full_rows),
            "control_5k_projection_sha256": sha256_projection(control_rows),
            "exact_match": exact,
        }

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "full_causal_finetune_preflight_pseudo_labels_not_gold_asl",
        "full_data_validation_ok": bool(validation.get("ok")),
        "parent_split_counts": {
            split: len(parent_ids[split]) for split in ("train", "validation", "test")
        },
        "action_split_counts": validation["action_split_counts"],
        "locked_evaluation_action_comparison": split_reports,
        "parent_id_overlap_counts": {
            "train_validation": len(parent_ids["train"] & parent_ids["validation"]),
            "train_test": len(parent_ids["train"] & parent_ids["test"]),
            "validation_test": len(parent_ids["validation"] & parent_ids["test"]),
        },
        "source_video_overlap_counts": {
            "train_validation": len(videos["train"] & videos["validation"]),
            "train_test": len(videos["train"] & videos["test"]),
            "validation_test": len(videos["validation"] & videos["test"]),
        },
        "all_targets_non_gold": bool(
            validation.get("all_pseudo_targets_and_alignments_marked_non_gold")
        ),
        "ok": (
            bool(validation.get("ok"))
            and len(parent_ids["train"]) == 18947
            and len(parent_ids["validation"]) == 500
            and len(parent_ids["test"]) == 500
            and validation["action_split_counts"]
            == {"train": 397700, "validation": 9209, "test": 8365}
            and all_exact
            and not any(
                (
                    parent_ids["train"] & parent_ids["validation"],
                    parent_ids["train"] & parent_ids["test"],
                    parent_ids["validation"] & parent_ids["test"],
                )
            )
            and not any(
                (
                    videos["train"] & videos["validation"],
                    videos["train"] & videos["test"],
                    videos["validation"] & videos["test"],
                )
            )
            and bool(validation.get("all_pseudo_targets_and_alignments_marked_non_gold"))
        ),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
