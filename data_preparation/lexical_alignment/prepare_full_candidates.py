#!/usr/bin/env python3
"""Prepare every structurally eligible Fine-tune 1 training parent for alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                row["_input_line_number"] = line_number
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_hash(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("request_id"))


def source_word_count(row: dict[str, Any]) -> int:
    return len(str(row.get("source_text_en", "")).split())


def source_length_bin(row: dict[str, Any]) -> str:
    count = source_word_count(row)
    if count <= 8:
        return "short_le_8"
    if count <= 15:
        return "medium_9_15"
    return "long_ge_16"


def selection_bucket(row: dict[str, Any]) -> str:
    return "automatic_review_flagged" if row.get("automatic_review_flagged") else "standard_completed"


def max_hyphen_components(row: dict[str, Any]) -> int:
    tokens = row.get("target_pseudo_asl_gloss_tokens") or str(
        row.get("target_pseudo_asl_gloss", "")
    ).split()
    return max((len(str(token).split("-")) for token in tokens), default=0)


def canonical_video_key(row: dict[str, Any]) -> str:
    video_ids = sorted(set(row.get("how2sign_video_ids") or []))
    return "|".join(video_ids) if video_ids else f"NO_VIDEO::{row_id(row)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--locked-parent-splits", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    seed = int(config["selection_seed"])
    maximum_hyphen = int(config["maximum_hyphen_components_per_gloss_token"])
    rows = read_jsonl(args.input)
    locked_rows = [
        row
        for row in read_jsonl(args.locked_parent_splits)
        if row.get("split") in {"validation", "test"}
    ]

    prefilter_reason_counts: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    for row in rows:
        reasons: list[str] = []
        if row.get("split") != config["source_split"]:
            reasons.append("not_requested_source_split")
        if not str(row.get("source_text_en", "")).strip():
            reasons.append("empty_source")
        if not str(row.get("target_pseudo_asl_gloss", "")).strip():
            reasons.append("empty_target")
        if row.get("generation_status") != "completed":
            reasons.append("not_llm_completed")
        if max_hyphen_components(row) > maximum_hyphen:
            reasons.append("overjoined_multiword_gloss_token")
        if row.get("pseudo_gloss_is_gold") is not False:
            reasons.append("pseudo_gold_status_not_explicitly_false")
        if reasons:
            prefilter_reason_counts.update(reasons)
            continue

        cleaned = {key: value for key, value in row.items() if key != "_input_line_number"}
        cleaned["candidate_selection_bucket"] = selection_bucket(row)
        cleaned["candidate_source_length_bin"] = source_length_bin(row)
        cleaned["candidate_source_word_count_whitespace"] = source_word_count(row)
        cleaned["candidate_video_group_key"] = canonical_video_key(row)
        eligible.append(cleaned)

    eligible.sort(key=lambda row: stable_hash(seed, row_id(row)))
    for index, row in enumerate(eligible):
        row["candidate_index"] = index
        row["causal_alignment_dataset_version"] = config["dataset_version"]

    candidate_ids = [row_id(row) for row in eligible]
    source_keys = [
        str(row.get("normalized_english_key") or row["source_text_en"]).strip().lower()
        for row in eligible
    ]
    locked_ids_by_split = {
        split: {str(row["id"]) for row in locked_rows if row["split"] == split}
        for split in ("validation", "test")
    }
    expected_locked = {
        split: int(config["locked_evaluation_parent_counts"][split])
        for split in ("validation", "test")
    }
    missing_locked = {
        split: sorted(locked_ids_by_split[split] - set(candidate_ids))
        for split in ("validation", "test")
    }
    locked_video_sets = {
        split: {
            video_id
            for row in locked_rows
            if row["split"] == split
            for video_id in row.get("how2sign_video_ids", [])
        }
        for split in ("validation", "test")
    }

    shard_count = int(config["shard_count"])
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    for index, row in enumerate(eligible):
        shards[index % shard_count].append(row)
    write_jsonl(args.output, eligible)
    args.shard_dir.mkdir(parents=True, exist_ok=True)
    for shard_index, shard_rows in enumerate(shards):
        write_jsonl(args.shard_dir / f"candidate_shard_{shard_index:02d}.jsonl", shard_rows)

    bucket_counts = Counter(row["candidate_selection_bucket"] for row in eligible)
    length_counts = Counter(row["candidate_source_length_bin"] for row in eligible)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": config["dataset_version"],
        "scientific_status": config["scientific_status"],
        "input_manifest": str(args.input),
        "locked_parent_splits_manifest": str(args.locked_parent_splits),
        "input_record_count": len(rows),
        "eligible_after_prefilter_count": len(eligible),
        "prefilter_reason_counts": dict(prefilter_reason_counts),
        "selection_bucket_counts": dict(bucket_counts),
        "source_length_bin_counts": dict(length_counts),
        "unique_parent_id_count": len(set(candidate_ids)),
        "unique_normalized_source_count": len(set(source_keys)),
        "unique_video_id_count": len(
            {video_id for row in eligible for video_id in row.get("how2sign_video_ids", [])}
        ),
        "locked_parent_counts": {
            split: len(locked_ids_by_split[split]) for split in ("validation", "test")
        },
        "missing_locked_parent_counts": {
            split: len(missing_locked[split]) for split in ("validation", "test")
        },
        "missing_locked_parent_ids_preview": {
            split: missing_locked[split][:20] for split in ("validation", "test")
        },
        "locked_video_counts": {
            split: len(locked_video_sets[split]) for split in ("validation", "test")
        },
        "locked_validation_test_video_overlap_count": len(
            locked_video_sets["validation"] & locked_video_sets["test"]
        ),
        "shard_counts": [len(shard) for shard in shards],
        "all_targets_marked_non_gold": all(
            row.get("pseudo_gloss_is_gold") is False for row in eligible
        ),
        "ok": (
            len(eligible) >= int(config["minimum_expected_train_parent_count"])
            and len(set(candidate_ids)) == len(candidate_ids)
            and len(set(source_keys)) == len(source_keys)
            and all(
                len(locked_ids_by_split[split]) == expected_locked[split]
                for split in ("validation", "test")
            )
            and not any(missing_locked.values())
            and not (locked_video_sets["validation"] & locked_video_sets["test"])
            and all(row.get("pseudo_gloss_is_gold") is False for row in eligible)
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
