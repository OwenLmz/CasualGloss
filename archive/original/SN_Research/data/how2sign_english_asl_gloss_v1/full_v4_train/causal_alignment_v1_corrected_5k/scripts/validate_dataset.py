#!/usr/bin/env python3
"""Audit the corrected 5K causal-alignment dataset and hash core files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--hash-report", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    pairs_path = args.root / "outputs" / "causal_aligned_pairs_5000.jsonl"
    actions_path = args.root / "outputs" / "prefix_action_examples_corrected_5000.jsonl"
    parent_splits_path = args.root / "manifests" / "parent_splits_5000.jsonl"
    pairs = read_jsonl(pairs_path)
    actions = read_jsonl(actions_path)
    parent_splits = read_jsonl(parent_splits_path)
    pair_by_id = {row["id"]: row for row in pairs}
    actions_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for action in actions:
        actions_by_parent[action["parent_id"]].append(action)

    duplicate_parent_id_count = len(pairs) - len(pair_by_id)
    duplicate_action_id_count = len(actions) - len({row["id"] for row in actions})
    duplicate_source_count = len(pairs) - len(
        {row["source_text_en"].strip().lower() for row in pairs}
    )
    missing_action_parent_count = sum(row["parent_id"] not in pair_by_id for row in actions)
    parent_without_actions_count = sum(row["id"] not in actions_by_parent for row in pairs)

    reconstruction_error_count = 0
    action_sequence_mismatch_count = 0
    end_count_error_count = 0
    noncausal_write_count = 0
    nonmonotonic_prefix_count = 0
    conflicting_state_count = 0
    full_source_wait_count = 0
    invalid_action_type_count = 0
    allowed_types = {"READ_WAIT", "WRITE_GLOSS", "WRITE_AFTER_SOURCE_END", "END"}
    for parent_id, pair in pair_by_id.items():
        parent_actions = sorted(actions_by_parent[parent_id], key=lambda row: int(row["id"].rsplit("_", 1)[-1]))
        target_tokens = pair["target_pseudo_asl_gloss_tokens"]
        written_tokens = [
            row["target_action"]
            for row in parent_actions
            if row["target_action_type"].startswith("WRITE")
        ]
        if written_tokens != target_tokens:
            reconstruction_error_count += 1
        if [row["target_action"] for row in parent_actions] != pair["causal_action_sequence"]:
            action_sequence_mismatch_count += 1
        if sum(row["target_action_type"] == "END" for row in parent_actions) != 1:
            end_count_error_count += 1
        previous_prefix = -1
        targets_by_state: dict[str, set[str]] = defaultdict(set)
        for row in parent_actions:
            invalid_action_type_count += int(row["target_action_type"] not in allowed_types)
            if row["source_prefix_length"] < previous_prefix:
                nonmonotonic_prefix_count += 1
            previous_prefix = row["source_prefix_length"]
            if row["target_action_type"].startswith("WRITE"):
                required = row.get("required_source_position")
                if required is None or int(required) > int(row["source_prefix_length"]):
                    noncausal_write_count += 1
            if row["target_action_type"] == "READ_WAIT" and row["source_complete"]:
                full_source_wait_count += 1
            targets_by_state[row["state_prompt"]].add(row["target_action"])
        conflicting_state_count += sum(len(targets) > 1 for targets in targets_by_state.values())

    videos_by_split = {
        split: {
            video_id
            for row in pairs
            if row["split"] == split
            for video_id in row.get("how2sign_video_ids", [])
        }
        for split in ("train", "validation", "test")
    }
    video_overlaps = {
        "train_validation": len(videos_by_split["train"] & videos_by_split["validation"]),
        "train_test": len(videos_by_split["train"] & videos_by_split["test"]),
        "validation_test": len(videos_by_split["validation"] & videos_by_split["test"]),
    }
    pair_split_counts = Counter(row["split"] for row in pairs)
    action_split_counts = Counter(row["split"] for row in actions)
    expected_split_counts = Counter(
        {key: int(value) for key, value in config["final_split_parent_counts"].items()}
    )
    split_manifest_mismatch_counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        pair_manifest = read_jsonl(args.root / "manifests" / f"{split}_pairs.jsonl")
        action_manifest = read_jsonl(args.root / "manifests" / f"{split}_actions.jsonl")
        split_manifest_mismatch_counts[f"{split}_pairs"] = abs(
            len(pair_manifest) - pair_split_counts[split]
        )
        split_manifest_mismatch_counts[f"{split}_actions"] = abs(
            len(action_manifest) - action_split_counts[split]
        )

    all_non_gold = (
        all(row.get("target_pseudo_gloss_is_gold") is False for row in pairs)
        and all(row.get("target_is_gold") is False for row in actions)
        and all(row.get("alignment_is_gold") is False for row in actions)
        and all(row.get("target_is_gold") is False for row in parent_splits)
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": config["dataset_version"],
        "scientific_status": config["scientific_status"],
        "timeline_policy": config["timeline_policy"],
        "parent_count": len(pairs),
        "action_count": len(actions),
        "parent_split_counts": dict(pair_split_counts),
        "action_split_counts": dict(action_split_counts),
        "parent_split_manifest_count": len(parent_splits),
        "duplicate_parent_id_count": duplicate_parent_id_count,
        "duplicate_action_id_count": duplicate_action_id_count,
        "duplicate_normalized_source_count": duplicate_source_count,
        "missing_action_parent_count": missing_action_parent_count,
        "parent_without_actions_count": parent_without_actions_count,
        "reconstruction_error_count": reconstruction_error_count,
        "action_sequence_mismatch_count": action_sequence_mismatch_count,
        "end_count_error_count": end_count_error_count,
        "noncausal_write_count": noncausal_write_count,
        "nonmonotonic_prefix_count": nonmonotonic_prefix_count,
        "conflicting_identical_state_target_count": conflicting_state_count,
        "full_source_wait_count": full_source_wait_count,
        "invalid_action_type_count": invalid_action_type_count,
        "source_video_overlap_counts": video_overlaps,
        "split_manifest_mismatch_counts": split_manifest_mismatch_counts,
        "all_pseudo_targets_and_alignments_marked_non_gold": all_non_gold,
        "qualified_asl_review_complete": False,
        "ok": (
            len(pairs) == int(config["final_parent_count"])
            and pair_split_counts == expected_split_counts
            and len(parent_splits) == len(pairs)
            and duplicate_parent_id_count == 0
            and duplicate_action_id_count == 0
            and duplicate_source_count == 0
            and missing_action_parent_count == 0
            and parent_without_actions_count == 0
            and reconstruction_error_count == 0
            and action_sequence_mismatch_count == 0
            and end_count_error_count == 0
            and noncausal_write_count == 0
            and nonmonotonic_prefix_count == 0
            and conflicting_state_count == 0
            and full_source_wait_count == 0
            and invalid_action_type_count == 0
            and all(value == 0 for value in video_overlaps.values())
            and all(value == 0 for value in split_manifest_mismatch_counts.values())
            and all_non_gold
        ),
    }
    write_json(args.report, report)

    core_paths = [
        args.config,
        args.root / "scripts" / "prepare_candidates.py",
        args.root / "scripts" / "build_causal_alignment_shard.py",
        args.root / "scripts" / "finalize_dataset.py",
        args.root / "scripts" / "validate_dataset.py",
        args.root / "scripts" / "run_corrected_causal_alignment_5k.sh",
        args.root / "manifests" / "candidate_source_6200.jsonl",
        args.root / "manifests" / "parent_splits_5000.jsonl",
        args.root / "manifests" / "train_actions.jsonl",
        args.root / "manifests" / "validation_actions.jsonl",
        args.root / "manifests" / "test_actions.jsonl",
        pairs_path,
        actions_path,
        args.root / "reviews" / "causal_alignment_review_50.tsv",
        args.root / "reports" / "candidate_selection_report.json",
        args.root / "reports" / "finalization_report.json",
        args.report,
    ]
    missing_core_files = [str(path) for path in core_paths if not path.is_file()]
    hashes = {
        str(path.relative_to(args.root)).replace("\\", "/"): sha256_file(path)
        for path in core_paths
        if path.is_file()
    }
    hash_report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "SHA-256",
        "root": str(args.root),
        "file_count": len(hashes),
        "missing_core_files": missing_core_files,
        "files": hashes,
        "ok": not missing_core_files,
    }
    write_json(args.hash_report, hash_report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(hash_report, indent=2, ensure_ascii=False))
    if not report["ok"] or not hash_report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
