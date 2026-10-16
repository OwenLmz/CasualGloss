#!/usr/bin/env python3
"""Filter full alignments while preserving the locked 5K evaluation protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def stable_hash(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def max_hyphen_components(tokens: list[str]) -> int:
    return max((len(token.split("-")) for token in tokens), default=0)


def exclusion_reasons(pair: dict[str, Any], config: dict[str, Any]) -> list[str]:
    policy = config["structural_filter"]
    flags = set(pair.get("causal_alignment_quality_flags") or [])
    reasons: list[str] = []
    if policy["exclude_high_unaligned_ratio"] and "high_unaligned_gloss_ratio" in flags:
        reasons.append("high_unaligned_gloss_ratio")
    if policy["exclude_low_lexical_alignment_ratio"] and "low_lexical_alignment_ratio" in flags:
        reasons.append("low_lexical_alignment_ratio")
    if policy["exclude_simalign_runtime_error"] and "simalign_runtime_error" in flags:
        reasons.append("simalign_runtime_error")
    maximum = int(policy["maximum_hyphen_components_per_gloss_token"])
    if max_hyphen_components(pair["target_pseudo_asl_gloss_tokens"]) > maximum:
        reasons.append("overjoined_multiword_gloss_token")
    return reasons


def state_prompt(action: dict[str, Any]) -> str:
    committed = action["committed_pseudo_asl_gloss"] or "<EMPTY>"
    status = "COMPLETE" if action["source_complete"] else "INCOMPLETE"
    return (
        "Observed English prefix:\n"
        f"{action['partial_source_text_en']}\n\n"
        "Already committed pseudo ASL gloss:\n"
        f"{committed}\n\n"
        f"Source status: {status}\n"
        "Return exactly one next action."
    )


def safe_mean(values: list[float]) -> float | None:
    return round(mean(values), 6) if values else None


def choose_review_rows(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def take(candidates: list[dict[str, Any]], number: int) -> None:
        for row in candidates:
            if number <= 0:
                break
            if row["id"] not in selected_ids:
                selected.append(row)
                selected_ids.add(row["id"])
                number -= 1

    take(sorted(rows, key=lambda row: (-row["metrics"]["unaligned_gloss_token_ratio"], row["id"])), 20)
    take(sorted(rows, key=lambda row: (-row["metrics"]["wait_action_ratio"], row["id"])), 20)
    take(
        sorted(
            [row for row in rows if row["candidate_selection_bucket"] == "automatic_review_flagged"],
            key=lambda row: stable_hash(seed, row["id"]),
        ),
        20,
    )
    take(sorted(rows, key=lambda row: stable_hash(seed, row["id"])), count - len(selected))
    return selected[:count]


def write_review(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "id",
        "split",
        "locked_evaluation_parent",
        "candidate_selection_bucket",
        "source_text_en",
        "target_pseudo_asl_gloss",
        "source_token_count",
        "gloss_token_count",
        "lexical_alignment_ratio",
        "simalign_fallback_ratio",
        "unaligned_gloss_token_ratio",
        "wait_action_ratio",
        "causal_action_sequence",
        "quality_flags",
        "structural_review_status",
        "asl_linguistic_review_status",
        "review_notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "split": row["split"],
                    "locked_evaluation_parent": row["locked_evaluation_parent"],
                    "candidate_selection_bucket": row["candidate_selection_bucket"],
                    "source_text_en": row["source_text_en"],
                    "target_pseudo_asl_gloss": row["target_pseudo_asl_gloss"],
                    "source_token_count": row["metrics"]["source_token_count"],
                    "gloss_token_count": row["metrics"]["gloss_token_count"],
                    "lexical_alignment_ratio": row["metrics"]["lexical_alignment_ratio"],
                    "simalign_fallback_ratio": row["metrics"]["simalign_fallback_ratio"],
                    "unaligned_gloss_token_ratio": row["metrics"]["unaligned_gloss_token_ratio"],
                    "wait_action_ratio": row["metrics"]["wait_action_ratio"],
                    "causal_action_sequence": " ".join(row["causal_action_sequence"]),
                    "quality_flags": ",".join(row["causal_alignment_quality_flags"]),
                    "structural_review_status": "passed_automatic_filter",
                    "asl_linguistic_review_status": "pending_qualified_reviewer",
                    "review_notes": "",
                }
            )


def compact_quarantine_row(pair: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    return {
        "id": pair["id"],
        "source_text_en": pair["source_text_en"],
        "target_pseudo_asl_gloss": pair["target_pseudo_asl_gloss"],
        "how2sign_video_ids": pair.get("how2sign_video_ids", []),
        "quarantine_reasons": reasons,
        "target_is_gold": False,
        "alignment_is_gold": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--locked-parent-splits", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    seed = int(config["selection_seed"])
    pair_files = sorted(args.shard_dir.glob("pairs_shard_*.jsonl"))
    action_files = sorted(args.shard_dir.glob("actions_shard_*.jsonl"))
    expected_shards = int(config["shard_count"])
    if len(pair_files) != expected_shards or len(action_files) != expected_shards:
        raise SystemExit(
            f"Expected {expected_shards} pair/action shards, got {len(pair_files)}/{len(action_files)}"
        )

    pairs = [row for path in pair_files for row in read_jsonl(path)]
    actions = [row for path in action_files for row in read_jsonl(path)]
    if len({row["id"] for row in pairs}) != len(pairs):
        raise SystemExit("Duplicate pair IDs after shard merge")

    excluded_rows: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for pair in pairs:
        reasons = exclusion_reasons(pair, config)
        if reasons:
            excluded_rows.append({"id": pair["id"], "reasons": reasons})
        else:
            eligible.append(pair)
    eligible_by_id = {row["id"]: row for row in eligible}

    locked_rows = [
        row
        for row in read_jsonl(args.locked_parent_splits)
        if row.get("split") in {"validation", "test"}
    ]
    locked_by_split = {
        split: {str(row["id"]): row for row in locked_rows if row["split"] == split}
        for split in ("validation", "test")
    }
    locked_ids = set(locked_by_split["validation"]) | set(locked_by_split["test"])
    missing_locked_ids = sorted(locked_ids - set(eligible_by_id))
    target_mismatch_ids = sorted(
        row_id
        for split in ("validation", "test")
        for row_id, locked in locked_by_split[split].items()
        if row_id in eligible_by_id
        and eligible_by_id[row_id]["target_pseudo_asl_gloss"] != locked["target_pseudo_asl_gloss"]
    )
    if missing_locked_ids or target_mismatch_ids:
        raise SystemExit(
            f"Locked evaluation mismatch: missing={len(missing_locked_ids)}, "
            f"target_mismatch={len(target_mismatch_ids)}"
        )

    locked_videos = {
        split: {
            video_id
            for row_id in locked_by_split[split]
            for video_id in eligible_by_id[row_id].get("how2sign_video_ids", [])
        }
        for split in ("validation", "test")
    }
    if locked_videos["validation"] & locked_videos["test"]:
        raise SystemExit("Locked validation/test sets share source video IDs")
    all_heldout_videos = locked_videos["validation"] | locked_videos["test"]

    selected_by_split: dict[str, list[dict[str, Any]]] = {
        "validation": [eligible_by_id[row_id] for row_id in locked_by_split["validation"]],
        "test": [eligible_by_id[row_id] for row_id in locked_by_split["test"]],
        "train": [],
    }
    quarantined: list[dict[str, Any]] = []
    for pair in eligible:
        if pair["id"] in locked_ids:
            continue
        videos = set(pair.get("how2sign_video_ids", []))
        overlap_validation = videos & locked_videos["validation"]
        overlap_test = videos & locked_videos["test"]
        if overlap_validation or overlap_test:
            reasons: list[str] = []
            if overlap_validation:
                reasons.append("shares_locked_validation_video")
            if overlap_test:
                reasons.append("shares_locked_test_video")
            quarantined.append(compact_quarantine_row(pair, reasons))
        else:
            selected_by_split["train"].append(pair)

    expected_locked = config["locked_evaluation_parent_counts"]
    if any(
        len(selected_by_split[split]) != int(expected_locked[split])
        for split in ("validation", "test")
    ):
        raise SystemExit("Locked evaluation parent counts changed")
    if len(selected_by_split["train"]) < int(config["minimum_expected_train_parent_count"]):
        raise SystemExit("Too few full-training parents after held-out-video quarantine")

    selected_pairs: list[dict[str, Any]] = []
    split_by_parent: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        rows = sorted(selected_by_split[split], key=lambda row: stable_hash(seed, row["id"]))
        for split_index, pair in enumerate(rows):
            pair["split"] = split
            pair["split_parent_index"] = split_index
            pair["eligible_for_finetune2"] = True
            pair["locked_evaluation_parent"] = split in {"validation", "test"}
            pair["structural_exclusion_reasons"] = []
            split_by_parent[pair["id"]] = split
            selected_pairs.append(pair)
    selected_pairs.sort(key=lambda row: (row["split"], row["split_parent_index"]))
    selected_ids = set(split_by_parent)

    action_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for action in actions:
        if action["parent_id"] in selected_ids:
            action_by_parent[action["parent_id"]].append(action)

    final_actions_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in selected_pairs:
        split = pair["split"]
        parent_actions = sorted(action_by_parent[pair["id"]], key=lambda row: row["action_index"])
        for action in parent_actions:
            final_actions_by_split[split].append(
                {
                    "id": action["example_id"],
                    "parent_id": pair["id"],
                    "split": split,
                    "state_prompt": state_prompt(action),
                    "target_action": action["target_action"],
                    "target_action_type": action["action_type"],
                    "source_prefix_length": action["source_prefix_length"],
                    "source_token_count": action["source_token_count"],
                    "source_complete": action["source_complete"],
                    "required_source_position": action["required_source_position"],
                    "partial_source_text_en": action["partial_source_text_en"],
                    "committed_pseudo_asl_gloss": action["committed_pseudo_asl_gloss"],
                    "parent_target_pseudo_asl_gloss": pair["target_pseudo_asl_gloss"],
                    "candidate_selection_bucket": pair["candidate_selection_bucket"],
                    "candidate_source_length_bin": pair["candidate_source_length_bin"],
                    "parent_quality_flags": pair["causal_alignment_quality_flags"],
                    "how2sign_video_ids": pair.get("how2sign_video_ids", []),
                    "locked_evaluation_parent": pair["locked_evaluation_parent"],
                    "target_is_gold": False,
                    "alignment_is_gold": False,
                    "timeline_policy": config["timeline_policy"],
                }
            )

    for split in ("train", "validation", "test"):
        final_actions_by_split[split].sort(key=lambda row: stable_hash(seed, row["id"]))
        write_jsonl(args.manifest_dir / f"{split}_actions.jsonl", final_actions_by_split[split])
        write_jsonl(
            args.manifest_dir / f"{split}_pairs.jsonl",
            [row for row in selected_pairs if row["split"] == split],
        )

    pairs_path = args.output_dir / "causal_aligned_pairs_full.jsonl"
    actions_path = args.output_dir / "prefix_action_examples_corrected_full.jsonl"
    write_jsonl(pairs_path, selected_pairs)
    write_jsonl(
        actions_path,
        [
            row
            for split in ("train", "validation", "test")
            for row in final_actions_by_split[split]
        ],
    )
    write_jsonl(args.output_dir / "structurally_excluded_pairs.jsonl", excluded_rows)
    write_jsonl(args.output_dir / "heldout_video_quarantine_pairs.jsonl", quarantined)

    parent_manifest = [
        {
            "id": row["id"],
            "split": row["split"],
            "source_text_en": row["source_text_en"],
            "target_pseudo_asl_gloss": row["target_pseudo_asl_gloss"],
            "candidate_selection_bucket": row["candidate_selection_bucket"],
            "candidate_source_length_bin": row["candidate_source_length_bin"],
            "how2sign_video_ids": row.get("how2sign_video_ids", []),
            "locked_evaluation_parent": row["locked_evaluation_parent"],
            "target_is_gold": False,
            "alignment_is_gold": False,
        }
        for row in selected_pairs
    ]
    write_jsonl(args.manifest_dir / "parent_splits_full.jsonl", parent_manifest)

    preview_rows = sorted(selected_pairs, key=lambda row: stable_hash(seed, row["id"]))[:100]
    write_review(args.preview, preview_rows)
    review_rows = choose_review_rows(selected_pairs, int(config["manual_review_size"]), seed)
    write_review(args.review, review_rows)

    videos_by_split = {
        split: {
            video_id
            for row in selected_pairs
            if row["split"] == split
            for video_id in row.get("how2sign_video_ids", [])
        }
        for split in ("train", "validation", "test")
    }
    overlaps = {
        "train_validation": len(videos_by_split["train"] & videos_by_split["validation"]),
        "train_test": len(videos_by_split["train"] & videos_by_split["test"]),
        "validation_test": len(videos_by_split["validation"] & videos_by_split["test"]),
    }
    metrics = [row["metrics"] for row in selected_pairs]
    locked_ids_output = {
        split: {row["id"] for row in selected_pairs if row["split"] == split}
        for split in ("validation", "test")
    }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": config["dataset_version"],
        "scientific_status": config["scientific_status"],
        "timeline_policy": config["timeline_policy"],
        "locked_evaluation_source": str(args.locked_parent_splits),
        "input_shard_pair_counts": {path.name: len(read_jsonl(path)) for path in pair_files},
        "merged_pair_count": len(pairs),
        "merged_action_count": len(actions),
        "eligible_after_alignment_filter_count": len(eligible),
        "structurally_excluded_parent_count": len(excluded_rows),
        "structural_exclusion_reason_counts": dict(
            Counter(reason for row in excluded_rows for reason in row["reasons"])
        ),
        "heldout_video_quarantine_parent_count": len(quarantined),
        "heldout_video_quarantine_reason_counts": dict(
            Counter(reason for row in quarantined for reason in row["quarantine_reasons"])
        ),
        "final_parent_count": len(selected_pairs),
        "parent_split_counts": dict(Counter(row["split"] for row in selected_pairs)),
        "action_split_counts": {
            split: len(final_actions_by_split[split])
            for split in ("train", "validation", "test")
        },
        "action_type_counts": {
            split: dict(Counter(row["target_action_type"] for row in final_actions_by_split[split]))
            for split in ("train", "validation", "test")
        },
        "locked_parent_counts": {
            split: len(locked_by_split[split]) for split in ("validation", "test")
        },
        "locked_parent_id_set_exact_match": {
            split: locked_ids_output[split] == set(locked_by_split[split])
            for split in ("validation", "test")
        },
        "missing_locked_parent_count": len(missing_locked_ids),
        "locked_target_mismatch_count": len(target_mismatch_ids),
        "selection_bucket_counts": dict(
            Counter(row["candidate_selection_bucket"] for row in selected_pairs)
        ),
        "source_length_bin_counts": dict(
            Counter(row["candidate_source_length_bin"] for row in selected_pairs)
        ),
        "unique_video_counts_by_split": {
            split: len(videos) for split, videos in videos_by_split.items()
        },
        "source_video_overlap_counts": overlaps,
        "mean_lexical_alignment_ratio": safe_mean(
            [item["lexical_alignment_ratio"] for item in metrics]
        ),
        "mean_simalign_fallback_ratio": safe_mean(
            [item["simalign_fallback_ratio"] for item in metrics]
        ),
        "mean_unaligned_gloss_token_ratio": safe_mean(
            [item["unaligned_gloss_token_ratio"] for item in metrics]
        ),
        "mean_wait_action_ratio": safe_mean([item["wait_action_ratio"] for item in metrics]),
        "median_wait_action_ratio": round(median(item["wait_action_ratio"] for item in metrics), 6),
        "manual_review_sample_count": len(review_rows),
        "all_targets_marked_non_gold": all(
            row["target_pseudo_gloss_is_gold"] is False for row in selected_pairs
        ),
        "important_limitations": [
            "Target glosses are fixed-checkpoint LLM pseudo labels, not gold ASL annotations.",
            "Token timing is lexical/SimAlign-derived pseudo alignment, not gold sign timing.",
            "Held-out validation/test parents are locked from the controlled 5K experiment.",
            "Rows sharing held-out source videos are quarantined rather than used for training.",
            "The 100-row review sheet is pending qualified ASL linguistic review.",
        ],
        "ok": (
            len(selected_by_split["train"]) >= int(config["minimum_expected_train_parent_count"])
            and all(
                len(selected_by_split[split]) == int(expected_locked[split])
                for split in ("validation", "test")
            )
            and not missing_locked_ids
            and not target_mismatch_ids
            and all(locked_ids_output[split] == set(locked_by_split[split]) for split in ("validation", "test"))
            and all(value == 0 for value in overlaps.values())
            and all(row["target_pseudo_gloss_is_gold"] is False for row in selected_pairs)
        ),
    }
    write_json(args.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
