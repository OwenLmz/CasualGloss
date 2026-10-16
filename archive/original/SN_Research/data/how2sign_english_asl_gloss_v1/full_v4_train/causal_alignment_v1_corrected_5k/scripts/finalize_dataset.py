#!/usr/bin/env python3
"""Merge shards, filter alignment noise, and build video-disjoint 5K splits."""

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


def build_components(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_row_by_video: dict[str, int] = {}
    for index, row in enumerate(rows):
        video_ids = sorted(set(row.get("how2sign_video_ids") or []))
        for video_id in video_ids:
            if video_id in first_row_by_video:
                union(index, first_row_by_video[video_id])
            else:
                first_row_by_video[video_id] = index

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[find(index)].append(row)
    return list(grouped.values())


def component_key(component: list[dict[str, Any]]) -> str:
    video_ids = sorted(
        {video_id for row in component for video_id in row.get("how2sign_video_ids", [])}
    )
    if video_ids:
        return "VIDEO::" + "|".join(video_ids)
    return "NO_VIDEO::" + "|".join(sorted(row["id"] for row in component))


def allocate_split(
    components: list[list[dict[str, Any]]],
    count: int,
    split: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]], dict[str, int]]:
    ordered = sorted(
        components,
        key=lambda group: stable_hash(seed, f"{split}|{component_key(group)}"),
    )
    selected: list[dict[str, Any]] = []
    used_component_keys: set[str] = set()
    dropped_from_partial_component = 0
    for component in ordered:
        if len(selected) >= count:
            break
        key = component_key(component)
        rows = sorted(component, key=lambda row: stable_hash(seed, f"{split}|{row['id']}"))
        needed = count - len(selected)
        selected.extend(rows[:needed])
        used_component_keys.add(key)
        dropped_from_partial_component += max(0, len(rows) - needed)
    if len(selected) != count:
        raise SystemExit(f"Unable to allocate {count} rows to {split}; got {len(selected)}")
    remaining = [group for group in components if component_key(group) not in used_component_keys]
    return selected, remaining, {
        "used_component_count": len(used_component_keys),
        "dropped_from_partial_component_count": dropped_from_partial_component,
    }


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

    take(sorted(rows, key=lambda row: (-row["metrics"]["unaligned_gloss_token_ratio"], row["id"])), 10)
    take(sorted(rows, key=lambda row: (-row["metrics"]["wait_action_ratio"], row["id"])), 10)
    take(
        sorted(
            [row for row in rows if row["candidate_selection_bucket"] == "automatic_review_flagged"],
            key=lambda row: stable_hash(seed, row["id"]),
        ),
        10,
    )
    take(sorted(rows, key=lambda row: stable_hash(seed, row["id"])), count - len(selected))
    return selected[:count]


def write_review(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "id",
        "split",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, required=True)
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
    final_count = int(config["final_parent_count"])
    if len(eligible) < final_count:
        raise SystemExit(f"Only {len(eligible)} eligible parents; need {final_count}")

    components = build_components(eligible)
    split_targets = config["final_split_parent_counts"]
    selected_by_split: dict[str, list[dict[str, Any]]] = {}
    allocation_reports: dict[str, dict[str, int]] = {}
    remaining = components
    for split in ("test", "validation", "train"):
        selected, remaining, allocation = allocate_split(
            remaining, int(split_targets[split]), split, seed
        )
        selected_by_split[split] = selected
        allocation_reports[split] = allocation

    selected_pairs: list[dict[str, Any]] = []
    split_by_parent: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        rows = sorted(selected_by_split[split], key=lambda row: stable_hash(seed, row["id"]))
        for split_index, pair in enumerate(rows):
            pair["split"] = split
            pair["split_parent_index"] = split_index
            pair["eligible_for_finetune2"] = True
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
    write_jsonl(args.output_dir / "causal_aligned_pairs_5000.jsonl", selected_pairs)
    all_final_actions = [
        row
        for split in ("train", "validation", "test")
        for row in final_actions_by_split[split]
    ]
    write_jsonl(args.output_dir / "prefix_action_examples_corrected_5000.jsonl", all_final_actions)
    write_jsonl(
        args.manifest_dir / "parent_splits_5000.jsonl",
        [
            {
                "id": row["id"],
                "split": row["split"],
                "source_text_en": row["source_text_en"],
                "target_pseudo_asl_gloss": row["target_pseudo_asl_gloss"],
                "candidate_selection_bucket": row["candidate_selection_bucket"],
                "candidate_source_length_bin": row["candidate_source_length_bin"],
                "how2sign_video_ids": row.get("how2sign_video_ids", []),
                "target_is_gold": False,
                "alignment_is_gold": False,
            }
            for row in selected_pairs
        ],
    )

    preview_rows = sorted(selected_pairs, key=lambda row: stable_hash(seed, row["id"]))[:50]
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
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": config["dataset_version"],
        "scientific_status": config["scientific_status"],
        "timeline_policy": config["timeline_policy"],
        "input_shard_pair_counts": {path.name: len(read_jsonl(path)) for path in pair_files},
        "merged_pair_count": len(pairs),
        "merged_action_count": len(actions),
        "eligible_after_alignment_filter_count": len(eligible),
        "excluded_after_alignment_filter_count": len(excluded_rows),
        "exclusion_reason_counts": dict(
            Counter(reason for row in excluded_rows for reason in row["reasons"])
        ),
        "final_parent_count": len(selected_pairs),
        "unused_eligible_parent_count": len(eligible) - len(selected_pairs),
        "parent_split_counts": dict(Counter(row["split"] for row in selected_pairs)),
        "action_split_counts": {
            split: len(final_actions_by_split[split])
            for split in ("train", "validation", "test")
        },
        "action_type_counts": {
            split: dict(Counter(row["target_action_type"] for row in final_actions_by_split[split]))
            for split in ("train", "validation", "test")
        },
        "selection_bucket_counts": dict(
            Counter(row["candidate_selection_bucket"] for row in selected_pairs)
        ),
        "source_length_bin_counts": dict(
            Counter(row["candidate_source_length_bin"] for row in selected_pairs)
        ),
        "selection_bucket_by_split": {
            split: dict(
                Counter(
                    row["candidate_selection_bucket"]
                    for row in selected_pairs
                    if row["split"] == split
                )
            )
            for split in ("train", "validation", "test")
        },
        "video_component_count_before_selection": len(components),
        "video_allocation": allocation_reports,
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
            "The 50-row review sheet is pending qualified ASL linguistic review.",
        ],
        "ok": (
            len(selected_pairs) == final_count
            and Counter(row["split"] for row in selected_pairs)
            == Counter({key: int(value) for key, value in split_targets.items()})
            and all(value == 0 for value in overlaps.values())
        ),
    }
    write_json(args.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
