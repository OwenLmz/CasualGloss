#!/usr/bin/env python3
"""Select and shard a deterministic 6,200-parent alignment candidate pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    seed = int(config["selection_seed"])
    maximum_hyphen = int(config["maximum_hyphen_components_per_gloss_token"])
    per_video_cap = int(config["maximum_selected_rows_per_video"])
    targets = config["candidate_selection_targets"]
    rows = read_jsonl(args.input)

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
        eligible.append(row)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        grouped[(selection_bucket(row), source_length_bin(row))].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: stable_hash(seed, row_id(row)))

    selected: list[dict[str, Any]] = []
    selected_per_video: Counter[str] = Counter()
    stratum_available_counts: dict[str, int] = {}
    stratum_selected_counts: Counter[str] = Counter()
    for bucket, length_targets in targets.items():
        for length_name, requested in length_targets.items():
            stratum = f"{bucket}::{length_name}"
            candidates = grouped[(bucket, length_name)]
            stratum_available_counts[stratum] = len(candidates)
            for row in candidates:
                video_key = canonical_video_key(row)
                if selected_per_video[video_key] >= per_video_cap:
                    continue
                cleaned = {key: value for key, value in row.items() if key != "_input_line_number"}
                cleaned["candidate_selection_bucket"] = bucket
                cleaned["candidate_source_length_bin"] = length_name
                cleaned["candidate_source_word_count_whitespace"] = source_word_count(row)
                cleaned["candidate_video_group_key"] = video_key
                selected.append(cleaned)
                selected_per_video[video_key] += 1
                stratum_selected_counts[stratum] += 1
                if stratum_selected_counts[stratum] == int(requested):
                    break
            if stratum_selected_counts[stratum] != int(requested):
                raise SystemExit(
                    f"Could not satisfy {stratum}: requested {requested}, "
                    f"selected {stratum_selected_counts[stratum]} with per-video cap {per_video_cap}"
                )

    selected.sort(key=lambda row: stable_hash(seed, row_id(row)))
    for index, row in enumerate(selected):
        row["candidate_index"] = index
        row["causal_alignment_dataset_version"] = config["dataset_version"]
    if len(selected) != int(config["candidate_pool_size"]):
        raise SystemExit(
            f"Candidate size mismatch: {len(selected)} != {config['candidate_pool_size']}"
        )

    shard_count = int(config["shard_count"])
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    for index, row in enumerate(selected):
        shards[index % shard_count].append(row)
    write_jsonl(args.output, selected)
    args.shard_dir.mkdir(parents=True, exist_ok=True)
    for shard_index, shard_rows in enumerate(shards):
        write_jsonl(args.shard_dir / f"candidate_shard_{shard_index:02d}.jsonl", shard_rows)

    ids = [row_id(row) for row in selected]
    source_keys = [
        str(row.get("normalized_english_key") or row["source_text_en"]).strip().lower()
        for row in selected
    ]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": config["dataset_version"],
        "scientific_status": config["scientific_status"],
        "input_manifest": str(args.input),
        "input_record_count": len(rows),
        "eligible_after_prefilter_count": len(eligible),
        "candidate_record_count": len(selected),
        "selection_seed": seed,
        "selection_targets": targets,
        "stratum_available_counts": stratum_available_counts,
        "stratum_selected_counts": dict(stratum_selected_counts),
        "prefilter_reason_counts": dict(prefilter_reason_counts),
        "unique_parent_id_count": len(set(ids)),
        "unique_normalized_source_count": len(set(source_keys)),
        "unique_video_group_count": len(selected_per_video),
        "maximum_rows_from_one_video": max(selected_per_video.values(), default=0),
        "video_group_size_counts": dict(Counter(selected_per_video.values())),
        "shard_counts": [len(rows) for rows in shards],
        "all_targets_marked_non_gold": all(row.get("pseudo_gloss_is_gold") is False for row in selected),
        "ok": (
            len(selected) == int(config["candidate_pool_size"])
            and len(set(ids)) == len(ids)
            and len(set(source_keys)) == len(source_keys)
            and max(selected_per_video.values(), default=0) <= per_video_cap
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
