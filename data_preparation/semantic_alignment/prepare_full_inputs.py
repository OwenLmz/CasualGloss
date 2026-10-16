#!/usr/bin/env python3
"""Prefilter and deterministically shard locked train/validation/test parents."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("alignment_v2_1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=2)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    alignment = load_module(args.finalizer)
    rows = []
    for split, path in (("train", args.train), ("validation", args.validation), ("test", args.test)):
        for row in read_jsonl(path):
            item = dict(row)
            item["split"] = split
            rows.append(item)

    retained = []
    excluded = []
    for global_index, row in enumerate(rows):
        anchors, order = alignment.occurrence_aware_anchors(
            row["target_pseudo_asl_gloss_tokens"], row["source_tokens"]
        )
        reasons = []
        if alignment.source_is_corrupt(row["source_text_en"]):
            reasons.append("corrupt_or_non_english_source_text")
        if (
            order["lexically_matchable_token_count"] >= int(config["noncausal_min_lexical_token_count"])
            and order["monotonic_coverage"] < float(config["noncausal_min_monotonic_coverage"])
            and (
                order["backward_jump_count"] > int(config["noncausal_max_backward_jump_count"])
                or order["backward_distance"] > int(config["noncausal_max_backward_distance"])
            )
        ):
            reasons.append("strong_streaming_noncausal_gloss_order")
        item = dict(row)
        item["semantic_v2_pilot_index"] = global_index
        item["semantic_v2_full_index"] = global_index
        item["semantic_v2_1_prefilter_order_diagnostics"] = order
        if reasons:
            item["v2_1_prefilter_exclusion_reasons"] = reasons
            excluded.append(item)
        else:
            retained.append(item)

    weighted = sorted(
        retained,
        key=lambda row: (-(len(row["source_tokens"]) + 1) * len(row["target_pseudo_asl_gloss_tokens"]), row["id"]),
    )
    shards = [[] for _ in range(args.shard_count)]
    loads = [0 for _ in range(args.shard_count)]
    for row in weighted:
        shard_id = min(range(args.shard_count), key=lambda value: (loads[value], value))
        item = dict(row)
        item["semantic_v2_shard_id"] = shard_id
        shards[shard_id].append(item)
        loads[shard_id] += (len(item["source_tokens"]) + 1) * len(item["target_pseudo_asl_gloss_tokens"])
    retained_ordered = sorted((row for shard in shards for row in shard), key=lambda row: row["semantic_v2_full_index"])

    write_jsonl(args.output_root / "manifests" / "parents_all_prefiltered.jsonl", retained_ordered)
    write_jsonl(args.output_root / "manifests" / "parents_preexcluded.jsonl", excluded)
    for shard_id, shard in enumerate(shards):
        write_jsonl(args.output_root / "shards" / f"parents_shard_{shard_id:02d}.jsonl", shard)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_parent_count": len(rows),
        "input_split_counts": dict(Counter(row["split"] for row in rows)),
        "retained_parent_count_for_semantic_scoring": len(retained),
        "retained_split_counts": dict(Counter(row["split"] for row in retained)),
        "preexcluded_parent_count": len(excluded),
        "preexcluded_reason_counts": dict(Counter(reason for row in excluded for reason in row["v2_1_prefilter_exclusion_reasons"])),
        "shard_parent_counts": [len(shard) for shard in shards],
        "shard_estimated_loads": loads,
        "shard_load_ratio_max_over_min": round(max(loads) / max(1, min(loads)), 6),
        "ok": len(rows) == len(retained) + len(excluded) and max(loads) / max(1, min(loads)) <= 1.01,
    }
    report_path = args.output_root / "reports" / "full_input_preparation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
