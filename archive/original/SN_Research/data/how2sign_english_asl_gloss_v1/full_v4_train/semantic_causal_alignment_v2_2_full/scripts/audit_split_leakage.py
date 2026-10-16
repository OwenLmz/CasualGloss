#!/usr/bin/env python3
"""Audit parent, video-group and exact-source leakage across data splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    parent_splits: dict[str, set[str]] = defaultdict(set)
    video_splits: dict[str, set[str]] = defaultdict(set)
    source_splits: dict[str, set[str]] = defaultdict(set)
    split_counts = Counter()
    missing_video_ids = []
    row_count = 0
    for row in read_jsonl(args.pairs):
        row_count += 1
        split = row.get("split", "train")
        split_counts[split] += 1
        parent_splits[row["id"]].add(split)
        video_ids = row.get("how2sign_video_ids", [])
        if not video_ids:
            missing_video_ids.append(row["id"])
        for video_id in video_ids:
            video_splits[str(video_id)].add(split)
        source_splits[normalize_text(row["source_text_en"])].add(split)

    parent_overlap = {
        key: sorted(value) for key, value in parent_splits.items() if len(value) > 1
    }
    video_overlap = {
        key: sorted(value) for key, value in video_splits.items() if len(value) > 1
    }
    exact_source_overlap = {
        key: sorted(value) for key, value in source_splits.items() if len(value) > 1
    }
    report: dict[str, Any] = {
        "scientific_status": "split_leakage_audit",
        "pairs": str(args.pairs),
        "row_count": row_count,
        "unique_parent_count": len(parent_splits),
        "unique_video_group_count": len(video_splits),
        "split_counts": dict(split_counts),
        "duplicate_parent_id_count": row_count - len(parent_splits),
        "parent_cross_split_overlap_count": len(parent_overlap),
        "parent_cross_split_overlap_preview": dict(list(parent_overlap.items())[:20]),
        "video_group_cross_split_overlap_count": len(video_overlap),
        "video_group_cross_split_overlap_preview": dict(list(video_overlap.items())[:20]),
        "exact_source_cross_split_overlap_count": len(exact_source_overlap),
        "exact_source_cross_split_overlap_preview": dict(list(exact_source_overlap.items())[:20]),
        "missing_video_id_count": len(missing_video_ids),
        "missing_video_id_preview": missing_video_ids[:20],
        "strict_video_group_disjoint": not video_overlap,
        "ok": not parent_overlap and not video_overlap and row_count == len(parent_splits),
        "note": (
            "Exact source duplication is reported separately because identical generic sentences may "
            "legitimately occur in different videos; video-group overlap is the strict leakage criterion."
        ),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
