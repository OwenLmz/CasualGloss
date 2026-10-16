#!/usr/bin/env python3
"""Prepare complete How2Sign train records and unique text-generation shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "VIDEO_ID",
    "VIDEO_NAME",
    "SENTENCE_ID",
    "SENTENCE_NAME",
    "START_REALIGNED",
    "END_REALIGNED",
    "SENTENCE",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{namespace}|{value}".encode("utf-8")).hexdigest()


def normalize_english(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--all-records-jsonl", type=Path, required=True)
    parser.add_argument("--unique-requests-jsonl", type=Path, required=True)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--expected-source-sha256", required=True)
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise SystemExit("--num-shards must be positive")
    source_sha256 = file_sha256(args.source_csv)
    if source_sha256 != args.expected_source_sha256:
        raise SystemExit(
            f"Source SHA-256 mismatch: expected {args.expected_source_sha256}, got {source_sha256}"
        )

    valid_rows: list[dict[str, Any]] = []
    invalid_counts: Counter[str] = Counter()
    with args.source_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing source columns: {sorted(missing)}")
        for source_row_index, source in enumerate(reader, start=1):
            sentence = re.sub(r"\s+", " ", str(source["SENTENCE"])).strip()
            if not sentence:
                invalid_counts["empty_sentence"] += 1
                continue
            try:
                start = float(source["START_REALIGNED"])
                end = float(source["END_REALIGNED"])
            except ValueError:
                invalid_counts["invalid_timestamp"] += 1
                continue
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                invalid_counts["invalid_time_interval"] += 1
                continue
            normalized_key = normalize_english(sentence)
            valid_rows.append(
                {
                    "source_row_index": source_row_index,
                    "how2sign_video_id": source["VIDEO_ID"],
                    "how2sign_video_name": source["VIDEO_NAME"],
                    "how2sign_sentence_id": source["SENTENCE_ID"],
                    "how2sign_sentence_name": source["SENTENCE_NAME"],
                    "start_realigned_sec": start,
                    "end_realigned_sec": end,
                    "duration_sec": round(end - start, 6),
                    "source_text_en": sentence,
                    "normalized_english_key": normalized_key,
                    "normalized_english_sha256": hashlib.sha256(
                        normalized_key.encode("utf-8")
                    ).hexdigest(),
                }
            )

    by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid_rows:
        by_text[row["normalized_english_key"]].append(row)

    ordered_keys = sorted(
        by_text,
        key=lambda key: stable_rank(args.seed, "full_unique_english", key),
    )
    request_index_by_key = {key: index for index, key in enumerate(ordered_keys)}
    request_id_by_key = {
        key: f"qwen_v4_full_request_{index:06d}"
        for key, index in request_index_by_key.items()
    }

    unique_requests: list[dict[str, Any]] = []
    for normalized_key in ordered_keys:
        group = by_text[normalized_key]
        canonical = min(
            group,
            key=lambda row: stable_rank(
                args.seed, "full_canonical", str(row["how2sign_sentence_name"])
            ),
        )
        request_index = request_index_by_key[normalized_key]
        request_id = request_id_by_key[normalized_key]
        unique_requests.append(
            {
                "request_id": request_id,
                "request_index": request_index,
                "source_text_en": canonical["source_text_en"],
                "normalized_english_key": normalized_key,
                "normalized_english_sha256": canonical["normalized_english_sha256"],
                "generation_input_sha256": hashlib.sha256(
                    canonical["source_text_en"].encode("utf-8")
                ).hexdigest(),
                "generation_context_sentences": 0,
                "canonical_how2sign_sentence_name": canonical[
                    "how2sign_sentence_name"
                ],
                "canonical_source_row_index": canonical["source_row_index"],
                "source_record_count_for_normalized_english": len(group),
                "source_sentence_names_preview": sorted(
                    str(row["how2sign_sentence_name"]) for row in group
                )[:10],
                "source_dataset": "How2Sign English Translation manually realigned",
                "source_csv_sha256": source_sha256,
                "selection_seed": args.seed,
                "generation_shard_id": request_index % args.num_shards,
            }
        )

    all_records: list[dict[str, Any]] = []
    for sample_index, source in enumerate(valid_rows):
        normalized_key = source["normalized_english_key"]
        request_index = request_index_by_key[normalized_key]
        request_id = request_id_by_key[normalized_key]
        duplicate_count = len(by_text[normalized_key])
        all_records.append(
            {
                "id": (
                    f"how2sign_train_full_v4_{sample_index:06d}_"
                    f"{source['how2sign_sentence_name']}"
                ),
                "sample_index": sample_index,
                "how2sign_split": "train",
                **source,
                "request_id": request_id,
                "request_index": request_index,
                "source_record_count_for_normalized_english": duplicate_count,
                "gloss_generation_reused_for_duplicate_english": duplicate_count > 1,
                "source_dataset": "How2Sign English Translation manually realigned",
                "source_csv_sha256": source_sha256,
                "dataset_construction_version": "full_v4_train",
            }
        )

    write_jsonl(args.all_records_jsonl, all_records)
    write_jsonl(args.unique_requests_jsonl, unique_requests)
    shard_rows = [[] for _ in range(args.num_shards)]
    for request in unique_requests:
        shard_rows[int(request["generation_shard_id"])].append(request)
    shard_reports: list[dict[str, Any]] = []
    for shard_id, rows in enumerate(shard_rows):
        shard_path = args.shards_dir / f"unique_requests.part{shard_id:03d}.jsonl"
        write_jsonl(shard_path, rows)
        shard_reports.append(
            {
                "shard_id": shard_id,
                "record_count": len(rows),
                "path": str(shard_path),
                "sha256": file_sha256(shard_path),
            }
        )

    unique_ids = {row["id"] for row in all_records}
    unique_request_ids = {row["request_id"] for row in unique_requests}
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(args.source_csv),
        "source_csv_sha256": source_sha256,
        "source_split": "train",
        "source_row_count": len(valid_rows) + sum(invalid_counts.values()),
        "valid_source_row_count": len(valid_rows),
        "invalid_source_row_counts": dict(invalid_counts),
        "source_video_count": len({row["how2sign_video_id"] for row in valid_rows}),
        "unique_normalized_english_count": len(ordered_keys),
        "duplicate_english_group_count": sum(
            len(group) > 1 for group in by_text.values()
        ),
        "duplicate_rows_beyond_unique_count": sum(
            len(group) - 1 for group in by_text.values()
        ),
        "all_records_count": len(all_records),
        "all_records_unique_id_count": len(unique_ids),
        "unique_generation_request_count": len(unique_requests),
        "unique_generation_request_id_count": len(unique_request_ids),
        "duration_sec": summarize([float(row["duration_sec"]) for row in all_records]),
        "num_generation_shards": args.num_shards,
        "shards": shard_reports,
        "all_records_jsonl": str(args.all_records_jsonl),
        "all_records_sha256": file_sha256(args.all_records_jsonl),
        "unique_requests_jsonl": str(args.unique_requests_jsonl),
        "unique_requests_sha256": file_sha256(args.unique_requests_jsonl),
        "seed": args.seed,
        "pipeline_ok": all(
            [
                len(all_records) == len(valid_rows),
                len(unique_ids) == len(all_records),
                len(unique_requests) == len(ordered_keys),
                len(unique_request_ids) == len(unique_requests),
                sum(len(rows) for rows in shard_rows) == len(unique_requests),
                set(row["request_id"] for row in all_records) == unique_request_ids,
                all(row["how2sign_split"] == "train" for row in all_records),
            ]
        ),
        "scientific_status": (
            "complete_official_text_manifest_pending_pseudo_gloss_generation"
        ),
        "interpretation": (
            "Every valid official How2Sign train sentence is retained. Identical "
            "normalized English sentences share one deterministic generation request; "
            "this saves inference without claiming that the generated gloss is gold ASL."
        ),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pipeline_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
