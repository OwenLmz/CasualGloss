#!/usr/bin/env python3
"""Audit currently written rows from resumable full-generation shards."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GLOSS_RE = re.compile(r"^[A-Z0-9][A-Z0-9'/-]*(?: [A-Z0-9][A-Z0-9'/-]*)*$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-output", type=Path, action="append", required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-batch-size", type=int, required=True)
    args = parser.parse_args()

    combined: list[dict[str, Any]] = []
    shard_counts: dict[str, int] = {}
    for path in args.shard_output:
        rows = read_jsonl(path)
        combined.extend(rows)
        shard_counts[path.name] = len(rows)
    request_ids = [str(row.get("request_id", "")) for row in combined]
    status_counts = Counter(str(row.get("generation_status", "missing")) for row in combined)
    invalid = [
        row.get("request_id")
        for row in combined
        if str(row.get("target_pseudo_asl_gloss", "")).strip()
        and not GLOSS_RE.fullmatch(str(row["target_pseudo_asl_gloss"]).strip())
    ]
    empty = [
        row.get("request_id")
        for row in combined
        if not str(row.get("target_pseudo_asl_gloss", "")).strip()
    ]
    revision_counts = Counter(
        str(row.get("generation", {}).get("resolved_revision", "missing"))
        for row in combined
    )
    batch_counts = Counter(
        int(row.get("generation", {}).get("batch_size", -1)) for row in combined
    )
    prompt_hash_counts = Counter(
        str(row.get("generation", {}).get("prompt_sha256", "missing"))
        for row in combined
    )
    accepted_raw_statuses = {
        "completed",
        "completed_with_quality_flags",
        "parse_failed",
    }
    unexpected_status_count = sum(
        count for status, count in status_counts.items() if status not in accepted_raw_statuses
    )
    repair_required_ids = set(str(value) for value in empty + invalid)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "shard_counts": shard_counts,
        "generated_count": len(combined),
        "expected_total": args.expected_total,
        "progress_fraction": round(len(combined) / args.expected_total, 6),
        "unique_request_id_count": len(set(request_ids)),
        "duplicate_request_id_count": len(request_ids) - len(set(request_ids)),
        "generation_status_counts": dict(status_counts),
        "empty_gloss_count": len(empty),
        "invalid_gloss_format_count": len(invalid),
        "repair_required_unique_request_count": len(repair_required_ids),
        "unexpected_generation_status_count": unexpected_status_count,
        "resolved_revision_counts": dict(revision_counts),
        "batch_size_counts": dict(batch_counts),
        "prompt_sha256_counts": dict(prompt_hash_counts),
        "raw_stream_integrity_ok": all(
            [
                len(request_ids) == len(set(request_ids)),
                unexpected_status_count == 0,
                revision_counts
                == Counter({args.expected_revision: len(combined)}),
                batch_counts
                == Counter({args.expected_batch_size: len(combined)}),
                len(prompt_hash_counts) == 1,
                all(row.get("pseudo_gloss_is_gold") is False for row in combined),
            ]
        ),
        "ready_without_repair": not empty and not invalid,
        "interpretation": (
            "raw_stream_integrity_ok permits explicitly flagged parse/format failures "
            "because the frozen pipeline preserves them and applies a transparent "
            "deterministic fallback after shard merging."
        ),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["raw_stream_integrity_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
