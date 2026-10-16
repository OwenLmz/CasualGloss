#!/usr/bin/env python3
"""Merge resumable multi-GPU Qwen generation shards in canonical request order."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests-jsonl", type=Path, required=True)
    parser.add_argument("--shard-output", type=Path, action="append", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()

    requests = read_jsonl(args.requests_jsonl)
    request_ids = [str(row["request_id"]) for row in requests]
    expected = set(request_ids)
    combined: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []
    for path in args.shard_output:
        rows = read_jsonl(path)
        combined.extend(rows)
        shard_summaries.append(
            {
                "path": str(path),
                "record_count": len(rows),
                "sha256": file_sha256(path),
            }
        )

    generated_ids = [str(row.get("request_id", "")) for row in combined]
    counts = Counter(generated_ids)
    duplicates = sorted(request_id for request_id, count in counts.items() if count > 1)
    missing = sorted(expected.difference(counts))
    unexpected = sorted(set(counts).difference(expected))
    if duplicates or missing or unexpected:
        raise SystemExit(
            json.dumps(
                {
                    "duplicate_request_ids_preview": duplicates[:20],
                    "missing_request_ids_preview": missing[:20],
                    "unexpected_request_ids_preview": unexpected[:20],
                },
                indent=2,
            )
        )

    by_id = {str(row["request_id"]): row for row in combined}
    ordered = [by_id[request_id] for request_id in request_ids]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "requests_jsonl": str(args.requests_jsonl),
        "requests_count": len(requests),
        "shards": shard_summaries,
        "combined_count": len(combined),
        "unique_generated_request_id_count": len(counts),
        "missing_request_id_count": len(missing),
        "duplicate_request_id_count": len(duplicates),
        "unexpected_request_id_count": len(unexpected),
        "output_jsonl": str(args.output_jsonl),
        "output_sha256": file_sha256(args.output_jsonl),
        "pipeline_ok": (
            len(ordered) == len(requests)
            and not missing
            and not duplicates
            and not unexpected
        ),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
