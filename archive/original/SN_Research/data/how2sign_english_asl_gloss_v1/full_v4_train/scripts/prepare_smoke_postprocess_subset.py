#!/usr/bin/env python3
"""Build matching unique-request and all-record subsets for postprocess smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-records-jsonl", type=Path, required=True)
    parser.add_argument("--unique-requests-jsonl", type=Path, required=True)
    parser.add_argument("--request-count", type=int, default=32)
    parser.add_argument("--output-all-records-jsonl", type=Path, required=True)
    parser.add_argument("--output-requests-jsonl", type=Path, required=True)
    args = parser.parse_args()

    requests = read_jsonl(args.unique_requests_jsonl)[: args.request_count]
    request_ids = {str(row["request_id"]) for row in requests}
    all_records = [
        row
        for row in read_jsonl(args.all_records_jsonl)
        if str(row["request_id"]) in request_ids
    ]
    write_jsonl(args.output_requests_jsonl, requests)
    write_jsonl(args.output_all_records_jsonl, all_records)
    print(
        json.dumps(
            {
                "unique_request_count": len(requests),
                "matching_all_record_count": len(all_records),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
