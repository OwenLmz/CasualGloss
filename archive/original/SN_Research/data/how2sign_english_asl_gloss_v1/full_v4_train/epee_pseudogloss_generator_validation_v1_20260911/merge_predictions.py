#!/usr/bin/env python3
"""Merge deterministic generator shards without overwriting shard evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()

    requests = read_jsonl(args.requests)
    request_ids = [str(row["request_id"]) for row in requests]
    predictions: dict[str, dict[str, Any]] = {}
    shard_counts = {}
    for shard in args.shard:
        rows = read_jsonl(shard)
        shard_counts[str(shard)] = len(rows)
        for row in rows:
            request_id = str(row["request_id"])
            if request_id in predictions:
                raise SystemExit(f"Duplicate prediction ID: {request_id}")
            predictions[request_id] = row
    if set(request_ids) != set(predictions):
        raise SystemExit("Merged predictions do not exactly cover requests")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for request_id in request_ids:
            handle.write(json.dumps(predictions[request_id], ensure_ascii=False) + "\n")
    report = {
        "request_count": len(request_ids),
        "prediction_count": len(predictions),
        "shard_counts": shard_counts,
        "exact_id_coverage": True,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
