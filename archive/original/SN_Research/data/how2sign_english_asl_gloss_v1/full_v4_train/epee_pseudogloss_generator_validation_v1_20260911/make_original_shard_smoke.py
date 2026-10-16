#!/usr/bin/env python3
"""Build complete batches from one frozen original generation shard."""

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
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-requests", type=Path, required=True)
    parser.add_argument("--original-predictions", type=Path, required=True)
    parser.add_argument("--smoke-requests", type=Path, required=True)
    parser.add_argument("--smoke-reference", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    if args.count % args.batch_size:
        raise SystemExit("count must contain complete original batches")

    all_requests = read_jsonl(args.original_requests)
    shard_requests = sorted(
        [row for row in all_requests if int(row["generation_shard_id"]) == args.shard_id],
        key=lambda row: int(row["request_index"]),
    )
    selected = shard_requests[: args.count]
    if len(selected) != args.count:
        raise SystemExit("Not enough requests in the selected original shard")

    predictions = {
        str(row["request_id"]): row
        for row in read_jsonl(args.original_predictions)
        if row.get("generation_status") == "completed" and row.get("target_pseudo_asl_gloss")
    }
    missing = [str(row["request_id"]) for row in selected if str(row["request_id"]) not in predictions]
    if missing:
        raise SystemExit(f"Missing original predictions: {missing}")

    write_jsonl(args.smoke_requests, selected)
    write_jsonl(args.smoke_reference, [predictions[str(row["request_id"])] for row in selected])
    print(
        json.dumps(
            {
                "selection": "complete_batches_from_original_generation_shard",
                "generation_shard_id": args.shard_id,
                "count": args.count,
                "batch_size": args.batch_size,
                "request_indices": [int(row["request_index"]) for row in selected],
                "source_length_range": [
                    min(len(str(row["source_text_en"]).split()) for row in selected),
                    max(len(str(row["source_text_en"]).split()) for row in selected),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
