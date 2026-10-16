#!/usr/bin/env python3
"""Shard Epee requests and freeze a length-stratified hardware smoke set."""

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
    parser.add_argument("--epee-requests", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--original-requests", type=Path, required=True)
    parser.add_argument("--original-predictions", type=Path, required=True)
    parser.add_argument("--smoke-requests", type=Path, required=True)
    parser.add_argument("--smoke-reference", type=Path, required=True)
    parser.add_argument("--smoke-count", type=int, default=20)
    args = parser.parse_args()

    epee = read_jsonl(args.epee_requests)
    for shard_index in range(args.shard_count):
        # Round-robin sharding balances prompt lengths while retaining stable IDs.
        write_jsonl(
            args.shard_dir / f"epee_requests.shard{shard_index}.jsonl",
            epee[shard_index :: args.shard_count],
        )

    requests = {str(row["request_id"]): row for row in read_jsonl(args.original_requests)}
    predictions = [
        row
        for row in read_jsonl(args.original_predictions)
        if row.get("generation_status") == "completed"
        and row.get("target_pseudo_asl_gloss")
        and str(row.get("request_id")) in requests
    ]
    predictions.sort(
        key=lambda row: (
            len(str(requests[str(row["request_id"])]["source_text_en"]).split()),
            int(requests[str(row["request_id"])]["request_index"]),
        )
    )
    if len(predictions) < args.smoke_count:
        raise SystemExit("Not enough valid original predictions for smoke selection")

    indices = [
        round(index * (len(predictions) - 1) / max(1, args.smoke_count - 1))
        for index in range(args.smoke_count)
    ]
    selected_predictions = [predictions[index] for index in indices]
    selected_requests = [requests[str(row["request_id"])] for row in selected_predictions]
    for request_index, row in enumerate(selected_requests):
        row["request_index"] = request_index

    write_jsonl(args.smoke_requests, selected_requests)
    write_jsonl(args.smoke_reference, selected_predictions)
    print(
        json.dumps(
            {
                "epee_count": len(epee),
                "shard_counts": [len(epee[index :: args.shard_count]) for index in range(args.shard_count)],
                "smoke_count": len(selected_requests),
                "smoke_source_length_range": [
                    len(selected_requests[0]["source_text_en"].split()),
                    len(selected_requests[-1]["source_text_en"].split()),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
