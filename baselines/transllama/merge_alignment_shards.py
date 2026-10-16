#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from transllama_data import file_sha256, read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()

    rows = [row for shard in args.shards for row in read_jsonl(shard)]
    rows.sort(key=lambda row: int(row["input_line_index"]))
    ids = [str(row["id"]) for row in rows]
    indices = [int(row["input_line_index"]) for row in rows]
    if len(rows) != args.expected_count:
        raise SystemExit(f"expected {args.expected_count} rows, found {len(rows)}")
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate parent IDs across shards")
    if indices != list(range(args.expected_count)):
        raise SystemExit("merged input indices are not contiguous")

    write_jsonl(args.output, rows)
    write_json(
        args.report,
        {
            "row_count": len(rows),
            "unique_parent_count": len(set(ids)),
            "shards": [str(path) for path in args.shards],
            "sha256": {
                "output": file_sha256(args.output),
                "shards": {str(path): file_sha256(path) for path in args.shards},
            },
        },
    )


if __name__ == "__main__":
    main()

