#!/usr/bin/env python3
"""Create deterministic, balanced round-robin JSONL shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.open(encoding="utf-8-sig") if line.strip()]
    outputs = [[] for _ in range(args.shards)]
    for index, row in enumerate(rows):
        outputs[index % args.shards].append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, shard in enumerate(outputs):
        path = args.output_dir / f"{args.prefix}.shard{index}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in shard:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"input_count": len(rows), "shard_counts": [len(x) for x in outputs]}))


if __name__ == "__main__":
    main()
