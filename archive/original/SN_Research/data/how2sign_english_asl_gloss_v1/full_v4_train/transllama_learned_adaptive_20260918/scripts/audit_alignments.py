#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transllama_data import WAIT_SYMBOL, make_prefix_example, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--check-all-prefixes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    errors: list[dict] = []
    seen_ids: set[str] = set()
    prefix_count = 0

    def fail(row_id: str, message: str) -> None:
        if len(errors) < 100:
            errors.append({"id": row_id, "error": message})

    for row in rows:
        row_id = str(row["id"])
        if row_id in seen_ids:
            fail(row_id, "duplicate id")
        seen_ids.add(row_id)

        source_words = list(row["source_words"])
        target_words = list(row["target_words"])
        aligned_source = list(row["aligned_source_words"])
        aligned_target = list(row["aligned_target_words"])
        if not source_words or not target_words:
            fail(row_id, "empty source or target")
        if WAIT_SYMBOL in source_words or WAIT_SYMBOL in target_words:
            fail(row_id, "reserved WAIT occurs in raw tokens")
        if len(aligned_source) != len(aligned_target):
            fail(row_id, "aligned lengths differ")
        if [word for word in aligned_source if word != WAIT_SYMBOL] != source_words:
            fail(row_id, "aligned source does not preserve raw source")
        if [word for word in aligned_target if word != WAIT_SYMBOL] != target_words:
            fail(row_id, "aligned target does not preserve raw target")

        fixed_before = [tuple(edge) for edge in row["fixed_alignment_before_wait_insertion"]]
        if len({edge[0] for edge in fixed_before}) != len(fixed_before):
            fail(row_id, "fixed alignment repeats a source index")
        if len({edge[1] for edge in fixed_before}) != len(fixed_before):
            fail(row_id, "fixed alignment repeats a target index")
        fixed_edges = [tuple(edge) for edge in row["fixed_alignment_after_wait_insertion"]]
        if any(target_index < source_index for source_index, target_index in fixed_edges):
            fail(row_id, "post-insertion alignment is non-causal")

        prefix_lengths = range(1, len(source_words) + 1) if args.check_all_prefixes else [len(source_words)]
        for prefix_length in prefix_lengths:
            prefix = make_prefix_example(row, prefix_length)
            assistant_words = list(prefix["assistant_target_words"])
            prefix_count += 1
            if not assistant_words:
                fail(row_id, f"empty assistant target at prefix {prefix_length}")
            wait_positions = [i for i, word in enumerate(assistant_words) if word == WAIT_SYMBOL]
            if wait_positions and wait_positions != [len(assistant_words) - 1]:
                fail(row_id, f"non-final or repeated WAIT at prefix {prefix_length}")
            if prefix_length == len(source_words) and assistant_words != target_words:
                fail(row_id, "full-source target differs from raw target")

    if len(rows) != args.expected_count:
        errors.append(
            {"id": "__file__", "error": f"expected {args.expected_count} rows, found {len(rows)}"}
        )

    report = {
        "input": str(args.input),
        "expected_count": args.expected_count,
        "row_count": len(rows),
        "unique_id_count": len(seen_ids),
        "prefix_count_checked": prefix_count,
        "check_all_prefixes": args.check_all_prefixes,
        "passed": not errors,
        "errors": errors,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit("alignment audit failed")


if __name__ == "__main__":
    main()
