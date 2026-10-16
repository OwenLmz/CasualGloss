#!/usr/bin/env python3
"""Merge ordered JSONL shards and verify exact manifest identity/status."""

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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--status-field", default="streaming_frontend_status")
    parser.add_argument("--completed-value", default="completed")
    args = parser.parse_args()

    expected = read_jsonl(args.manifest)
    rows = [row for path in args.shard for row in read_jsonl(path)]
    by_id = {row["id"]: row for row in rows}
    duplicate_count = len(rows) - len(by_id)
    expected_ids = [row["id"] for row in expected]
    missing = [identifier for identifier in expected_ids if identifier not in by_id]
    extra = sorted(set(by_id) - set(expected_ids))
    ordered = [by_id[identifier] for identifier in expected_ids if identifier in by_id]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    completed_count = sum(
        row.get(args.status_field) == args.completed_value for row in ordered
    )
    report = {
        "expected_count": len(expected),
        "input_row_count": len(rows),
        "output_count": len(ordered),
        "duplicate_count": duplicate_count,
        "missing_ids": missing,
        "extra_ids": extra,
        "status_field": args.status_field,
        "completed_value": args.completed_value,
        "completed_count": completed_count,
        "ok": (
            len(ordered) == len(expected)
            and duplicate_count == 0
            and not missing
            and not extra
            and completed_count == len(expected)
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
