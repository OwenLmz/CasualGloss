#!/usr/bin/env python3
"""Check deterministic equivalence of two generation batch sizes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()

    first = read_jsonl(args.first)
    second = read_jsonl(args.second)
    same_ids = [row["request_id"] for row in first] == [
        row["request_id"] for row in second
    ]
    same_gloss_count = sum(
        left.get("target_pseudo_asl_gloss") == right.get("target_pseudo_asl_gloss")
        for left, right in zip(first, second)
    )
    same_raw_count = sum(
        left.get("raw_model_output") == right.get("raw_model_output")
        for left, right in zip(first, second)
    )
    report = {
        "first_count": len(first),
        "second_count": len(second),
        "same_request_id_order": same_ids,
        "same_gloss_count": same_gloss_count,
        "same_raw_output_count": same_raw_count,
        "first_status_counts": dict(
            Counter(row.get("generation_status") for row in first)
        ),
        "second_status_counts": dict(
            Counter(row.get("generation_status") for row in second)
        ),
        "pipeline_ok": (
            len(first) == len(second)
            and same_ids
            and same_gloss_count == len(first)
            and same_raw_count == len(first)
        ),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pipeline_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
