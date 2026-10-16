#!/usr/bin/env python3
"""Replace JSONL records by ID while preserving base order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--replacement", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--id-field", default="id")
    args = parser.parse_args()

    base = read_jsonl(args.base)
    replacements: dict[str, dict] = {}
    for path in args.replacement:
        for row in read_jsonl(path):
            key = row[args.id_field]
            if key in replacements:
                raise ValueError(f"Duplicate replacement ID: {key}")
            replacements[key] = row
    base_ids = {row[args.id_field] for row in base}
    unknown = sorted(set(replacements) - base_ids)
    if unknown:
        raise ValueError(f"Replacement IDs absent from base: {unknown[:10]}")
    merged = [replacements.get(row[args.id_field], row) for row in base]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "base": str(args.base),
        "replacement_files": [str(path) for path in args.replacement],
        "output": str(args.output),
        "base_record_count": len(base),
        "replacement_record_count": len(replacements),
        "output_record_count": len(merged),
        "replaced_ids": sorted(replacements),
        "ok": len(merged) == len(base),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
