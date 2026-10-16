#!/usr/bin/env python3
"""Reconcile an in-flight scoring population after a deterministic rule fix."""

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
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--excluded", type=Path, required=True)
    parser.add_argument("--consistency-report", type=Path, required=True)
    parser.add_argument("--output-parents", type=Path, required=True)
    parser.add_argument("--output-delta", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    retained = read_jsonl(args.retained)
    excluded = read_jsonl(args.excluded)
    report = json.loads(args.consistency_report.read_text(encoding="utf-8-sig"))
    delta_ids = set(report["newly_retained_ids"])
    delta = [row for row in excluded if row["id"] in delta_ids]
    if {row["id"] for row in delta} != delta_ids:
        raise SystemExit("Could not recover every newly retained row")
    for row in delta:
        row["semantic_v2_shard_id"] = 2
        row["recovered_after_contraction_rule_fix"] = True
    reconciled = sorted(retained + delta, key=lambda row: row["semantic_v2_full_index"])
    write_jsonl(args.output_parents, reconciled)
    write_jsonl(args.output_delta, delta)
    output = {
        "original_scoring_parent_count": len(retained),
        "delta_scoring_parent_count": len(delta),
        "reconciled_parent_count": len(reconciled),
        "newly_excluded_in_finalizer_count": report["newly_excluded_count"],
        "note": "Newly excluded rows may have scores but will be removed deterministically during finalization.",
        "ok": len(reconciled) == len(retained) + len(delta),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
