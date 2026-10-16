#!/usr/bin/env python3
"""Build a deterministic WAIT-balanced continuation-training manifest.

This is an optional calibration experiment. It preserves every WRITE and END
state while selecting WAIT states evenly across each parent's source prefixes.
The original full manifest remains unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
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


def action_type(row: dict[str, Any]) -> str:
    target_action = str(row.get("target_action", "")).upper()
    if target_action == "<WAIT>":
        return "WAIT"
    if target_action == "<END>":
        return "END"
    value = str(row.get("target_action_type", row.get("action_type", ""))).upper()
    if value in {"READ", "READ_WAIT"} or value.endswith("_WAIT"):
        return "WAIT"
    if value.startswith("WRITE"):
        return "WRITE"
    if value.startswith("WAIT"):
        return "WAIT"
    if value.startswith("END"):
        return "END"
    return "OTHER"


def evenly_spaced_indices(count: int, keep: int) -> set[int]:
    if keep >= count:
        return set(range(count))
    if keep <= 1:
        return {count // 2}
    return {
        min(count - 1, round(index * (count - 1) / (keep - 1)))
        for index in range(keep)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--maximum-wait-per-write", type=float, default=1.1)
    parser.add_argument("--minimum-wait-per-parent", type=int, default=2)
    args = parser.parse_args()

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.input):
        grouped[row["parent_id"]].append(row)

    selected: list[dict[str, Any]] = []
    parent_reports = []
    before = Counter()
    after = Counter()
    for parent_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row.get("action_index", 0)))
        waits = [row for row in rows if action_type(row) == "WAIT"]
        writes = [row for row in rows if action_type(row) == "WRITE"]
        target_wait_count = min(
            len(waits),
            max(
                args.minimum_wait_per_parent,
                math.ceil(max(1, len(writes)) * args.maximum_wait_per_write),
            ),
        )
        keep_wait_indices = evenly_spaced_indices(len(waits), target_wait_count)
        wait_index = 0
        kept = []
        for row in rows:
            kind = action_type(row)
            before[kind] += 1
            keep = kind != "WAIT" or wait_index in keep_wait_indices
            if kind == "WAIT":
                wait_index += 1
            if keep:
                copied = dict(row)
                copied["sampling_status"] = (
                    "kept_wait_evenly_spaced" if kind == "WAIT" else "kept_all_non_wait"
                )
                kept.append(copied)
                after[kind] += 1
        selected.extend(kept)
        parent_reports.append(
            {
                "parent_id": parent_id,
                "wait_before": len(waits),
                "wait_after": target_wait_count,
                "write_count": len(writes),
            }
        )

    write_jsonl(args.output, selected)
    report = {
        "scientific_status": "optional_wait_prior_calibration_not_primary_training_data",
        "input": str(args.input),
        "output": str(args.output),
        "parent_count": len(grouped),
        "input_action_count": sum(before.values()),
        "output_action_count": sum(after.values()),
        "input_action_type_counts": dict(before),
        "output_action_type_counts": dict(after),
        "maximum_wait_per_write": args.maximum_wait_per_write,
        "minimum_wait_per_parent": args.minimum_wait_per_parent,
        "selection": "all WRITE/END/OTHER plus evenly spaced WAIT states per parent",
        "important_note": (
            "Use only if the primary v2.2 model over-predicts WAIT on the locked test. "
            "Do not replace the immutable full manifest."
        ),
    }
    if before["WAIT"] and after["WAIT"] >= before["WAIT"]:
        raise RuntimeError(
            "WAIT balancing selected every WAIT state; check action-type parsing or thresholds."
        )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
