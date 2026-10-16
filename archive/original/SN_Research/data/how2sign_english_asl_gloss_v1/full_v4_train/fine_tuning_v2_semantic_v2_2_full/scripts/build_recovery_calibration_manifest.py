#!/usr/bin/env python3
"""Merge balanced on-policy states with off-policy delay-recovery states."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def action_class(row: dict[str, Any]) -> str:
    action = str(row.get("target_action", ""))
    if action == "<WAIT>":
        return "WAIT"
    if action == "<END>":
        return "END"
    return "WRITE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced-actions", type=Path, required=True)
    parser.add_argument("--recovery-actions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    seen_ids: set[str] = set()
    counts = Counter()
    source_counts = Counter()
    output: list[dict[str, Any]] = []
    for source_name, path in (
        ("balanced_on_policy", args.balanced_actions),
        ("delay_recovery", args.recovery_actions),
    ):
        for row in read_jsonl(path):
            row_id = str(row["id"])
            if row_id in seen_ids:
                raise RuntimeError(f"Duplicate action id: {row_id}")
            seen_ids.add(row_id)
            copied = dict(row)
            copied["calibration_source"] = source_name
            output.append(copied)
            counts[action_class(copied)] += 1
            source_counts[source_name] += 1

    if not output or not all(counts[name] for name in ("WAIT", "WRITE", "END")):
        raise RuntimeError(f"Invalid calibration class counts: {dict(counts)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "scientific_status": "pseudo_label_delay_recovery_calibration_not_gold_asl",
        "output_action_count": len(output),
        "action_class_counts": dict(counts),
        "action_class_ratios": {
            key: round(value / len(output), 8) for key, value in counts.items()
        },
        "source_counts": dict(source_counts),
        "duplicate_id_count": 0,
        "sha256": {
            "balanced_actions": sha256(args.balanced_actions),
            "recovery_actions": sha256(args.recovery_actions),
            "output": sha256(args.output),
        },
        "important_note": (
            "Use only as a separate continuation-training experiment after the primary v2.2.1 run. "
            "The manifest balances WAIT/WRITE exposure and teaches recovery after extra READ delay."
        ),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
