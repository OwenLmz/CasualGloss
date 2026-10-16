#!/usr/bin/env python3
"""Apply explicit audited repairs to a small set of flagged translations."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--repairs", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    repairs = json.loads(args.repairs.read_text(encoding="utf-8-sig"))
    remaining = set(repairs)
    applied: list[dict[str, Any]] = []

    for manifest in sorted(args.shard_dir.glob("translated.shard*.jsonl")):
        rows = read_jsonl(manifest)
        shard_applied: list[dict[str, Any]] = []
        for row in rows:
            identifier = str(row["id"])
            if identifier not in repairs:
                continue
            if row.get("source_text_zh_status") != "flagged":
                raise ValueError(f"Repair target {identifier} is not flagged")
            repair = repairs[identifier]
            before = row.get("source_text_zh")
            row.update(
                {
                    "source_text_zh_model_output_before_repair": before,
                    "source_text_zh": repair["translation"],
                    "source_text_zh_status": "completed",
                    "source_text_zh_quality_flags": [],
                    "source_text_zh_repair_method": "explicit_curated_quality_repair_v1",
                    "source_text_zh_repair_reason": repair["reason"],
                    "source_text_zh_repaired_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            entry = {
                "id": identifier,
                "manifest": str(manifest),
                "before": before,
                "after": repair["translation"],
                "reason": repair["reason"],
            }
            shard_applied.append(entry)
            applied.append(entry)
            remaining.discard(identifier)
        if shard_applied:
            write_jsonl_atomic(manifest, rows)
            shard_number = manifest.stem.replace("translated.shard", "")
            report_path = args.report_dir / f"translation.shard{shard_number}.json"
            report = json.loads(report_path.read_text(encoding="utf-8-sig"))
            completed = sum(row.get("source_text_zh_status") == "completed" for row in rows)
            flagged = sum(row.get("source_text_zh_status") == "flagged" for row in rows)
            failed = len(rows) - completed - flagged
            report.update(
                {
                    "completed_count": completed,
                    "flagged_count": flagged,
                    "failed_count": failed,
                    "flagged_rows": [],
                    "quality_repair_count": len(shard_applied),
                    "quality_repairs": shard_applied,
                    "ok": completed == len(rows),
                }
            )
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repair_specification": str(args.repairs),
        "expected_repair_count": len(repairs),
        "applied_repair_count": len(applied),
        "missing_repair_ids": sorted(remaining),
        "repairs": applied,
        "ok": len(applied) == len(repairs) and not remaining,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
