#!/usr/bin/env python3
"""Export split-specific v2.1 pair/action manifests for Fine-tune 2."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
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
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    pairs = read_jsonl(args.pairs)
    actions = read_jsonl(args.actions)
    eligible_ids = {row["id"] for row in pairs if row["eligible_for_finetune2_v2_1"]}
    errors = []
    for split in ("train", "validation", "test"):
        split_pairs = [row for row in pairs if row["split"] == split and row["eligible_for_finetune2_v2_1"]]
        split_ids = {row["id"] for row in split_pairs}
        split_actions = [row for row in actions if row["parent_id"] in split_ids]
        write_jsonl(args.output_root / "manifests" / f"{split}_pairs.jsonl", split_pairs)
        write_jsonl(args.output_root / "manifests" / f"{split}_actions.jsonl", split_actions)
        if any(row["parent_id"] not in split_ids for row in split_actions):
            errors.append(f"{split}_action_parent_mismatch")
    if {row["parent_id"] for row in actions} != eligible_ids:
        errors.append("global_action_parent_set_mismatch")
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_scored_parent_count": len(pairs),
        "eligible_parent_count": len(eligible_ids),
        "excluded_after_scoring_count": len(pairs) - len(eligible_ids),
        "eligible_parent_split_counts": dict(Counter(row["split"] for row in pairs if row["eligible_for_finetune2_v2_1"])),
        "action_split_counts": dict(Counter(row["split"] for row in actions)),
        "total_action_count": len(actions),
        "errors": errors,
        "ok": not errors,
    }
    path = args.output_root / "reports" / "split_export_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 2)


if __name__ == "__main__":
    main()
