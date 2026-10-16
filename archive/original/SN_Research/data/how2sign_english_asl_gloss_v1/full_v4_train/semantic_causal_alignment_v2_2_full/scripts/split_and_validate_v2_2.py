#!/usr/bin/env python3
"""Validate v2.2 transition semantics and export deterministic split manifests."""

from __future__ import annotations

import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    pairs = read_jsonl(args.pairs)
    actions = read_jsonl(args.actions)
    pair_by_id = {row["id"]: row for row in pairs}
    errors = []
    if len(pair_by_id) != len(pairs):
        errors.append("duplicate_pair_ids")
    action_ids = [row["id"] for row in actions]
    if len(set(action_ids)) != len(action_ids):
        errors.append("duplicate_action_ids")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in actions:
        grouped[row["parent_id"]].append(row)
    for parent_id, rows in grouped.items():
        pair = pair_by_id.get(parent_id)
        if pair is None:
            errors.append(f"missing_pair:{parent_id}")
            continue
        rows.sort(key=lambda row: int(row["id"].rsplit("_", 1)[-1]))
        expected_prefix = 1
        committed = []
        for index, row in enumerate(rows):
            if int(row["source_prefix_length"]) != expected_prefix:
                errors.append(f"prefix_transition:{parent_id}:{index}")
                break
            action_type = row["target_action_type"]
            if action_type.startswith("WRITE"):
                committed.append(row["target_action"])
            elif action_type == "READ_WAIT":
                if expected_prefix >= int(row["source_token_count"]):
                    errors.append(f"wait_after_complete:{parent_id}:{index}")
                    break
                expected_prefix += 1
            elif action_type == "END":
                if index != len(rows) - 1 or expected_prefix != int(row["source_token_count"]):
                    errors.append(f"invalid_end:{parent_id}:{index}")
                break
            else:
                errors.append(f"unknown_action_type:{parent_id}:{index}:{action_type}")
                break
        if committed != pair["target_pseudo_asl_gloss_tokens"]:
            errors.append(f"committed_gloss_mismatch:{parent_id}")

    pair_splits = {split: [row for row in pairs if row.get("split", "train") == split] for split in ("train", "validation", "test")}
    action_splits = {split: [row for row in actions if row.get("split", "train") == split] for split in ("train", "validation", "test")}
    for split, rows in pair_splits.items():
        write_jsonl(args.output_dir / f"{split}_pairs.jsonl", rows)
    for split, rows in action_splits.items():
        write_jsonl(args.output_dir / f"{split}_actions.jsonl", rows)

    class_counts = {
        split: Counter(
            "WRITE" if row["target_action_type"].startswith("WRITE") else "WAIT" if row["target_action_type"] == "READ_WAIT" else "END"
            for row in rows
        )
        for split, rows in action_splits.items()
    }
    report = {
        "pair_split_counts": {split: len(rows) for split, rows in pair_splits.items()},
        "action_split_counts": {split: len(rows) for split, rows in action_splits.items()},
        "action_class_counts": {split: dict(counts) for split, counts in class_counts.items()},
        "validated_parent_count": len(grouped),
        "error_count": len(errors),
        "errors_preview": errors[:50],
        "ok": not errors and len(grouped) == len(pairs),
        "transition_semantics": "WRITE_keeps_prefix_WAIT_advances_prefix",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
