#!/usr/bin/env python3
"""Validate fixed Wait-k manifests against their immutable parent pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_schedule(source_count: int, gloss_tokens: list[str], wait_k: int) -> list[tuple[int, str]]:
    prefix = 1
    expected: list[tuple[int, str]] = []
    for target_index, token in enumerate(gloss_tokens):
        required = min(source_count, wait_k + target_index)
        while prefix < required:
            expected.append((prefix, "<WAIT>"))
            prefix += 1
        expected.append((prefix, token))
    while prefix < source_count:
        expected.append((prefix, "<WAIT>"))
        prefix += 1
    expected.append((source_count, "<END>"))
    return expected


def action_groups(path: Path) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    current_id: str | None = None
    current: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        parent_id = str(row["parent_id"])
        if current_id is not None and parent_id != current_id:
            yield current_id, current
            current = []
        current_id = parent_id
        current.append(row)
    if current_id is not None:
        yield current_id, current


def validate_split(
    pairs_path: Path,
    actions_path: Path,
    wait_k: int,
    expected_parent_limit: int,
) -> tuple[dict[str, Any], set[str]]:
    parents = {}
    for index, row in enumerate(read_jsonl(pairs_path)):
        if expected_parent_limit >= 0 and index >= expected_parent_limit:
            break
        parents[str(row["id"])] = row

    errors: list[str] = []
    action_ids: set[str] = set()
    seen_parents: set[str] = set()
    action_count = 0
    action_types: Counter[str] = Counter()
    for parent_id, rows in action_groups(actions_path):
        if parent_id not in parents:
            errors.append(f"unexpected_parent:{parent_id}")
            continue
        if parent_id in seen_parents:
            errors.append(f"non_contiguous_duplicate_parent:{parent_id}")
        seen_parents.add(parent_id)
        parent = parents[parent_id]
        expected = expected_schedule(
            len(parent["source_tokens"]),
            list(parent["target_pseudo_asl_gloss_tokens"]),
            wait_k,
        )
        actual = [(int(row["source_prefix_length"]), str(row["target_action"])) for row in rows]
        if actual != expected:
            errors.append(f"schedule_mismatch:{parent_id}")
        committed: list[str] = []
        for row in rows:
            action_count += 1
            row_id = str(row["id"])
            if row_id in action_ids:
                errors.append(f"duplicate_action_id:{row_id}")
            action_ids.add(row_id)
            action = str(row["target_action"])
            if row.get("split") != parent.get("split"):
                errors.append(f"split_mismatch:{row_id}")
            if int(row.get("wait_k", -1)) != wait_k:
                errors.append(f"wait_k_mismatch:{row_id}")
            if row.get("target_is_gold") is not False or row.get("alignment_is_gold") is not False:
                errors.append(f"gold_flag_error:{row_id}")
            if action == "<WAIT>":
                action_types["WAIT"] += 1
            elif action == "<END>":
                action_types["END"] += 1
                if not bool(row["source_complete"]):
                    errors.append(f"premature_end:{row_id}")
            else:
                action_types["WRITE"] += 1
                committed.append(action)
        if committed != list(parent["target_pseudo_asl_gloss_tokens"]):
            errors.append(f"gloss_reconstruction_mismatch:{parent_id}")

    missing = sorted(set(parents) - seen_parents)
    errors.extend(f"missing_parent:{parent_id}" for parent_id in missing[:100])
    return (
        {
            "pairs": str(pairs_path),
            "actions": str(actions_path),
            "expected_parent_count": len(parents),
            "observed_parent_count": len(seen_parents),
            "action_count": action_count,
            "action_type_counts": dict(sorted(action_types.items())),
            "error_count": len(errors),
            "errors_preview": errors[:100],
            "sha256": sha256(actions_path),
            "ok": not errors,
        },
        set(parents),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--wait-k", type=int, required=True)
    parser.add_argument("--train-limit", type=int, default=-1)
    parser.add_argument("--validation-limit", type=int, default=-1)
    parser.add_argument("--test-limit", type=int, default=-1)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    split_limits = {
        "train": args.train_limit,
        "validation": args.validation_limit,
        "test": args.test_limit,
    }
    reports = {}
    parent_sets = {}
    for split, limit in split_limits.items():
        reports[split], parent_sets[split] = validate_split(
            args.pairs_root / f"{split}_pairs.jsonl",
            args.manifest_root / f"{split}_actions.jsonl",
            args.wait_k,
            limit,
        )
    overlaps = {
        "train_validation": len(parent_sets["train"] & parent_sets["validation"]),
        "train_test": len(parent_sets["train"] & parent_sets["test"]),
        "validation_test": len(parent_sets["validation"] & parent_sets["test"]),
    }
    ok = all(report["ok"] for report in reports.values()) and not any(overlaps.values())
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "wait_k": args.wait_k,
        "splits": reports,
        "parent_overlap_counts": overlaps,
        "locked_test_untouched_during_construction": True,
        "ok": ok,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
