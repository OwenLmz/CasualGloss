#!/usr/bin/env python3
"""Audit Epee rollout structure and state-transition invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from epee_common import load_manifest, read_jsonl, write_json


def fail(errors: list[str], row_id: str, message: str) -> None:
    errors.append(f"{row_id}: {message}")


def validate_row(source: dict[str, Any], row: dict[str, Any], errors: list[str]) -> None:
    row_id = str(source["id"])
    if str(row.get("id")) != row_id:
        fail(errors, row_id, f"rollout ID mismatch: {row.get('id')}")
    if row.get("source_tokens") != source["source_tokens"]:
        fail(errors, row_id, "source tokens differ from frozen manifest")
    if row.get("source_text_en") != source["source_text_en"]:
        fail(errors, row_id, "source text differs from frozen manifest")
    if row.get("inference_dtype") != "float16":
        fail(errors, row_id, f"unexpected inference dtype: {row.get('inference_dtype')}")
    if row.get("reference_exposed_to_model") is not False:
        fail(errors, row_id, "reference_exposed_to_model is not explicitly false")
    if row.get("rollout_history_uses_reference_gloss") is not False:
        fail(errors, row_id, "rollout history reference flag is not explicitly false")

    source_count = len(source["source_tokens"])
    events = row.get("rollout_events")
    if not isinstance(events, list):
        fail(errors, row_id, "rollout_events is not a list")
        return
    if row.get("action_count") != len(events):
        fail(errors, row_id, "action_count does not match event count")
    if len(events) > int(row.get("max_action_count", -1)):
        fail(errors, row_id, "event count exceeds shared maximum action count")

    expected_writes: list[int] = []
    last_after: int | None = None
    for index, event in enumerate(events):
        before = event.get("source_prefix_length")
        after = event.get("source_prefix_length_after")
        if not isinstance(before, int) or not isinstance(after, int):
            fail(errors, row_id, f"event {index} lacks integer source positions")
            continue
        if not 1 <= before <= source_count or not 1 <= after <= source_count:
            fail(errors, row_id, f"event {index} source position outside [1,{source_count}]")
        if last_after is not None and before != last_after:
            fail(errors, row_id, f"event {index} prefix is not continuous")
        delta = after - before
        if delta not in (0, 1):
            fail(errors, row_id, f"event {index} advances source by {delta}")
        advanced = bool(event.get("source_advanced_after_action", False))
        if advanced != (delta == 1):
            fail(errors, row_id, f"event {index} advance flag disagrees with positions")
        action_type = event.get("predicted_action_class")
        if action_type == "WRITE":
            expected_writes.append(before)
            if delta != 0:
                fail(errors, row_id, f"event {index} WRITE also advances source")
        if any("reference" in str(key).lower() for key in event):
            fail(errors, row_id, f"event {index} contains a reference-named field")
        last_after = after

    if expected_writes != row.get("write_source_positions"):
        fail(errors, row_id, "WRITE event positions do not match write_source_positions")
    if len(expected_writes) != len(row.get("predicted_gloss_tokens", [])):
        fail(errors, row_id, "WRITE count does not match predicted gloss token count")
    allowed_termination = {
        "end_after_full_source",
        "premature_end",
        "maximum_actions_reached",
        "loop_detected",
    }
    if row.get("termination_reason") not in allowed_termination:
        fail(errors, row_id, f"unexpected termination reason: {row.get('termination_reason')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = load_manifest(args.manifest, args.expected_count)
    rows = read_jsonl(args.rollout)
    row_by_id = {str(row.get("id")): row for row in rows}
    expected_ids = {str(source["id"]) for source in sources}
    errors: list[str] = []
    if len(rows) != len(row_by_id):
        errors.append("rollout contains duplicate IDs")
    missing = expected_ids - set(row_by_id)
    extra = set(row_by_id) - expected_ids
    if missing:
        errors.append(f"missing rollout IDs: {sorted(missing)}")
    if extra:
        errors.append(f"unexpected rollout IDs: {sorted(extra)}")
    for source in sources:
        if str(source["id"]) in row_by_id:
            validate_row(source, row_by_id[str(source["id"])], errors)

    report = {
        "passed": not errors,
        "expected_source_count": args.expected_count,
        "rollout_row_count": len(rows),
        "error_count": len(errors),
        "errors": errors,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
