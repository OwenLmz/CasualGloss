#!/usr/bin/env python3
"""Validate v2 parent records and expanded causal action trajectories."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parents", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    parents = read_jsonl(args.parents)
    pairs = read_jsonl(args.pairs)
    actions = read_jsonl(args.actions)
    scores = [row for path in args.scores for row in read_jsonl(path)]
    parent_ids = {row["id"] for row in parents}
    pair_ids = [row["id"] for row in pairs]
    score_ids = [row["id"] for row in scores]
    failures = []
    action_counts = Counter(row["parent_id"] for row in actions)

    for row in pairs:
        gloss = row["target_pseudo_asl_gloss_tokens"]
        writes = [item["action"] for item in row["causal_timeline"] if str(item["action_type"]).startswith("WRITE")]
        timeline_positions = [int(item["source_prefix_length"]) for item in row["causal_timeline"]]
        semantic_positions = [item["scheduled_emit_source_position"] for item in row["semantic_gloss_token_alignment"]]
        if writes != gloss:
            failures.append(f"{row['id']}: write sequence does not reconstruct target gloss")
        if row["causal_action_sequence"][-1] != config["end_token"]:
            failures.append(f"{row['id']}: END is not the final action")
        if row["causal_action_sequence"].count(config["end_token"]) != 1:
            failures.append(f"{row['id']}: END count is not one")
        if timeline_positions != sorted(timeline_positions):
            failures.append(f"{row['id']}: action source positions are not monotonic")
        if semantic_positions != sorted(semantic_positions):
            failures.append(f"{row['id']}: semantic emit positions are not monotonic")
        if action_counts[row["id"]] != len(row["causal_timeline"]):
            failures.append(f"{row['id']}: action manifest count mismatch")
        if row["target_pseudo_gloss_is_gold"] is not False or row["alignment_is_gold"] is not False:
            failures.append(f"{row['id']}: pseudo labels are not explicitly marked non-gold")

    expected = int(config["pilot_parent_count"])
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_parent_count": expected,
        "input_parent_count": len(parents),
        "pair_count": len(pairs),
        "score_count": len(scores),
        "action_count": len(actions),
        "unique_input_parent_count": len(parent_ids),
        "unique_pair_parent_count": len(set(pair_ids)),
        "unique_score_parent_count": len(set(score_ids)),
        "parent_id_sets_match": parent_ids == set(pair_ids) == set(score_ids),
        "validation_failure_count": len(failures),
        "validation_failures_preview": failures[:50],
        "all_targets_marked_non_gold": all(row["target_pseudo_gloss_is_gold"] is False for row in pairs),
        "all_alignments_marked_non_gold": all(row["alignment_is_gold"] is False for row in pairs),
        "ok": (
            len(parents) == len(pairs) == len(scores) == expected
            and len(parent_ids) == len(set(pair_ids)) == len(set(score_ids)) == expected
            and parent_ids == set(pair_ids) == set(score_ids)
            and not failures
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
