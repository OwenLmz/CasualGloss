#!/usr/bin/env python3
"""Check whether a rule revision changes the already prepared scoring population."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("alignment_v2_1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--excluded", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    alignment = load_module(args.finalizer)
    old_retained_rows = read_jsonl(args.retained)
    old_excluded_rows = read_jsonl(args.excluded)
    all_rows = old_retained_rows + old_excluded_rows
    new_retained = set()
    reasons_by_id = {}
    for row in all_rows:
        _, order = alignment.occurrence_aware_anchors(
            row["target_pseudo_asl_gloss_tokens"], row["source_tokens"]
        )
        reasons = []
        if alignment.source_is_corrupt(row["source_text_en"]):
            reasons.append("corrupt_or_non_english_source_text")
        if (
            order["lexically_matchable_token_count"] >= int(config["noncausal_min_lexical_token_count"])
            and order["monotonic_coverage"] < float(config["noncausal_min_monotonic_coverage"])
            and (
                order["backward_jump_count"] > int(config["noncausal_max_backward_jump_count"])
                or order["backward_distance"] > int(config["noncausal_max_backward_distance"])
            )
        ):
            reasons.append("strong_streaming_noncausal_gloss_order")
        reasons_by_id[row["id"]] = reasons
        if not reasons:
            new_retained.add(row["id"])
    old_retained = {row["id"] for row in old_retained_rows}
    report = {
        "old_retained_count": len(old_retained),
        "new_retained_count": len(new_retained),
        "newly_retained_count": len(new_retained - old_retained),
        "newly_excluded_count": len(old_retained - new_retained),
        "newly_retained_ids": sorted(new_retained - old_retained),
        "newly_excluded": [
            {"id": row_id, "reasons": reasons_by_id[row_id]}
            for row_id in sorted(old_retained - new_retained)
        ],
        "scoring_population_remains_valid": not (old_retained - new_retained),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["scoring_population_remains_valid"] else 2)


if __name__ == "__main__":
    main()
