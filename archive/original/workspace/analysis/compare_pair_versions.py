#!/usr/bin/env python3
"""Verify parent text/gloss identity between v1 and v2.2 pair manifests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path("/123090274/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train")
V1 = ROOT / "causal_alignment_v1_corrected_full/outputs/causal_aligned_pairs_full.jsonl"
V22 = ROOT / "semantic_causal_alignment_v2_2_full/outputs/semantic_causal_aligned_pairs_v2_2_full.jsonl"


def rows(path: Path) -> dict[str, dict]:
    return {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        )
    }


def main() -> None:
    v1 = rows(V1)
    v22 = rows(V22)
    shared = set(v1) & set(v22)
    mismatches = []
    for parent_id in sorted(shared):
        left, right = v1[parent_id], v22[parent_id]
        if (
            left["source_tokens"] != right["source_tokens"]
            or left["target_pseudo_asl_gloss_tokens"]
            != right["target_pseudo_asl_gloss_tokens"]
        ):
            mismatches.append(parent_id)
    report = {
        "v1_split_counts": dict(Counter(row.get("split") for row in v1.values())),
        "v2_2_split_counts": dict(Counter(row.get("split") for row in v22.values())),
        "shared_count": len(shared),
        "text_or_gloss_mismatch_count": len(mismatches),
        "v1_only_validation_ids": sorted(
            parent_id
            for parent_id, row in v1.items()
            if row.get("split") == "validation" and parent_id not in v22
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if mismatches:
        raise SystemExit("Shared v1/v2.2 text or pseudo gloss mismatch")


if __name__ == "__main__":
    main()
