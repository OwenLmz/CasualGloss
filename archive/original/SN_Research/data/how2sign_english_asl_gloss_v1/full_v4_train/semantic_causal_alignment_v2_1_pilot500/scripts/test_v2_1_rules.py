#!/usr/bin/env python3
"""Dependency-free regression checks for the v2.1 lexical constraints."""

from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("finalize_semantic_alignment_v2_1.py")
SPEC = importlib.util.spec_from_file_location("alignment_v2_1", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ALIGNMENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ALIGNMENT)


def main() -> None:
    anchors, report = ALIGNMENT.occurrence_aware_anchors(
        ["ROOT", "COLOR", "ROOT", "COLOR"],
        ["root", "color", "then", "root", "color", "."],
    )
    assert [item["source_position"] for item in anchors] == [1, 2, 4, 5]
    assert report["monotonic_coverage"] == 1.0

    assert ALIGNMENT.match_strength("NOT", "don't") == 3
    assert ALIGNMENT.match_strength("EYE", "eying") > 0
    assert ALIGNMENT.source_is_corrupt("This is a valid English sentence.") is False
    assert ALIGNMENT.source_is_corrupt("\u8292\u804f\u8076 broken source") is True

    items = [
        {"source_position": 2, "match_method": "occurrence_aware_exact"},
        {"source_position": None, "match_method": "unmatched"},
        {"source_position": 6, "match_method": "occurrence_aware_exact"},
    ]
    ALIGNMENT.infer_missing_anchors(items, 8)
    assert [item["source_position"] for item in items] == [2, 4, 6]
    assert items[1]["anchor_is_lexical"] is False
    print("v2.1 rule regression checks: PASS")


if __name__ == "__main__":
    main()
