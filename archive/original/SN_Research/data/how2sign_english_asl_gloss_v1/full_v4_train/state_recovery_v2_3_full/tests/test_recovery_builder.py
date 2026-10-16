#!/usr/bin/env python3
"""Small deterministic tests for the v2.3 recovery-state builder."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/prepare_v2_3_manifests.py"
SPEC = importlib.util.spec_from_file_location("prepare_v2_3", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def pair(source_count: int, target_count: int) -> dict:
    return {
        "id": "fixture",
        "source_tokens": [f"s{index}" for index in range(source_count)],
        "target_pseudo_asl_gloss_tokens": [f"G{index}" for index in range(target_count)],
        "target_pseudo_asl_gloss": " ".join(f"G{index}" for index in range(target_count)),
        "semantic_gloss_token_alignment": [
            {
                "scheduled_emit_source_position": min(source_count, index + 1),
                "v2_2_emit_source_position": min(source_count, index + 1),
            }
            for index in range(target_count)
        ],
    }


def main() -> None:
    rows = MODULE.build_recovery_rows(pair(6, 5))
    assert rows
    assert len({row["id"] for row in rows}) == len(rows)
    assert any(
        row["augmentation_type"] == "full_source_empty_history"
        and row["source_complete"]
        and row["committed_pseudo_asl_gloss"] == ""
        and row["target_action"] == "G0"
        for row in rows
    )
    assert any(row["augmentation_type"] == "prefix_burst_catchup" for row in rows)
    for row in rows:
        assert row["target_action"] not in {"<WAIT>", "<END>"}
        assert row["target_is_gold"] is False
        assert row["alignment_is_gold"] is False
        assert 1 <= row["source_prefix_length"] <= row["source_token_count"]

    short_rows = MODULE.build_recovery_rows(pair(2, 2))
    assert any(row["source_complete"] and row["target_action"] == "G0" for row in short_rows)
    assert any(row["source_complete"] and row["target_action"] == "G1" for row in short_rows)

    one_row = MODULE.build_recovery_rows(pair(1, 1))
    assert len(one_row) == 1
    assert one_row[0]["target_action"] == "G0"
    print("v2.3 recovery builder tests passed")


if __name__ == "__main__":
    main()
