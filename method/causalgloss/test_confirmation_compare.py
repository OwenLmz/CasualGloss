#!/usr/bin/env python3
"""Unit checks for paired confirmation statistics."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "compare_confirmation.py"
SPEC = importlib.util.spec_from_file_location("compare_confirmation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def row(f1: float, seq: float, laal: float, complete: bool, maximum: bool = False) -> dict:
    return {
        "pseudo_gloss_token_f1": f1,
        "pseudo_gloss_sequence_similarity": seq,
        "length_adaptive_average_lagging_source_tokens": laal,
        "completed_after_full_source": complete,
        "termination_reason": "maximum_actions_reached" if maximum else "end_after_full_source",
    }


def main() -> None:
    baseline = {"a": row(0.4, 0.5, 2.0, True), "b": row(0.6, 0.7, 3.0, False, True)}
    candidate = {"a": row(0.5, 0.6, 1.5, True), "b": row(0.8, 0.8, 2.5, True)}
    report = MODULE.comparison(baseline, candidate, repetitions=100, seed=7)
    assert report["parent_count"] == 2
    assert abs(report["metrics"]["token_f1"]["mean_difference"] - 0.15) < 1e-9
    assert report["metrics"]["laal_source_tokens"]["mean_difference"] == -0.5
    assert report["metrics"]["valid_end_completion"]["mean_difference"] == 0.5
    print("v2.3 confirmation comparison tests passed")


if __name__ == "__main__":
    main()
