#!/usr/bin/env python3
"""Small deterministic tests for the frozen Epee normalization and metrics."""

from compare_epee_multi_reference import edit_distance, score_references, token_f1
from prepare_epee_external_eval import collapse_exact_period, content_normalize


def main() -> None:
    collapsed, factor = collapse_exact_period(["GOOD", "MORNING", "GOOD", "MORNING"])
    assert collapsed == ["GOOD", "MORNING"] and factor == 2
    content, factor = content_normalize(["WHAT_2", "QUESTION", "WHAT_2", "QUESTION"])
    assert content == ["WHAT"] and factor == 2
    assert token_f1(["YOU", "LIVE"], ["LIVE", "YOU"]) == 1.0
    assert edit_distance(["YOU", "LIVE"], ["YOU", "WORK"]) == 1
    scores = score_references(
        ["YOU", "LIVE"], [["LIVE", "YOU"], ["YOU", "WORK"]]
    )
    assert scores["best_token_f1"] == 1.0
    assert scores["any_exact_match"] == 0.0
    print("Epee protocol tests passed")


if __name__ == "__main__":
    main()

