#!/usr/bin/env python3
"""Small deterministic checks for the v2.3 local-agreement policy."""

from __future__ import annotations

from evaluate_local_agreement_policy import evaluate_parent, longest_common_prefix


def average_lagging(positions, source_count):
    return (sum(positions) / len(positions), sum(positions) / len(positions) / source_count)


def scalar(*_args):
    return 1.0


def overlap(target, prediction):
    value = float(target == prediction)
    return value, value, value


METRICS = {
    "average_lagging": average_lagging,
    "reference_average_lagging": scalar,
    "length_adaptive_average_lagging": scalar,
    "average_proportion": scalar,
    "differentiable_average_lagging": scalar,
    "token_overlap": overlap,
}


def rows(predictions):
    return [
        {
            "parent_id": "example",
            "split": "validation",
            "source_position": index,
            "source_token_count": len(predictions),
            "source_text_en": "one two three four",
            "source_tokens": ["one", "two", "three", "four"],
            "reference_pseudo_asl_gloss_tokens": ["A", "B", "C"],
            "predicted_pseudo_asl_gloss_tokens": prediction,
        }
        for index, prediction in enumerate(predictions, 1)
    ]


def main() -> None:
    assert longest_common_prefix([["A", "B"], ["A", "C"]]) == ["A"]
    stable = evaluate_parent(
        rows([["A"], ["A", "B"], ["A", "B"], ["A", "B", "C"]]),
        stability_window=2,
        minimum_prefix_tokens=1,
        latency_functions=METRICS,
    )
    assert stable["predicted_tokens"] == ["A", "B", "C"]
    assert stable["write_positions"] == [2, 3, 4]
    assert stable["irreversible_prefix_conflict_count"] == 0
    conflict = evaluate_parent(
        rows([["A"], ["A", "B"], ["X"], ["X", "C"]]),
        stability_window=2,
        minimum_prefix_tokens=1,
        latency_functions=METRICS,
    )
    assert conflict["predicted_tokens"] == ["A", "X", "C"]
    assert conflict["irreversible_prefix_conflict_count"] > 0
    print("local-agreement policy checks passed")


if __name__ == "__main__":
    main()
