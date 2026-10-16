#!/usr/bin/env python3
"""Small deterministic checks for the word-position latency metrics."""

from __future__ import annotations

from evaluate_free_running_rollout import (
    average_lagging,
    average_proportion,
    differentiable_average_lagging,
    finalize_state,
    length_adaptive_average_lagging,
    reference_average_lagging,
)


def close(actual: float | None, expected: float) -> None:
    assert actual is not None
    assert abs(actual - expected) < 1e-8, (actual, expected)


def main() -> None:
    # One READ before every WRITE: g(i) = [1, 2, 3, 4].
    positions = [1, 2, 3, 4]
    al, normalized = average_lagging(positions, source_count=4)
    close(al, 1.0)
    close(normalized, 0.25)
    close(reference_average_lagging(positions, 4, 4), 1.0)
    close(length_adaptive_average_lagging(positions, 4, 4), 1.0)
    close(average_proportion(positions, 4), 0.625)
    close(differentiable_average_lagging(positions, 4, 4), 1.0)

    # Two-token bursts at prefixes 1 and 2 are earlier than the wait-1 path.
    burst = [1, 1, 2, 2]
    burst_al, burst_normalized = average_lagging(burst, source_count=4)
    close(burst_al, 0.0)
    close(burst_normalized, 0.0)
    close(reference_average_lagging(burst, 4, 4), 0.0)
    close(length_adaptive_average_lagging(burst, 4, 4), 0.0)
    close(average_proportion(burst, 4), 0.375)
    close(differentiable_average_lagging(burst, 4, 4), 1.0)

    # A wait-until-complete policy has substantially higher lag.
    full_sentence = [4, 4, 4, 4]
    full_al, full_normalized = average_lagging(full_sentence, source_count=4)
    close(full_al, 4.0)
    close(full_normalized, 1.0)
    close(reference_average_lagging(full_sentence, 4, 4), 4.0)
    close(length_adaptive_average_lagging(full_sentence, 4, 4), 4.0)
    close(average_proportion(full_sentence, 4), 1.0)
    close(differentiable_average_lagging(full_sentence, 4, 4), 4.0)

    # Any END is retained for backwards compatibility, but only an END after
    # the complete source is a valid completion for model selection.
    state = {
        "id": "completion-test",
        "source_text_en": "one two",
        "source_tokens": ["one", "two"],
        "target_tokens": ["ONE"],
        "prefix_length": 1,
        "committed": [],
        "write_positions": [],
        "events": [],
        "termination_reason": "premature_end",
        "premature_end": True,
        "max_actions": 20,
    }
    premature = finalize_state(state, "test")
    assert premature["completed_with_end"] is True
    assert premature["completed_after_full_source"] is False

    state["prefix_length"] = 2
    state["termination_reason"] = "end_after_full_source"
    state["premature_end"] = False
    valid = finalize_state(state, "test")
    assert valid["completed_with_end"] is True
    assert valid["completed_after_full_source"] is True

    print("latency and completion-semantics checks passed")


if __name__ == "__main__":
    main()
