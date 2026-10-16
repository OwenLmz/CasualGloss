#!/usr/bin/env python3
"""Deterministic checks for speech-time latency and action semantics."""

from __future__ import annotations

from run_speech_to_gloss_e2e import (
    action_class,
    average_lagging,
    average_proportion,
    differentiable_average_lagging,
    length_adaptive_average_lagging,
    serial_processing_interval,
    state_prompt,
)


def close(actual, expected, tolerance=1e-9):
    assert actual is not None
    assert abs(actual - expected) <= tolerance, (actual, expected)


def main() -> None:
    delays = [1.0, 2.0, 3.0, 4.0]
    close(average_lagging(delays, 4.0, 4), 1.0)
    close(length_adaptive_average_lagging(delays, 4.0, 4), 1.0)
    close(average_proportion(delays, 4.0), 0.625)
    close(differentiable_average_lagging(delays, 4.0, 4), 1.0)

    start, ready = serial_processing_interval(1.0, 0.0, 0.4)
    close(start, 1.0)
    close(ready, 1.4)
    start, ready = serial_processing_interval(1.1, ready, 0.5)
    close(start, 1.4)
    close(ready, 1.9)

    assert action_class("<WAIT>", "<WAIT>", "<END>") == "WAIT"
    assert action_class("<END>", "<WAIT>", "<END>") == "END"
    assert action_class("WHITE", "<WAIT>", "<END>") == "WRITE"
    assert action_class("two tokens", "<WAIT>", "<END>") == "INVALID"
    prompt = state_prompt(["we", "'re", "ready"], ["WE"], False)
    assert "we're ready" in prompt
    assert "Source status: INCOMPLETE" in prompt
    print("end-to-end latency and action checks passed")


if __name__ == "__main__":
    main()
