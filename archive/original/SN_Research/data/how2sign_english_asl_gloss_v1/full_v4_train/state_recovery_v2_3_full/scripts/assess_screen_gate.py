#!/usr/bin/env python3
"""Select a v2.3 development checkpoint and apply the frozen screen gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def score(metrics: dict) -> tuple[float, float, float]:
    laal = metrics.get("mean_length_adaptive_average_lagging_source_tokens")
    return (
        float(metrics["mean_pseudo_gloss_token_f1"]),
        float(metrics["mean_pseudo_gloss_sequence_similarity"]),
        -float(laal if laal is not None else 1e9),
    )


def eligible(metrics: dict) -> bool:
    return (
        float(metrics["valid_end_completion_rate"]) >= 0.99
        and float(metrics["mean_invalid_action_ratio"]) <= 0.01
        and float(metrics["maximum_actions_reached_rate"]) <= 0.01
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True, help="name=metrics.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = load(args.baseline)
    candidates = {}
    for item in args.candidate:
        name, path = item.split("=", 1)
        candidates[name] = load(Path(path))
    eligible_names = [name for name, metrics in candidates.items() if eligible(metrics)]
    selected = max(eligible_names, key=lambda name: score(candidates[name])) if eligible_names else max(candidates, key=lambda name: score(candidates[name]))
    candidate = candidates[selected]

    differences = {
        "valid_end_completion_rate": float(candidate["valid_end_completion_rate"]) - float(baseline["valid_end_completion_rate"]),
        "maximum_actions_reached_rate": float(candidate["maximum_actions_reached_rate"]) - float(baseline["maximum_actions_reached_rate"]),
        "mean_pseudo_gloss_token_f1": float(candidate["mean_pseudo_gloss_token_f1"]) - float(baseline["mean_pseudo_gloss_token_f1"]),
        "mean_pseudo_gloss_sequence_similarity": float(candidate["mean_pseudo_gloss_sequence_similarity"]) - float(baseline["mean_pseudo_gloss_sequence_similarity"]),
        "mean_length_adaptive_average_lagging_source_tokens": float(candidate["mean_length_adaptive_average_lagging_source_tokens"]) - float(baseline["mean_length_adaptive_average_lagging_source_tokens"]),
    }
    recovery_improvement = (
        differences["valid_end_completion_rate"] >= 0.002
        or differences["maximum_actions_reached_rate"] <= -0.002
        or differences["mean_pseudo_gloss_token_f1"] >= 0.005
    )
    checks = {
        "completion": float(candidate["valid_end_completion_rate"]) >= 0.99,
        "invalid": float(candidate["mean_invalid_action_ratio"]) <= 0.01,
        "max_actions": float(candidate["maximum_actions_reached_rate"]) <= 0.01,
        "token_f1": differences["mean_pseudo_gloss_token_f1"] >= -0.005,
        "sequence_similarity": differences["mean_pseudo_gloss_sequence_similarity"] >= -0.005,
        "laal": differences["mean_length_adaptive_average_lagging_source_tokens"] <= 0.20,
        "recovery_improvement": recovery_improvement,
    }
    report = {
        "selected_candidate": selected,
        "selected_by": "frozen development rule: eligibility, token F1, sequence similarity, lower LAAL",
        "baseline": baseline,
        "candidates": candidates,
        "selected_minus_baseline": differences,
        "gate_checks": checks,
        "full_training_gate_passed": all(checks.values()),
        "important_note": "No confirmation, locked-test, or CLERC result was used by this decision.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
