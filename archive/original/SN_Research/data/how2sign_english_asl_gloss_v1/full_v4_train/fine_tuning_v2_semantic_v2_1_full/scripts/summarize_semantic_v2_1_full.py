#!/usr/bin/env python3
"""Summarize training and locked-test metrics for the v2.1 full run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--alignment-summary", type=Path, required=True)
    parser.add_argument("--quality-gate", type=Path, required=True)
    parser.add_argument("--action-metrics", type=Path, required=True)
    parser.add_argument("--rollout-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    history = read_jsonl(args.history)
    training = read_json(args.training_summary)
    alignment = read_json(args.alignment_summary)
    gate = read_json(args.quality_gate)
    action = read_json(args.action_metrics)
    rollout = read_json(args.rollout_metrics)
    validation_events = [row for row in history if row.get("event") == "validation"]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "pseudo_causal_action_lora_not_gold_asl",
        "alignment_quality_gate_passed": gate["quality_gate_passed"],
        "alignment": alignment,
        "training": training,
        "validation_curve": validation_events,
        "locked_test_gold_state_metrics": action,
        "locked_test_free_running_metrics": rollout,
        "interpretation": [
            "The run tests whether a model can imitate v2.1 pseudo causal actions on held-out How2Sign groups.",
            "Improvements against pseudo targets are not verified ASL translation accuracy.",
            "Qualified ASL evaluation remains required for linguistic claims."
        ]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "alignment_quality_gate_passed": report["alignment_quality_gate_passed"],
        "best_checkpoint": training.get("best_checkpoint"),
        "gold_state_metrics": action,
        "free_running_metrics": rollout,
    }, indent=2))


if __name__ == "__main__":
    main()
