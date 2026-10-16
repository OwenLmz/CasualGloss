#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from transllama_data import write_json


NAMES = ("screen_lr_1e-5", "screen_lr_3e-5", "screen_lr_6e-5")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    candidates = []
    for name in NAMES:
        training = json.loads(
            (root / "runs" / "lr_screen" / name / "training_summary.json").read_text(
                encoding="utf-8"
            )
        )
        config = json.loads((root / "configs" / f"{name}.json").read_text(encoding="utf-8"))
        sweep = json.loads(
            (root / "runs" / "lr_screen_dev" / name / "sweep_summary.json").read_text(
                encoding="utf-8"
            )
        )
        results = sweep["results"]
        gate_checks = {
            "eight_predeclared_k_values": len(results) == 8
            and [row["wait_k"] for row in results] == list(range(1, 9)),
            "all_have_500_parents": all(row["parent_count"] == 500 for row in results),
            "completion_at_least_0.95": all(row["completion_rate"] >= 0.95 for row in results),
            "max_action_rate_at_most_0.05": all(
                row["maximum_actions_reached_rate"] <= 0.05 for row in results
            ),
            "invalid_action_ratio_at_most_0.05": all(
                row["mean_invalid_action_ratio"] <= 0.05 for row in results
            ),
        }
        candidates.append(
            {
                "name": name,
                "learning_rate": config["learning_rate"],
                "best_validation_target_only_loss": training["best_checkpoint"][
                    "validation_target_only_loss"
                ],
                "best_checkpoint": training["best_checkpoint"],
                "sanity_gate": gate_checks,
                "passed_sanity_gate": all(gate_checks.values()),
                "dev_results": results,
            }
        )

    passing = [candidate for candidate in candidates if candidate["passed_sanity_gate"]]
    selection = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_split": "validation_only",
        "locked_test_accessed": False,
        "selection_rule": (
            "lowest all-prefix target-only validation loss among candidates passing the "
            "predeclared free-running sanity gate"
        ),
        "candidates": candidates,
        "selected": None,
    }
    if passing:
        selection["selected"] = min(
            passing, key=lambda candidate: candidate["best_validation_target_only_loss"]
        )["name"]
    write_json(root / "runs" / "lr_screen_selection.json", selection)
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    if not passing:
        raise SystemExit("no learning-rate candidate passed the frozen sanity gate")


if __name__ == "__main__":
    main()
