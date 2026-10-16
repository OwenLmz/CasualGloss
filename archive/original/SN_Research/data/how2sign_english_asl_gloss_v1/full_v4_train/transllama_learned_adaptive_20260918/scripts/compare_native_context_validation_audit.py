#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


NAMES = ("screen_lr_1e-5", "screen_lr_3e-5", "screen_lr_6e-5")
IGNORED_RESULT_FIELDS = {"stage"}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": "pre-native-context validation versus native-context audit",
        "ignored_result_fields": sorted(IGNORED_RESULT_FIELDS),
        "candidates": [],
        "all_metrics_identical": True,
    }
    for name in NAMES:
        old_path = root / "runs" / "lr_screen_dev" / name / "sweep_summary.json"
        new_path = (
            root
            / "runs"
            / "lr_screen_dev_native_context_audit"
            / name
            / "sweep_summary.json"
        )
        old = json.loads(old_path.read_text(encoding="utf-8"))
        new = json.loads(new_path.read_text(encoding="utf-8"))
        differences = []
        for old_row, new_row in zip(old["results"], new["results"]):
            keys = (set(old_row) | set(new_row)) - IGNORED_RESULT_FIELDS
            for key in sorted(keys):
                if old_row.get(key) != new_row.get(key):
                    differences.append(
                        {
                            "wait_k": old_row["wait_k"],
                            "field": key,
                            "before": old_row.get(key),
                            "after": new_row.get(key),
                        }
                    )
        candidate = {
            "name": name,
            "result_count_before": len(old["results"]),
            "result_count_after": len(new["results"]),
            "native_inference_context_window": new["generation"][
                "native_inference_context_window"
            ],
            "max_observed_prompt_tokens": new["generation"][
                "max_observed_prompt_tokens"
            ],
            "differences": differences,
            "metrics_identical": not differences
            and len(old["results"]) == len(new["results"]),
        }
        report["candidates"].append(candidate)
        report["all_metrics_identical"] &= candidate["metrics_identical"]

    output_path = root / "runs" / "native_context_validation_audit_comparison.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_metrics_identical"]:
        raise SystemExit("native-context validation audit changed one or more metrics")


if __name__ == "__main__":
    main()
