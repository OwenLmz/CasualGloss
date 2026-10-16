#!/usr/bin/env python3
"""Audit v2.2 recovery manifests and free-running rollout failures."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(
    "/123090274/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train"
)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def main() -> None:
    reports = [
        ROOT
        / "fine_tuning_v2_semantic_v2_2_full/reports/"
        "delay_recovery_v2_2_1_build_report.json",
        ROOT
        / "fine_tuning_v2_semantic_v2_2_full/reports/"
        "recovery_calibration_v2_2_2_build_report.json",
    ]
    print("===== RECOVERY REPORTS =====")
    for path in reports:
        if path.exists():
            print(path)
            print(path.read_text(encoding="utf-8"))

    print("===== V2.2 ROLLOUTS =====")
    candidates = sorted(ROOT.glob("**/sentence_rollouts.jsonl"))
    selected = [
        path
        for path in candidates
        if "v2_2" in str(path)
        and ("locked" in str(path) or "validation" in str(path))
    ]
    for path in selected:
        rows = load_jsonl(path)
        terminations = Counter(row.get("termination_reason") for row in rows)
        empty = [row for row in rows if not row.get("predicted_gloss_tokens")]
        completion = sum(bool(row.get("completed_after_full_source")) for row in rows)
        print(path)
        print(
            json.dumps(
                {
                    "row_count": len(rows),
                    "termination_counts": dict(terminations),
                    "empty_prediction_count": len(empty),
                    "completion_rate": completion / len(rows) if rows else None,
                    "empty_source_length_counts": dict(
                        sorted(Counter(len(row.get("source_tokens", [])) for row in empty).items())
                    ),
                    "empty_examples": [
                        {
                            "id": row.get("id"),
                            "source_text_en": row.get("source_text_en"),
                            "source_token_count": len(row.get("source_tokens", [])),
                            "termination_reason": row.get("termination_reason"),
                            "action_class_counts": row.get("action_class_counts"),
                        }
                        for row in empty[:10]
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
