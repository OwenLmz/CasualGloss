#!/usr/bin/env python3
"""Audit label and rollout latency without changing any trained model.

The existing project reports a sentence-level AL divided by source length.  This
script keeps that metric for backwards compatibility and additionally reports
reference-length AL, LAAL, AP, and DAL.  It also separates semantic-required
positions from positions delayed by the one-WRITE-per-source-prefix scheduler.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return round(mean(clean), 8) if clean else None


def safe_median(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return round(median(clean), 8) if clean else None


def percentile(values: Iterable[float | None], q: float) -> float | None:
    clean = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not clean:
        return None
    index = (len(clean) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    value = clean[lower] if lower == upper else clean[lower] * (upper - index) + clean[upper] * (index - lower)
    return round(value, 8)


def al(delays: list[int], source_length: int, reference_length: int) -> float | None:
    if not delays or source_length <= 0 or reference_length <= 0:
        return None
    gamma = reference_length / source_length
    tau = next((index for index, delay in enumerate(delays, 1) if delay >= source_length), len(delays))
    return mean(delays[index - 1] - (index - 1) / gamma for index in range(1, tau + 1))


def laal(delays: list[int], source_length: int, reference_length: int) -> float | None:
    if not delays or source_length <= 0 or reference_length <= 0:
        return None
    adaptive_length = max(reference_length, len(delays))
    gamma = adaptive_length / source_length
    tau = next((index for index, delay in enumerate(delays, 1) if delay >= source_length), len(delays))
    return mean(delays[index - 1] - (index - 1) / gamma for index in range(1, tau + 1))


def current_project_al(delays: list[int], source_length: int) -> tuple[float | None, float | None]:
    if not delays or source_length <= 0:
        return None, None
    value = al(delays, source_length, len(delays))
    return value, None if value is None else value / source_length


def ap(delays: list[int], source_length: int) -> float | None:
    if not delays or source_length <= 0:
        return None
    return sum(delays) / (source_length * len(delays))


def dal(delays: list[int], source_length: int, reference_length: int) -> float | None:
    if not delays or source_length <= 0 or reference_length <= 0:
        return None
    gamma = reference_length / source_length
    adjusted = [float(delays[0])]
    for delay in delays[1:]:
        adjusted.append(max(float(delay), adjusted[-1] + 1.0 / gamma))
    return mean(value - index / gamma for index, value in enumerate(adjusted))


def latency_record(delays: list[int], source_length: int, reference_length: int) -> dict[str, float | int | None]:
    project_al, project_nal = current_project_al(delays, source_length)
    return {
        "source_length": source_length,
        "reference_length": reference_length,
        "prediction_length": len(delays),
        "project_al_prediction_length": project_al,
        "project_normalized_al": project_nal,
        "reference_length_al": al(delays, source_length, reference_length),
        "laal": laal(delays, source_length, reference_length),
        "ap": ap(delays, source_length),
        "dal": dal(delays, source_length, reference_length),
        "full_source_write_ratio": sum(value >= source_length for value in delays) / max(1, len(delays)),
        "first_write_position": delays[0] if delays else None,
    }


def summarize(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"count": len(rows)}
    for key in keys:
        values = [row.get(key) for row in rows]
        result[key] = {
            "mean": safe_mean(values),
            "median": safe_median(values),
            "p90": percentile(values, 0.90),
            "p95": percentile(values, 0.95),
        }
    return result


def rollout_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    output = []
    source = {}
    for row in read_jsonl(path):
        delays = [int(value) for value in row.get("write_source_positions", [])]
        source_length = int(row["source_token_count"])
        reference_length = int(row["target_gloss_token_count"])
        record = {
            "id": row["id"],
            **latency_record(delays, source_length, reference_length),
            "sequence_similarity": float(row["pseudo_gloss_sequence_similarity"]),
            "token_f1": float(row["pseudo_gloss_token_f1"]),
            "exact_match": bool(row["pseudo_gloss_exact_match"]),
            "completed_with_end": bool(row.get("completed_with_end", False)),
            "completed_after_full_source": (
                row.get("completed_after_full_source")
                if row.get("completed_after_full_source") is not None
                else row["termination_reason"] == "end_after_full_source"
            ),
            "wait_ratio": float(row["wait_action_ratio"]),
            "invalid_ratio": float(row["invalid_action_ratio"]),
            "premature_end": bool(row["premature_end"]),
            "maximum_actions_reached": row["termination_reason"] == "maximum_actions_reached",
            "prediction_reference_length_absolute_error": abs(
                len(delays) / max(1, reference_length) - 1.0
            ),
            "termination_reason": row["termination_reason"],
        }
        output.append(record)
        source[row["id"]] = row
    return output, source


def action_label_rows(path: Path) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        grouped[row["parent_id"]].append(row)
    output = []
    for parent_id, actions in grouped.items():
        actions.sort(key=lambda row: int(row.get("action_index", str(row.get("id", "")).rsplit("_", 1)[-1])))
        writes = [row for row in actions if str(row.get("target_action_type", row.get("action_type", ""))).startswith("WRITE")]
        delays = [int(row["source_prefix_length"]) for row in writes]
        source_length = int(actions[0]["source_token_count"])
        reference_length = len(writes)
        output.append({"id": parent_id, **latency_record(delays, source_length, reference_length)})
    return output


def semantic_pair_rows(path: Path) -> list[dict[str, Any]]:
    output = []
    for row in read_jsonl(path):
        if not row.get("eligible_for_finetune2_v2_1", True):
            continue
        source_length = len(row["source_tokens"])
        reference_length = len(row["target_pseudo_asl_gloss_tokens"])
        required = [int(value) for value in row.get("semantic_required_source_positions", [])]
        scheduled = [int(value) for value in row.get("scheduled_emit_source_positions", [])]
        if not required:
            required = [int(item["semantic_required_source_position"]) for item in row["semantic_gloss_token_alignment"]]
        if not scheduled:
            scheduled = [int(item["scheduled_emit_source_position"]) for item in row["semantic_gloss_token_alignment"]]
        required_metrics = latency_record(required, source_length, reference_length)
        scheduled_metrics = latency_record(scheduled, source_length, reference_length)
        output.append(
            {
                "id": row["id"],
                "required_project_normalized_al": required_metrics["project_normalized_al"],
                "scheduled_project_normalized_al": scheduled_metrics["project_normalized_al"],
                "scheduler_nal_penalty": (
                    None
                    if required_metrics["project_normalized_al"] is None or scheduled_metrics["project_normalized_al"] is None
                    else scheduled_metrics["project_normalized_al"] - required_metrics["project_normalized_al"]
                ),
                "required_laal": required_metrics["laal"],
                "scheduled_laal": scheduled_metrics["laal"],
                "scheduler_laal_penalty": (
                    None
                    if required_metrics["laal"] is None or scheduled_metrics["laal"] is None
                    else scheduled_metrics["laal"] - required_metrics["laal"]
                ),
                "same_prefix_required_write_count": sum(
                    required[index] == required[index - 1] for index in range(1, len(required))
                ),
                "tokens_shifted_by_scheduler": sum(a != b for a, b in zip(required, scheduled)),
                "mean_scheduler_shift_tokens": mean([b - a for a, b in zip(required, scheduled)]) if required else 0.0,
            }
        )
    return output


def paired_summary(v1: list[dict[str, Any]], v2: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    v1_by_id = {row["id"]: row for row in v1}
    v2_by_id = {row["id"]: row for row in v2}
    paired = []
    for parent_id in sorted(v1_by_id.keys() & v2_by_id.keys()):
        left, right = v1_by_id[parent_id], v2_by_id[parent_id]
        record = {"id": parent_id}
        for key in [
            "project_normalized_al",
            "reference_length_al",
            "laal",
            "ap",
            "dal",
            "sequence_similarity",
            "token_f1",
            "wait_ratio",
            "full_source_write_ratio",
        ]:
            record[f"v1_{key}"] = left[key]
            record[f"v2_1_{key}"] = right[key]
            record[f"delta_{key}"] = (
                None if right.get(key) is None or left.get(key) is None else right[key] - left[key]
            )
        record["v1_prediction_length"] = left["prediction_length"]
        record["v2_1_prediction_length"] = right["prediction_length"]
        paired.append(record)
    metrics = [key.removeprefix("delta_") for key in paired[0] if key.startswith("delta_")] if paired else []
    return (
        {
            "paired_count": len(paired),
            "mean_v2_1_minus_v1": {key: safe_mean(row[f"delta_{key}"] for row in paired) for key in metrics},
            "v2_1_better_count": {
                key: sum(
                    False
                    if row[f"delta_{key}"] is None
                    else (
                        row[f"delta_{key}"] < 0
                        if key in {"project_normalized_al", "reference_length_al", "laal", "ap", "dal", "wait_ratio", "full_source_write_ratio"}
                        else row[f"delta_{key}"] > 0
                    )
                    for row in paired
                )
                for key in metrics
            },
        },
        paired,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-rollouts", type=Path, required=True)
    parser.add_argument("--v2-rollouts", type=Path, required=True)
    parser.add_argument("--v1-actions", type=Path)
    parser.add_argument("--v2-actions", type=Path)
    parser.add_argument("--v2-pairs", type=Path)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-paired-tsv", type=Path, required=True)
    args = parser.parse_args()

    v1, _ = rollout_rows(args.v1_rollouts)
    v2, _ = rollout_rows(args.v2_rollouts)
    summary_keys = [
        "project_normalized_al",
        "reference_length_al",
        "laal",
        "ap",
        "dal",
        "full_source_write_ratio",
        "sequence_similarity",
        "token_f1",
        "wait_ratio",
    ]
    paired_stats, paired = paired_summary(v1, v2)
    report: dict[str, Any] = {
        "scientific_status": "latency_metric_and_policy_audit_not_gold_asl",
        "metric_note": (
            "project_normalized_al is the legacy prediction-length AL divided by source length; "
            "reference_length_al and LAAL follow standard simultaneous-translation conventions."
        ),
        "v1_rollout": summarize(v1, summary_keys),
        "v2_1_rollout": summarize(v2, summary_keys),
        "paired_rollout": paired_stats,
        "termination_counts": {
            "v1": Counter(row["termination_reason"] for row in v1),
            "v2_1": Counter(row["termination_reason"] for row in v2),
        },
    }
    if args.v1_actions:
        rows = action_label_rows(args.v1_actions)
        report["v1_label_policy"] = summarize(rows, ["project_normalized_al", "reference_length_al", "laal", "ap", "dal"])
    if args.v2_actions:
        rows = action_label_rows(args.v2_actions)
        report["v2_1_label_policy"] = summarize(rows, ["project_normalized_al", "reference_length_al", "laal", "ap", "dal"])
    if args.v2_pairs:
        rows = semantic_pair_rows(args.v2_pairs)
        report["v2_1_scheduler_decomposition"] = summarize(
            rows,
            [
                "required_project_normalized_al",
                "scheduled_project_normalized_al",
                "scheduler_nal_penalty",
                "required_laal",
                "scheduled_laal",
                "scheduler_laal_penalty",
                "same_prefix_required_write_count",
                "tokens_shifted_by_scheduler",
                "mean_scheduler_shift_tokens",
            ],
        )
        report["v2_1_scheduler_decomposition"]["parents_with_scheduler_shift"] = sum(
            row["tokens_shifted_by_scheduler"] > 0 for row in rows
        )
    write_json(args.output_report, report)
    args.output_paired_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_paired_tsv.open("w", encoding="utf-8", newline="") as handle:
        if paired:
            writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
