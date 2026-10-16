#!/usr/bin/env python3
"""Evaluate irreversible gloss commits from stable Fine-tune 1 prefix translations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean, median
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_mean(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return round(mean(present), 8) if present else None


def longest_common_prefix(sequences: list[list[str]]) -> list[str]:
    if not sequences:
        return []
    limit = min(len(sequence) for sequence in sequences)
    index = 0
    while index < limit and len({sequence[index] for sequence in sequences}) == 1:
        index += 1
    return sequences[0][:index]


def evaluate_parent(
    rows: list[dict[str, Any]],
    stability_window: int,
    minimum_prefix_tokens: int,
    latency_functions: dict[str, Any],
) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: int(row["source_position"]))
    first = rows[0]
    source_count = int(first["source_token_count"])
    expected_positions = list(range(1, source_count + 1))
    actual_positions = [int(row["source_position"]) for row in rows]
    if actual_positions != expected_positions:
        raise ValueError(
            f"Incomplete prefix sequence for {first['parent_id']}: "
            f"expected {expected_positions[:3]}...{expected_positions[-3:]}, "
            f"got {actual_positions[:3]}...{actual_positions[-3:]}"
        )

    reference = [str(token) for token in first["reference_pseudo_asl_gloss_tokens"]]
    predictions = [
        [str(token) for token in row["predicted_pseudo_asl_gloss_tokens"]]
        for row in rows
    ]
    committed: list[str] = []
    write_positions: list[int] = []
    events: list[dict[str, Any]] = []
    conflict_count = 0
    revision_similarities = []

    for index, (row, prediction) in enumerate(zip(rows, predictions)):
        position = int(row["source_position"])
        if index:
            revision_similarities.append(
                SequenceMatcher(None, predictions[index - 1], prediction).ratio()
            )
        new_tokens: list[str] = []
        stable_prefix: list[str] = []
        if position >= minimum_prefix_tokens and index + 1 >= stability_window:
            stable_prefix = longest_common_prefix(
                predictions[index + 1 - stability_window : index + 1]
            )
            if stable_prefix[: len(committed)] == committed:
                new_tokens = stable_prefix[len(committed) :]
            elif committed[: len(stable_prefix)] != stable_prefix:
                conflict_count += 1
        for token in new_tokens:
            committed.append(token)
            write_positions.append(position)
            events.append(
                {
                    "source_position": position,
                    "action": token,
                    "action_type": "WRITE_STABLE_PREFIX",
                    "stability_window": stability_window,
                }
            )
        if position < source_count:
            events.append(
                {
                    "source_position": position,
                    "action": "<WAIT>",
                    "action_type": "READ_WAIT",
                }
            )

    final_prediction = predictions[-1]
    final_lcp = longest_common_prefix([committed, final_prediction])
    if final_prediction[: len(committed)] == committed:
        final_suffix = final_prediction[len(committed) :]
        final_merge = "append_uncommitted_final_suffix"
    else:
        conflict_count += 1
        final_suffix = final_prediction[len(final_lcp) :]
        final_merge = "irreversible_conflict_append_after_common_prefix"
    for token in final_suffix:
        committed.append(token)
        write_positions.append(source_count)
        events.append(
            {
                "source_position": source_count,
                "action": token,
                "action_type": "WRITE_AT_SOURCE_END",
                "final_merge": final_merge,
            }
        )
    events.append(
        {
            "source_position": source_count,
            "action": "<END>",
            "action_type": "END",
        }
    )

    precision, recall, f1 = latency_functions["token_overlap"](reference, committed)
    sequence_similarity = SequenceMatcher(None, reference, committed).ratio()
    al, project_nal = latency_functions["average_lagging"](
        write_positions, source_count
    )
    reference_al = latency_functions["reference_average_lagging"](
        write_positions, source_count, len(reference)
    )
    laal = latency_functions["length_adaptive_average_lagging"](
        write_positions, source_count, len(reference)
    )
    ap = latency_functions["average_proportion"](write_positions, source_count)
    dal = latency_functions["differentiable_average_lagging"](
        write_positions, source_count, len(reference)
    )
    return {
        "id": first["parent_id"],
        "split": first["split"],
        "source_text_en": first["source_text_en"],
        "source_tokens": first["source_tokens"],
        "reference_pseudo_asl_gloss": " ".join(reference),
        "predicted_streaming_pseudo_asl_gloss": " ".join(committed),
        "full_source_finetune1_prediction": " ".join(final_prediction),
        "reference_tokens": reference,
        "predicted_tokens": committed,
        "write_positions": write_positions,
        "write_source_positions": write_positions,
        "events": events,
        "source_token_count": source_count,
        "target_gloss_token_count": len(reference),
        "predicted_gloss_token_count": len(committed),
        "pseudo_gloss_sequence_similarity": round(sequence_similarity, 8),
        "pseudo_gloss_token_f1": round(f1, 8),
        "pseudo_gloss_exact_match": committed == reference,
        "completed_with_end": True,
        "completed_after_full_source": True,
        "premature_end": False,
        "termination_reason": "end_after_full_source",
        "invalid_output_count": 0,
        "invalid_action_ratio": 0.0,
        "wait_action_ratio": round(
            sum(event["action_type"] == "READ_WAIT" for event in events)
            / max(1, len(events)),
            8,
        ),
        "irreversible_prefix_conflict_count": conflict_count,
        "final_merge_method": final_merge,
        "mean_adjacent_prefix_prediction_similarity": round(
            mean(revision_similarities), 8
        )
        if revision_similarities
        else None,
        "pseudo_label_sequence_exact_match": committed == reference,
        "pseudo_label_sequence_similarity": round(sequence_similarity, 8),
        "pseudo_label_token_precision": round(precision, 8),
        "pseudo_label_token_recall": round(recall, 8),
        "pseudo_label_token_f1": round(f1, 8),
        "average_lagging_source_tokens": al,
        "normalized_average_lagging": project_nal,
        "reference_length_average_lagging_source_tokens": reference_al,
        "length_adaptive_average_lagging_source_tokens": laal,
        "average_proportion": ap,
        "differentiable_average_lagging_source_tokens": dal,
        "stability_window": stability_window,
        "minimum_prefix_tokens": minimum_prefix_tokens,
        "target_is_gold": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--causal-scripts", type=Path, required=True)
    parser.add_argument("--stability-window", type=int, required=True)
    parser.add_argument("--minimum-prefix-tokens", type=int, default=1)
    parser.add_argument("--output-rollouts", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.stability_window < 1 or args.minimum_prefix_tokens < 1:
        raise SystemExit("stability-window and minimum-prefix-tokens must be positive")

    sys.path.insert(0, str(args.causal_scripts.resolve()))
    import evaluate_free_running_rollout as metrics  # noqa: PLC0415

    latency_functions = {
        name: getattr(metrics, name)
        for name in (
            "average_lagging",
            "reference_average_lagging",
            "length_adaptive_average_lagging",
            "average_proportion",
            "differentiable_average_lagging",
            "token_overlap",
        )
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.predictions):
        grouped[str(row["parent_id"])].append(row)
    rollouts = [
        evaluate_parent(
            rows,
            args.stability_window,
            args.minimum_prefix_tokens,
            latency_functions,
        )
        for _, rows in sorted(grouped.items())
    ]
    write_jsonl(args.output_rollouts, rollouts)

    conflict_counts = Counter(
        "has_conflict" if row["irreversible_prefix_conflict_count"] else "no_conflict"
        for row in rollouts
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "finetune1_retranslation_local_agreement_v2_3",
        "scientific_status": "evaluation_against_pseudo_gloss_not_gold_asl",
        "split": rollouts[0]["split"] if rollouts else None,
        "parameters": {
            "stability_window": args.stability_window,
            "minimum_prefix_tokens": args.minimum_prefix_tokens,
        },
        "parent_count": len(rollouts),
        "valid_completion_ratio": safe_mean(
            [float(row["completed_after_full_source"]) for row in rollouts]
        ),
        "pseudo_label_sequence_exact_match": safe_mean(
            [float(row["pseudo_label_sequence_exact_match"]) for row in rollouts]
        ),
        "pseudo_label_sequence_similarity": safe_mean(
            [row["pseudo_label_sequence_similarity"] for row in rollouts]
        ),
        "pseudo_label_token_f1": safe_mean(
            [row["pseudo_label_token_f1"] for row in rollouts]
        ),
        "mean_reference_length_average_lagging_source_tokens": safe_mean(
            [row["reference_length_average_lagging_source_tokens"] for row in rollouts]
        ),
        "mean_length_adaptive_average_lagging_source_tokens": safe_mean(
            [row["length_adaptive_average_lagging_source_tokens"] for row in rollouts]
        ),
        "mean_normalized_average_lagging": safe_mean(
            [row["normalized_average_lagging"] for row in rollouts]
        ),
        "mean_average_proportion": safe_mean(
            [row["average_proportion"] for row in rollouts]
        ),
        "mean_differentiable_average_lagging_source_tokens": safe_mean(
            [row["differentiable_average_lagging_source_tokens"] for row in rollouts]
        ),
        "mean_adjacent_prefix_prediction_similarity": safe_mean(
            [row["mean_adjacent_prefix_prediction_similarity"] for row in rollouts]
        ),
        "parents_with_irreversible_conflict": conflict_counts["has_conflict"],
        "irreversible_conflict_parent_ratio": round(
            conflict_counts["has_conflict"] / max(1, len(rollouts)), 8
        ),
        "median_first_write_source_position": round(
            median(
                row["write_positions"][0]
                for row in rollouts
                if row["write_positions"]
            ),
            8,
        )
        if any(row["write_positions"] for row in rollouts)
        else None,
        "important_limitations": [
            "The reference is Qwen-generated pseudo gloss, not expert gold ASL gloss.",
            "The policy is irreversible; conflicts with later prefix translations are retained and reported.",
            "Policy parameters must be selected on validation only before one locked test evaluation.",
        ],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
