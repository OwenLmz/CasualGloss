#!/usr/bin/env python3
"""Fuse semantic prefix evidence with lexical anchors and build causal actions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from nltk.tokenize import TreebankWordDetokenizer


DETOKENIZER = TreebankWordDetokenizer()
LEXICAL_METHODS = {"lexical_exact", "lexical_stem"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_mean(values: list[float]) -> float | None:
    return round(mean(values), 6) if values else None


def paired_bootstrap_difference(
    differences: list[float], seed: int, resamples: int = 5000
) -> dict[str, Any]:
    if not differences:
        return {"paired_parent_count": 0, "mean_difference": None, "confidence_interval_95": None}
    generator = random.Random(seed)
    count = len(differences)
    estimates = []
    for _ in range(resamples):
        estimates.append(mean(differences[generator.randrange(count)] for _ in range(count)))
    estimates.sort()
    lower = estimates[int(0.025 * resamples)]
    upper = estimates[min(resamples - 1, int(0.975 * resamples))]
    return {
        "paired_parent_count": count,
        "resample_count": resamples,
        "seed": seed,
        "mean_difference": round(mean(differences), 6),
        "confidence_interval_95": [round(lower, 6), round(upper, 6)],
    }


def average_lagging(positions: list[int], source_count: int, target_count: int) -> tuple[float | None, float | None]:
    if not positions or source_count <= 0 or target_count <= 0:
        return None, None
    ratio = target_count / source_count
    tau = next((index for index, value in enumerate(positions, 1) if value >= source_count), target_count)
    values = [positions[index - 1] - (index - 1) / ratio for index in range(1, tau + 1)]
    value = mean(values)
    return round(value, 6), round(value / source_count, 6)


def v1_scheduled_positions(row: dict[str, Any]) -> list[int]:
    writes = [
        int(item["source_prefix_length"])
        for item in row["causal_timeline"]
        if str(item["action_type"]).startswith("WRITE")
    ]
    if len(writes) != len(row["target_pseudo_asl_gloss_tokens"]):
        raise ValueError(f"v1 write count mismatch for {row['id']}")
    return writes


def stable_positions(
    token_logprobs: list[float],
    full_logprob: float,
    margin: float,
    minimum_probability: float,
    window: int,
) -> list[int]:
    source_count = len(token_logprobs) - 1
    threshold = max(full_logprob - margin, math.log(minimum_probability))
    valid = []
    for position in range(1, source_count + 1):
        if position == source_count:
            valid.append(position)
            continue
        end = position + window - 1
        if end > source_count:
            continue
        if all(token_logprobs[index] >= threshold for index in range(position, end + 1)):
            valid.append(position)
    return sorted(set(valid))


def monotonic_dynamic_programming(costs: list[list[float]]) -> list[int]:
    target_count = len(costs)
    source_count = len(costs[0]) - 1
    dp = [[float("inf")] * (source_count + 1) for _ in range(target_count)]
    back = [[0] * (source_count + 1) for _ in range(target_count)]
    for position in range(1, source_count + 1):
        dp[0][position] = costs[0][position]
    for target_index in range(1, target_count):
        best_value = float("inf")
        best_position = 1
        for position in range(1, source_count + 1):
            candidate = dp[target_index - 1][position]
            if candidate < best_value:
                best_value = candidate
                best_position = position
            dp[target_index][position] = costs[target_index][position] + best_value
            back[target_index][position] = best_position
    final_position = min(range(1, source_count + 1), key=lambda value: dp[-1][value])
    positions = [final_position]
    for target_index in range(target_count - 1, 0, -1):
        final_position = back[target_index][final_position]
        positions.append(final_position)
    return list(reversed(positions))


def schedule_positions(required: list[int], source_count: int) -> list[int]:
    scheduled = []
    previous = 0
    for position in required:
        if previous < source_count:
            actual = min(source_count, max(position, previous + 1))
        else:
            actual = source_count
        scheduled.append(actual)
        previous = actual
    return scheduled


def build_policy_alignment(
    source_row: dict[str, Any], score_row: dict[str, Any], policy_name: str, policy: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    source_count = len(source_row["source_tokens"])
    gloss_tokens = source_row["target_pseudo_asl_gloss_tokens"]
    matrix = score_row["gloss_token_mean_logprob_by_source_position"]
    v1_tokens = source_row["gloss_token_alignment"]
    margin = float(policy["relative_logprob_margin_nats"])
    min_prob = float(policy["minimum_geometric_mean_probability"])
    window = int(policy["stability_window_prefixes"])
    minimum_gain = float(config["minimum_source_conditioning_gain_nats"])
    infeasible = 1_000_000.0
    costs: list[list[float]] = []
    token_meta = []

    for gloss_index, gloss_token in enumerate(gloss_tokens):
        token_logprobs = [float(row[gloss_index]) for row in matrix]
        empty_lp = token_logprobs[0]
        full_lp = token_logprobs[-1]
        source_gain = full_lp - empty_lp
        v1 = v1_tokens[gloss_index]
        v1_position = int(v1["evidence_source_position"])
        v1_method = str(v1["alignment_method"])
        candidates = stable_positions(token_logprobs, full_lp, margin, min_prob, window)
        weak_conditioning = source_gain < minimum_gain
        fallback_reason = None
        if weak_conditioning:
            if v1_method in LEXICAL_METHODS:
                candidates = [v1_position, source_count]
                fallback_reason = "lexical_anchor_due_to_weak_source_conditioning"
            else:
                candidates = [source_count]
                fallback_reason = "full_source_due_to_weak_source_conditioning"
        candidates = sorted(set(max(1, min(source_count, value)) for value in candidates))
        row_costs = [infeasible] * (source_count + 1)
        for position in candidates:
            semantic_gap = max(0.0, full_lp - token_logprobs[position])
            cost = float(policy["latency_weight"]) * position / source_count
            cost += float(policy["semantic_gap_weight"]) * semantic_gap
            if v1_method in LEXICAL_METHODS:
                if position < v1_position:
                    cost += float(policy["lexical_anchor_early_weight"]) * (v1_position - position) / source_count
                else:
                    cost += float(policy["lexical_anchor_late_weight"]) * (position - v1_position) / source_count
            row_costs[position] = cost
        costs.append(row_costs)
        token_meta.append(
            {
                "gloss_index": gloss_index,
                "gloss_token": gloss_token,
                "empty_source_mean_logprob": round(empty_lp, 6),
                "full_source_mean_logprob": round(full_lp, 6),
                "source_conditioning_gain_nats": round(source_gain, 6),
                "weak_source_conditioning": weak_conditioning,
                "stable_candidate_source_positions": candidates,
                "v1_evidence_source_position": v1_position,
                "v1_alignment_method": v1_method,
                "fallback_reason": fallback_reason,
            }
        )

    required = monotonic_dynamic_programming(costs)
    scheduled = schedule_positions(required, source_count)
    for index, meta in enumerate(token_meta):
        required_position = required[index]
        scheduled_position = scheduled[index]
        full_lp = float(meta["full_source_mean_logprob"])
        required_lp = float(matrix[required_position][index])
        scheduled_lp = float(matrix[scheduled_position][index])
        meta.update(
            {
                "semantic_required_source_position": required_position,
                "scheduled_emit_source_position": scheduled_position,
                "required_position_mean_logprob": round(required_lp, 6),
                "scheduled_position_mean_logprob": round(scheduled_lp, 6),
                "scheduled_geometric_mean_probability": round(math.exp(scheduled_lp), 8),
                "scheduled_to_full_logprob_gap_nats": round(max(0.0, full_lp - scheduled_lp), 6),
                "scheduled_minus_v1_source_position": scheduled_position - int(meta["v1_evidence_source_position"]),
            }
        )
    al, normalized_al = average_lagging(scheduled, source_count, len(gloss_tokens))
    return {
        "policy_name": policy_name,
        "semantic_required_source_positions": required,
        "scheduled_emit_source_positions": scheduled,
        "gloss_token_alignment": token_meta,
        "metrics": {
            "word_position_average_lagging_source_tokens": al,
            "word_position_normalized_average_lagging": normalized_al,
            "mean_scheduled_emit_source_proportion": safe_mean([value / source_count for value in scheduled]),
            "mean_scheduled_to_full_logprob_gap_nats": safe_mean([item["scheduled_to_full_logprob_gap_nats"] for item in token_meta]),
            "weak_source_conditioning_token_count": sum(item["weak_source_conditioning"] for item in token_meta),
            "full_source_emit_token_count": sum(value == source_count for value in scheduled),
        },
    }


def build_timeline(
    source_row: dict[str, Any], policy_alignment: dict[str, Any], wait_token: str, end_token: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_tokens = source_row["source_tokens"]
    gloss_tokens = source_row["target_pseudo_asl_gloss_tokens"]
    scheduled = policy_alignment["scheduled_emit_source_positions"]
    required = policy_alignment["semantic_required_source_positions"]
    source_count = len(source_tokens)
    actions: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    target_index = 0

    def add(source_position: int, action: str, action_type: str) -> None:
        action_index = len(actions)
        committed = [item["action"] for item in actions if str(item["action_type"]).startswith("WRITE")]
        gloss_index = target_index if action_type.startswith("WRITE") else None
        item = {
            "action_index": action_index,
            "source_prefix_length": source_position,
            "action": action,
            "action_type": action_type,
            "target_gloss_index": gloss_index,
            "semantic_required_source_position": required[target_index] if gloss_index is not None else None,
            "scheduled_emit_source_position": scheduled[target_index] if gloss_index is not None else None,
        }
        actions.append(item)
        examples.append(
            {
                "id": f"{source_row['id']}__semantic_v2_action_{action_index:03d}",
                "parent_id": source_row["id"],
                "split": "pilot_train",
                "state_prompt": (
                    "Observed English prefix:\n"
                    + DETOKENIZER.detokenize(source_tokens[:source_position])
                    + "\n\nAlready committed pseudo ASL gloss:\n"
                    + (" ".join(committed) if committed else "<EMPTY>")
                    + "\n\nSource status: "
                    + ("COMPLETE" if source_position >= source_count else "INCOMPLETE")
                    + "\nReturn exactly one next action."
                ),
                "target_action": action,
                "target_action_type": action_type,
                "source_prefix_length": source_position,
                "source_token_count": source_count,
                "source_complete": source_position >= source_count,
                "partial_source_text_en": DETOKENIZER.detokenize(source_tokens[:source_position]),
                "committed_pseudo_asl_gloss": " ".join(committed),
                "parent_target_pseudo_asl_gloss": source_row["target_pseudo_asl_gloss"],
                "target_is_gold": False,
                "alignment_is_gold": False,
                "alignment_version": "semantic_causal_alignment_v2_pilot500_v1",
            }
        )

    for source_position in range(1, source_count + 1):
        if target_index < len(gloss_tokens) and scheduled[target_index] <= source_position:
            add(source_position, gloss_tokens[target_index], "WRITE_GLOSS")
            target_index += 1
        elif source_position == source_count and target_index == len(gloss_tokens):
            add(source_position, end_token, "END")
        else:
            add(source_position, wait_token, "READ_WAIT")
    while target_index < len(gloss_tokens):
        add(source_count, gloss_tokens[target_index], "WRITE_AFTER_SOURCE_END")
        target_index += 1
    if not actions or actions[-1]["action_type"] != "END":
        add(source_count, end_token, "END")
    return actions, examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parents", type=Path, required=True)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-pairs", type=Path, required=True)
    parser.add_argument("--output-actions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    parents = read_jsonl(args.parents)
    scores = [row for path in args.scores for row in read_jsonl(path)]
    score_by_id = {row["id"]: row for row in scores}
    primary = config["primary_policy"]
    pair_rows = []
    action_rows = []
    comparison_rows = []

    for parent in parents:
        score = score_by_id[parent["id"]]
        variants = {
            name: build_policy_alignment(parent, score, name, policy, config)
            for name, policy in config["policies"].items()
        }
        chosen = variants[primary]
        timeline, examples = build_timeline(parent, chosen, config["wait_token"], config["end_token"])
        action_rows.extend(examples)
        v1_positions = v1_scheduled_positions(parent)
        v1_al, v1_normalized_al = average_lagging(
            v1_positions, len(parent["source_tokens"]), len(parent["target_pseudo_asl_gloss_tokens"])
        )
        matrix = score["gloss_token_mean_logprob_by_source_position"]
        v1_gaps = []
        v1_sufficient = 0
        balanced_margin = float(config["policies"][primary]["relative_logprob_margin_nats"])
        balanced_min_prob = float(config["policies"][primary]["minimum_geometric_mean_probability"])
        for index, position in enumerate(v1_positions):
            gap = max(0.0, float(matrix[-1][index]) - float(matrix[position][index]))
            v1_gaps.append(gap)
            if gap <= balanced_margin and math.exp(float(matrix[position][index])) >= balanced_min_prob:
                v1_sufficient += 1
        chosen_positions = chosen["scheduled_emit_source_positions"]
        v2_sufficient = sum(
            item["scheduled_to_full_logprob_gap_nats"] <= balanced_margin
            and item["scheduled_geometric_mean_probability"] >= balanced_min_prob
            for item in chosen["gloss_token_alignment"]
        )
        deltas = [new - old for new, old in zip(chosen_positions, v1_positions)]
        flags = []
        weak_count = chosen["metrics"]["weak_source_conditioning_token_count"]
        if weak_count:
            flags.append("contains_weak_source_conditioning_tokens")
        if chosen["metrics"]["full_source_emit_token_count"] / max(1, len(chosen_positions)) > 0.5:
            flags.append("mostly_emitted_at_full_source")
        if any(delta <= -4 for delta in deltas):
            flags.append("large_earlier_shift_vs_v1")
        if any(delta >= 4 for delta in deltas):
            flags.append("large_later_shift_vs_v1")
        if any(prob < float(config["low_full_source_probability_threshold"]) for prob in score["full_source_geometric_mean_probabilities"]):
            flags.append("low_full_source_target_probability")
        pair_rows.append(
            {
                "id": parent["id"],
                "semantic_v2_pilot_index": parent["semantic_v2_pilot_index"],
                "source_text_en": parent["source_text_en"],
                "source_tokens": parent["source_tokens"],
                "target_pseudo_asl_gloss": parent["target_pseudo_asl_gloss"],
                "target_pseudo_asl_gloss_tokens": parent["target_pseudo_asl_gloss_tokens"],
                "target_pseudo_gloss_is_gold": False,
                "source_dataset": parent.get("source_dataset"),
                "how2sign_video_ids": parent.get("how2sign_video_ids", []),
                "alignment_method": "finetune1_prefix_semantic_probability_plus_lexical_anchor_monotonic_dp_v2",
                "alignment_model": config["base_model_id"],
                "alignment_model_revision": config["base_model_revision"],
                "alignment_adapter": config["finetune1_adapter"],
                "primary_policy": primary,
                "semantic_gloss_token_alignment": chosen["gloss_token_alignment"],
                "alignment_policy_variants": {
                    name: {
                        "semantic_required_source_positions": value["semantic_required_source_positions"],
                        "scheduled_emit_source_positions": value["scheduled_emit_source_positions"],
                        "metrics": value["metrics"],
                    }
                    for name, value in variants.items()
                },
                "causal_action_sequence": [item["action"] for item in timeline],
                "causal_action_source_prefix_lengths": [item["source_prefix_length"] for item in timeline],
                "causal_timeline": timeline,
                "v1_scheduled_emit_source_positions": v1_positions,
                "v1_metrics": {
                    "word_position_average_lagging_source_tokens": v1_al,
                    "word_position_normalized_average_lagging": v1_normalized_al,
                    "mean_emit_to_full_logprob_gap_nats": safe_mean(v1_gaps),
                },
                "metrics": chosen["metrics"],
                "v2_minus_v1_scheduled_position_deltas": deltas,
                "semantic_alignment_quality_flags": flags,
                "manual_review_status": "pending",
                "alignment_is_gold": False,
            }
        )
        comparison_rows.append(
            {
                "id": parent["id"],
                "source_text_en": parent["source_text_en"],
                "target_pseudo_asl_gloss": parent["target_pseudo_asl_gloss"],
                "v1_positions": v1_positions,
                "v2_positions": chosen_positions,
                "deltas": deltas,
                "v1_normalized_al": v1_normalized_al,
                "v2_normalized_al": chosen["metrics"]["word_position_normalized_average_lagging"],
                "v1_mean_gap": safe_mean(v1_gaps),
                "v2_mean_gap": chosen["metrics"]["mean_scheduled_to_full_logprob_gap_nats"],
                "token_count": len(chosen_positions),
                "v1_semantically_sufficient_token_count": v1_sufficient,
                "v2_semantically_sufficient_token_count": v2_sufficient,
                "flags": flags,
            }
        )

    write_jsonl(args.output_pairs, pair_rows)
    write_jsonl(args.output_actions, action_rows)
    all_deltas = [delta for row in comparison_rows for delta in row["deltas"]]
    al_differences = [
        row["v2_normalized_al"] - row["v1_normalized_al"]
        for row in comparison_rows
        if row["v1_normalized_al"] is not None and row["v2_normalized_al"] is not None
    ]
    gap_differences = [
        row["v2_mean_gap"] - row["v1_mean_gap"]
        for row in comparison_rows
        if row["v1_mean_gap"] is not None and row["v2_mean_gap"] is not None
    ]
    total_tokens = sum(row["token_count"] for row in comparison_rows)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": config["scientific_status"],
        "parent_count": len(pair_rows),
        "action_count": len(action_rows),
        "primary_policy": primary,
        "gloss_token_count": len(all_deltas),
        "position_comparison": {
            "unchanged_token_count": sum(delta == 0 for delta in all_deltas),
            "earlier_token_count": sum(delta < 0 for delta in all_deltas),
            "later_token_count": sum(delta > 0 for delta in all_deltas),
            "changed_token_ratio": round(sum(delta != 0 for delta in all_deltas) / max(1, len(all_deltas)), 6),
            "mean_v2_minus_v1_source_position": safe_mean(all_deltas),
            "median_v2_minus_v1_source_position": median(all_deltas) if all_deltas else None,
            "mean_absolute_position_change": safe_mean([abs(delta) for delta in all_deltas]),
        },
        "v1": {
            "mean_normalized_average_lagging": safe_mean([row["v1_normalized_al"] for row in comparison_rows if row["v1_normalized_al"] is not None]),
            "mean_emit_to_full_logprob_gap_nats": safe_mean([row["v1_mean_gap"] for row in comparison_rows if row["v1_mean_gap"] is not None]),
        },
        "v2_balanced": {
            "mean_normalized_average_lagging": safe_mean([row["v2_normalized_al"] for row in comparison_rows if row["v2_normalized_al"] is not None]),
            "mean_emit_to_full_logprob_gap_nats": safe_mean([row["v2_mean_gap"] for row in comparison_rows if row["v2_mean_gap"] is not None]),
            "mean_weak_source_conditioning_tokens_per_parent": safe_mean([row["metrics"]["weak_source_conditioning_token_count"] for row in pair_rows]),
            "mean_full_source_emit_tokens_per_parent": safe_mean([row["metrics"]["full_source_emit_token_count"] for row in pair_rows]),
        },
        "balanced_semantic_sufficiency_at_emission": {
            "definition": "emission logprob is within the balanced margin of full-source logprob and exceeds the balanced absolute probability floor",
            "v1_token_ratio": round(sum(row["v1_semantically_sufficient_token_count"] for row in comparison_rows) / max(1, total_tokens), 6),
            "v2_token_ratio": round(sum(row["v2_semantically_sufficient_token_count"] for row in comparison_rows) / max(1, total_tokens), 6),
        },
        "paired_parent_bootstrap_5000": {
            "normalized_average_lagging_v2_minus_v1": paired_bootstrap_difference(
                al_differences, int(config["selection_seed"]) + 101
            ),
            "semantic_logprob_gap_v2_minus_v1": paired_bootstrap_difference(
                gap_differences, int(config["selection_seed"]) + 202
            ),
            "interpretation": "Intervals quantify internal pseudo-alignment diagnostics, not gold ASL accuracy.",
        },
        "policy_sensitivity": {
            name: {
                "mean_normalized_average_lagging": safe_mean([row["alignment_policy_variants"][name]["metrics"]["word_position_normalized_average_lagging"] for row in pair_rows]),
                "mean_semantic_gap_nats": safe_mean([row["alignment_policy_variants"][name]["metrics"]["mean_scheduled_to_full_logprob_gap_nats"] for row in pair_rows]),
                "mean_full_source_emit_tokens": safe_mean([row["alignment_policy_variants"][name]["metrics"]["full_source_emit_token_count"] for row in pair_rows]),
            }
            for name in config["policies"]
        },
        "quality_flag_counts": dict(Counter(flag for row in pair_rows for flag in row["semantic_alignment_quality_flags"])),
        "important_limitations": config["important_limitations"],
        "ok": len(pair_rows) == int(config["pilot_parent_count"]),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    priority = sorted(
        comparison_rows,
        key=lambda row: (
            -len(row["flags"]),
            -max([abs(value) for value in row["deltas"]] or [0]),
            row["id"],
        ),
    )[:50]
    columns = [
        "id",
        "source_text_en",
        "target_pseudo_asl_gloss",
        "v1_positions",
        "v2_positions",
        "deltas",
        "v1_normalized_al",
        "v2_normalized_al",
        "v1_mean_gap",
        "v2_mean_gap",
        "token_count",
        "v1_semantically_sufficient_token_count",
        "v2_semantically_sufficient_token_count",
        "flags",
    ]
    for path in (args.preview, args.review):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns + (["review_decision", "reviewer_notes"] if path == args.review else []), delimiter="\t")
            writer.writeheader()
            for row in priority:
                output = dict(row)
                for key in ("v1_positions", "v2_positions", "deltas", "flags"):
                    output[key] = json.dumps(output[key], ensure_ascii=False)
                if path == args.review:
                    output["review_decision"] = "PENDING"
                    output["reviewer_notes"] = ""
                writer.writerow(output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
