#!/usr/bin/env python3
"""Build full v2.2 causal trajectories with confident-prefix burst commits."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from nltk.tokenize import TreebankWordDetokenizer


NEGATION = {"NOT", "NO", "NONE", "NEVER", "NOTHING", "NOBODY", "WITHOUT", "CANNOT", "CAN'T", "DON'T"}
DETOKENIZER = TreebankWordDetokenizer()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def detokenize(tokens: list[str]) -> str:
    return DETOKENIZER.detokenize(tokens)


def stable_positions(values: list[float], full: float, margin: float, minimum_probability: float, window: int) -> list[int]:
    source_count = len(values) - 1
    threshold = max(full - margin, math.log(minimum_probability))
    output = []
    for position in range(1, source_count + 1):
        end = position + window - 1
        if position == source_count or (
            end <= source_count and all(values[index] >= threshold for index in range(position, end + 1))
        ):
            output.append(position)
    return output


def average_lagging(positions: list[int], source_count: int, target_count: int) -> tuple[float, float]:
    gamma = target_count / source_count
    tau = next((index for index, value in enumerate(positions, 1) if value >= source_count), target_count)
    value = mean(positions[index - 1] - (index - 1) / gamma for index in range(1, tau + 1))
    return value, value / source_count


def choose_positions(pair: dict[str, Any], score: dict[str, Any], config: dict[str, Any]) -> tuple[list[int], list[dict[str, Any]]]:
    matrix = score["gloss_token_mean_logprob_by_source_position"]
    source_count = len(pair["source_tokens"])
    positions = []
    output_meta = []
    previous = 1
    for index, (token, old_meta) in enumerate(zip(pair["target_pseudo_asl_gloss_tokens"], pair["semantic_gloss_token_alignment"])):
        values = [float(row[index]) for row in matrix]
        full = values[-1]
        anchor = max(1, min(source_count, int(old_meta["source_position"])))
        stable = [
            value
            for value in stable_positions(
                values,
                full,
                float(config["semantic_margin"]),
                float(config["minimum_probability"]),
                int(config["stability_window"]),
            )
            if value >= anchor
        ]
        earliest_stable = stable[0] if stable else source_count
        anchor_gap = max(0.0, full - values[anchor])
        anchor_probability = math.exp(values[anchor])
        method = str(old_meta.get("match_method", ""))
        strong_lexical = bool(old_meta.get("anchor_is_lexical")) and (
            int(old_meta.get("match_strength", 0)) >= 2 or "exact" in method or "stem" in method
        )
        if token.upper() in NEGATION and strong_lexical:
            proposed = anchor
            reason = "negation_lexical_anchor"
        elif strong_lexical and anchor_gap <= float(config["lexical_gap_margin"]) and anchor_probability >= float(config["minimum_probability"]):
            proposed = anchor
            reason = "lexical_anchor_semantically_supported"
        else:
            rescue_cap = min(source_count, anchor + int(config["maximum_rescue_delay"]))
            rescue_gap = max(0.0, full - values[rescue_cap])
            if earliest_stable <= rescue_cap:
                proposed = earliest_stable
                reason = "stable_prefix"
            elif rescue_gap <= float(config["rescue_gap_margin"]):
                proposed = rescue_cap
                reason = "bounded_semantic_rescue"
            else:
                proposed = earliest_stable
                reason = "late_stable_prefix"
        position = max(previous, min(source_count, proposed))
        previous = position
        gap = max(0.0, full - values[position])
        probability = math.exp(values[position])
        flags = []
        if position > proposed:
            flags.append("delayed_by_target_order_monotonicity")
        if token.upper() in NEGATION and position > anchor + int(config["negation_maximum_delay_after_anchor"]):
            flags.append("negation_delayed_by_target_order")
        if gap > float(config["semantic_sufficiency_margin"]) or probability < float(config["semantic_sufficiency_minimum_probability"]):
            flags.append("weak_prefix_semantic_support")
        output_meta.append(
            {
                **old_meta,
                "v2_2_anchor_source_position": anchor,
                "v2_2_earliest_stable_source_position": earliest_stable,
                "v2_2_proposed_source_position": proposed,
                "v2_2_emit_source_position": position,
                "v2_2_decision_reason": reason,
                "v2_2_position_mean_logprob": round(values[position], 6),
                "v2_2_position_probability": round(probability, 8),
                "v2_2_to_full_logprob_gap_nats": round(gap, 6),
                "v2_2_quality_flags": flags,
            }
        )
        positions.append(position)
    return positions, output_meta


def build_timeline(pair: dict[str, Any], positions: list[int], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_tokens = pair["source_tokens"]
    gloss_tokens = pair["target_pseudo_asl_gloss_tokens"]
    source_count = len(source_tokens)
    timeline: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    committed: list[str] = []
    target_index = 0

    def add(source_position: int, action: str, action_type: str, gloss_index: int | None) -> None:
        action_index = len(timeline)
        timeline.append(
            {
                "action_index": action_index,
                "source_prefix_length": source_position,
                "action": action,
                "action_type": action_type,
                "target_gloss_index": gloss_index,
                "scheduled_emit_source_position": positions[gloss_index] if gloss_index is not None else None,
            }
        )
        examples.append(
            {
                "id": f"{pair['id']}__semantic_v2_2_action_{action_index:03d}",
                "parent_id": pair["id"],
                "split": pair.get("split", "train"),
                "state_prompt": (
                    "Observed English prefix:\n" + detokenize(source_tokens[:source_position])
                    + "\n\nAlready committed pseudo ASL gloss:\n" + (" ".join(committed) if committed else "<EMPTY>")
                    + "\n\nSource status: " + ("COMPLETE" if source_position >= source_count else "INCOMPLETE")
                    + "\nReturn exactly one next action."
                ),
                "target_action": action,
                "target_action_type": action_type,
                "source_prefix_length": source_position,
                "source_token_count": source_count,
                "source_complete": source_position >= source_count,
                "partial_source_text_en": detokenize(source_tokens[:source_position]),
                "committed_pseudo_asl_gloss": " ".join(committed),
                "parent_target_pseudo_asl_gloss": pair["target_pseudo_asl_gloss"],
                "target_is_gold": False,
                "alignment_is_gold": False,
                "alignment_version": config["dataset_version"],
                "transition_semantics": "WRITE_keeps_prefix_WAIT_advances_prefix",
            }
        )

    for source_position in range(1, source_count + 1):
        while target_index < len(gloss_tokens) and positions[target_index] <= source_position:
            add(source_position, gloss_tokens[target_index], "WRITE_GLOSS", target_index)
            committed.append(gloss_tokens[target_index])
            target_index += 1
        if source_position < source_count:
            add(source_position, config["wait_token"], "READ_WAIT", None)
        elif target_index == len(gloss_tokens):
            add(source_position, config["end_token"], "END", None)
    while target_index < len(gloss_tokens):
        add(source_count, gloss_tokens[target_index], "WRITE_AFTER_SOURCE_END", target_index)
        committed.append(gloss_tokens[target_index])
        target_index += 1
    if not timeline or timeline[-1]["action_type"] != "END":
        add(source_count, config["end_token"], "END", None)
    return timeline, examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-pairs", type=Path, required=True)
    parser.add_argument("--output-actions", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    pairs = read_jsonl(args.pairs)
    scores = {row["id"]: row for path in args.scores for row in read_jsonl(path)}
    missing_scores = sorted({row["id"] for row in pairs} - scores.keys())
    if missing_scores:
        raise ValueError(f"Missing semantic scores for {len(missing_scores)} parents: {missing_scores[:5]}")

    output_pairs = []
    output_actions = []
    split_counts = Counter()
    action_split_counts = Counter()
    decision_counts = Counter()
    flag_counts = Counter()
    normalized_als = []
    als = []
    gaps = []
    sufficient = 0
    token_count = 0
    burst_token_count = 0
    full_source_token_count = 0
    for pair in pairs:
        positions, meta = choose_positions(pair, scores[pair["id"]], config)
        timeline, examples = build_timeline(pair, positions, config)
        source_count = len(pair["source_tokens"])
        target_count = len(pair["target_pseudo_asl_gloss_tokens"])
        value, normalized = average_lagging(positions, source_count, target_count)
        parent_flags = sorted(set(pair.get("v2_1_exclusion_reasons", [])) | {flag for item in meta for flag in item["v2_2_quality_flags"]})
        for item in meta:
            decision_counts[item["v2_2_decision_reason"]] += 1
            flag_counts.update(item["v2_2_quality_flags"])
            gap = float(item["v2_2_to_full_logprob_gap_nats"])
            probability = float(item["v2_2_position_probability"])
            gaps.append(gap)
            token_count += 1
            sufficient += gap <= float(config["semantic_sufficiency_margin"]) and probability >= float(config["semantic_sufficiency_minimum_probability"])
        burst_token_count += sum(positions[index] == positions[index - 1] for index in range(1, len(positions)))
        full_source_token_count += sum(position >= source_count for position in positions)
        split = pair.get("split", "train")
        split_counts[split] += 1
        action_split_counts[split] += len(examples)
        als.append(value)
        normalized_als.append(normalized)
        output_pairs.append(
            {
                **pair,
                "alignment_method": "confidence_guarded_lexical_anchor_finetune1_semantics_burst_v2_2",
                "alignment_version": config["dataset_version"],
                "semantic_v2_2_emit_source_positions": positions,
                "semantic_v2_2_gloss_token_alignment": meta,
                "causal_action_sequence": [row["action"] for row in timeline],
                "causal_action_source_prefix_lengths": [row["source_prefix_length"] for row in timeline],
                "causal_timeline": timeline,
                "v2_2_metrics": {
                    "word_position_average_lagging_source_tokens": round(value, 6),
                    "word_position_normalized_average_lagging": round(normalized, 6),
                    "mean_semantic_gap_nats": round(mean(float(item["v2_2_to_full_logprob_gap_nats"]) for item in meta), 6),
                    "semantic_sufficiency_ratio": round(sum(
                        float(item["v2_2_to_full_logprob_gap_nats"]) <= float(config["semantic_sufficiency_margin"])
                        and float(item["v2_2_position_probability"]) >= float(config["semantic_sufficiency_minimum_probability"])
                        for item in meta
                    ) / max(1, len(meta)), 6),
                    "burst_write_token_count": sum(positions[index] == positions[index - 1] for index in range(1, len(positions))),
                    "full_source_emit_token_count": sum(position >= source_count for position in positions),
                },
                "v2_2_quality_flags": parent_flags,
                "eligible_for_finetune2_v2_2": True,
                "alignment_is_gold": False,
            }
        )
        output_actions.extend(examples)

    write_jsonl(args.output_pairs, output_pairs)
    write_jsonl(args.output_actions, output_actions)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "confidence_guarded_burst_pseudo_timing_not_gold_asl",
        "config": config,
        "parent_count": len(output_pairs),
        "action_count": len(output_actions),
        "gloss_token_count": token_count,
        "parent_split_counts": dict(split_counts),
        "action_split_counts": dict(action_split_counts),
        "mean_normalized_average_lagging": round(mean(normalized_als), 6),
        "mean_average_lagging_source_tokens": round(mean(als), 6),
        "mean_semantic_gap_nats": round(mean(gaps), 6),
        "semantic_sufficiency_ratio": round(sufficient / max(1, token_count), 6),
        "mean_burst_tokens_per_parent": round(burst_token_count / max(1, len(output_pairs)), 6),
        "mean_full_source_tokens_per_parent": round(full_source_token_count / max(1, len(output_pairs)), 6),
        "decision_reason_counts": dict(decision_counts),
        "quality_flag_counts": dict(flag_counts),
        "retained_previous_v2_1_exclusion_reason_counts": dict(Counter(
            reason for row in output_pairs for reason in row.get("v2_1_exclusion_reasons", [])
        )),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
