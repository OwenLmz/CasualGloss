#!/usr/bin/env python3
"""Build Semantic Causal Alignment v2.1 from frozen v2 prefix score matrices."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

NEGATION_GLOSSES = {"NEG", "NO", "NOT", "NEVER", "CANNOT", "CAN'T", "DON'T", "DOESN'T", "WON'T"}
FUNCTION_GLOSSES = {
    "A", "AN", "THE", "TO", "OF", "IN", "ON", "AT", "AS", "BE", "IS", "ARE", "WAS", "WERE",
    "DO", "DOES", "DID", "HAVE", "HAS", "HAD", "AND", "BUT", "OR", "THAT", "THIS", "THESE",
    "THOSE", "YOU", "I", "ME", "WE", "THEY", "HE", "SHE", "IT", "YOUR", "MY", "OUR", "THEIR",
}
NEGATION_SOURCE = {"no", "not", "never", "cannot", "cant", "dont", "doesnt", "didnt", "wont", "wouldnt"}
ALIASES = {
    "neg": NEGATION_SOURCE,
    "cannot": {"cannot", "cant"},
    "people": {"people", "person", "somebody", "someone"},
    "person": {"person", "people", "somebody", "someone"},
    "child": {"child", "children", "kid", "kids"},
    "buy": {"buy", "purchase"},
    "car": {"car", "vehicle"},
    "eye": {"eye", "eyes", "eying", "eyed"},
}
MOJIBAKE_MARKERS = ("\ufffd", "\u00c3", "\u00c2", "\u00e2", "\u20ac", "\u2122", "\u8292\u804f")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:%)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    text = re.sub(r"\s+'\s+", "'", text)
    return text


def simple_stem(value: str) -> str:
    value = normalize(value)
    for suffix in ("ingly", "edly", "ation", "ments", "ment", "ness", "ingly", "ing", "ied", "ies", "ed", "es", "s"):
        if value.endswith(suffix) and len(value) - len(suffix) >= 3:
            if suffix in {"ied", "ies"}:
                return value[: -len(suffix)] + "y"
            return value[: -len(suffix)]
    return value


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
        actual = min(source_count, max(position, previous + 1)) if previous < source_count else source_count
        scheduled.append(actual)
        previous = actual
    return scheduled


def average_lagging(positions: list[int], source_count: int, target_count: int) -> tuple[float | None, float | None]:
    if not positions or source_count <= 0 or target_count <= 0:
        return None, None
    ratio = target_count / source_count
    tau = next((index for index, value in enumerate(positions, 1) if value >= source_count), target_count)
    values = [positions[index - 1] - (index - 1) / ratio for index in range(1, tau + 1)]
    result = mean(values)
    return round(result, 6), round(result / source_count, 6)


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower().replace("’", "'")
    value = re.sub(r"[^a-z0-9']+", "", value)
    return value.replace("'", "")


def source_is_corrupt(text: str) -> bool:
    if any(marker in text for marker in MOJIBAKE_MARKERS):
        return True
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return True
    cjk = sum("\u4e00" <= char <= "\u9fff" for char in letters)
    return cjk / len(letters) > 0.05


def match_strength(gloss_piece: str, source_token: str) -> int:
    gloss = normalize(gloss_piece)
    source = normalize(source_token)
    if not gloss or not source:
        return 0
    if gloss == source:
        return 3
    if source in ALIASES.get(gloss, set()) or gloss in ALIASES.get(source, set()):
        return 3
    if gloss in {"not", "no", "never", "neg"} and source in NEGATION_SOURCE:
        return 3
    if len(gloss) >= 3 and len(source) >= 3 and simple_stem(gloss) == simple_stem(source):
        return 2
    return 0


def gloss_pieces(gloss_token: str) -> list[str]:
    return [piece for piece in re.split(r"[-_/]+", gloss_token) if normalize(piece)] or [gloss_token]


def token_candidates(gloss_token: str, source_tokens: list[str]) -> list[tuple[int, int]]:
    pieces = gloss_pieces(gloss_token)
    if len(pieces) == 1:
        return [
            (index + 1, match_strength(pieces[0], token))
            for index, token in enumerate(source_tokens)
            if match_strength(pieces[0], token) > 0
        ]
    candidates: list[tuple[int, int]] = []
    for start in range(len(source_tokens)):
        cursor = start
        strengths = []
        for piece in pieces:
            found = None
            for source_index in range(cursor, min(len(source_tokens), cursor + 4)):
                strength = match_strength(piece, source_tokens[source_index])
                if strength:
                    found = (source_index, strength)
                    break
            if found is None:
                strengths = []
                break
            cursor = found[0] + 1
            strengths.append(found[1])
        if strengths:
            candidates.append((cursor, min(strengths)))
    return candidates


def occurrence_aware_anchors(gloss_tokens: list[str], source_tokens: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_candidates = [token_candidates(token, source_tokens) for token in gloss_tokens]
    independent = [candidates[0][0] if candidates else None for candidates in all_candidates]
    anchors: list[dict[str, Any]] = []
    previous = 0
    matched = 0
    backward_jumps = 0
    backward_distance = 0
    last_independent = 0
    for index, (token, candidates) in enumerate(zip(gloss_tokens, all_candidates)):
        if independent[index] is not None:
            if independent[index] < last_independent:
                backward_jumps += 1
                backward_distance += last_independent - independent[index]
            last_independent = independent[index]
        valid = [item for item in candidates if item[0] >= previous]
        chosen = valid[0] if valid else None
        if chosen:
            previous = chosen[0]
            matched += 1
        anchors.append(
            {
                "gloss_index": index,
                "gloss_token": token,
                "source_position": chosen[0] if chosen else None,
                "match_strength": chosen[1] if chosen else 0,
                "independent_first_source_position": independent[index],
                "candidate_source_positions": [item[0] for item in candidates],
                "match_method": "occurrence_aware_exact" if chosen and chosen[1] == 3 else (
                    "occurrence_aware_stem" if chosen else "unmatched"
                ),
            }
        )
    lexical_count = sum(bool(candidates) for candidates in all_candidates)
    report = {
        "lexically_matchable_token_count": lexical_count,
        "monotonically_matched_token_count": matched,
        "monotonic_coverage": round(matched / max(1, lexical_count), 6),
        "backward_jump_count": backward_jumps,
        "backward_distance": backward_distance,
    }
    return anchors, report


def infer_missing_anchors(anchors: list[dict[str, Any]], source_count: int) -> None:
    """Give unmatched tokens a monotonic interpolation anchor without claiming lexical evidence."""
    matched_indices = [index for index, item in enumerate(anchors) if item["source_position"] is not None]
    for index, item in enumerate(anchors):
        if item["source_position"] is not None:
            continue
        previous_indices = [value for value in matched_indices if value < index]
        next_indices = [value for value in matched_indices if value > index]
        previous_index = previous_indices[-1] if previous_indices else None
        next_index = next_indices[0] if next_indices else None
        previous_position = anchors[previous_index]["source_position"] if previous_index is not None else 0
        next_position = anchors[next_index]["source_position"] if next_index is not None else source_count
        if previous_index is not None and next_index is not None:
            fraction = (index - previous_index) / (next_index - previous_index)
            inferred = round(previous_position + fraction * (next_position - previous_position))
        elif previous_index is not None:
            inferred = previous_position + min(2, index - previous_index)
        elif next_index is not None:
            inferred = next_position - min(2, next_index - index)
        else:
            inferred = source_count
        item["source_position"] = max(1, min(source_count, int(inferred)))
        item["match_method"] = "monotonic_neighbor_interpolation"
        item["anchor_is_lexical"] = False
    for item in anchors:
        item.setdefault("anchor_is_lexical", item["match_method"] != "unmatched")


def stable_positions(values: list[float], full: float, margin: float, minimum_probability: float, window: int) -> list[int]:
    source_count = len(values) - 1
    threshold = max(full - margin, math.log(minimum_probability))
    result = []
    for position in range(1, source_count + 1):
        end = position + window - 1
        if position == source_count or (end <= source_count and all(values[i] >= threshold for i in range(position, end + 1))):
            result.append(position)
    return result


def delay_for(token: str, config: dict[str, Any]) -> int:
    upper = token.upper()
    if upper in NEGATION_GLOSSES:
        return int(config["bounded_delay_negation"])
    if upper in FUNCTION_GLOSSES:
        return int(config["bounded_delay_function_word"])
    return int(config["bounded_delay_default"])


def build_alignment(parent: dict[str, Any], score: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    source_tokens = parent["source_tokens"]
    gloss_tokens = parent["target_pseudo_asl_gloss_tokens"]
    source_count = len(source_tokens)
    matrix = score["gloss_token_mean_logprob_by_source_position"]
    anchors, order = occurrence_aware_anchors(gloss_tokens, source_tokens)
    infer_missing_anchors(anchors, source_count)
    exclusion_reasons = []
    if source_is_corrupt(parent["source_text_en"]):
        exclusion_reasons.append("corrupt_or_non_english_source_text")
    if (
        order["lexically_matchable_token_count"] >= int(config["noncausal_min_lexical_token_count"])
        and order["monotonic_coverage"] < float(config["noncausal_min_monotonic_coverage"])
        and (
            order["backward_jump_count"] > int(config["noncausal_max_backward_jump_count"])
            or order["backward_distance"] > int(config["noncausal_max_backward_distance"])
        )
    ):
        exclusion_reasons.append("strong_streaming_noncausal_gloss_order")

    infeasible = 1_000_000.0
    costs: list[list[float]] = []
    meta = []
    margin = float(config["relative_logprob_margin_nats"])
    min_prob = float(config["minimum_geometric_mean_probability"])
    window = int(config["stability_window_prefixes"])
    for gloss_index, gloss_token in enumerate(gloss_tokens):
        values = [float(row[gloss_index]) for row in matrix]
        full_lp = values[-1]
        empty_lp = values[0]
        candidates = set(stable_positions(values, full_lp, margin, min_prob, window))
        anchor = anchors[gloss_index]["source_position"]
        rescue_positions = []
        if anchor is not None:
            delay = delay_for(gloss_token, config)
            rescue_positions = list(range(anchor, min(source_count, anchor + delay) + 1))
            candidates.update(rescue_positions)
            candidates = {position for position in candidates if position >= anchor}
        candidates.add(source_count)
        row_costs = [infeasible] * (source_count + 1)
        for position in sorted(candidates):
            semantic_gap = max(0.0, full_lp - values[position])
            cost = float(config["latency_weight"]) * position / source_count
            cost += float(config["semantic_gap_weight"]) * semantic_gap
            if anchor is not None:
                if position < anchor:
                    cost += float(config["lexical_anchor_early_weight"]) * (anchor - position) / source_count
                else:
                    cost += float(config["lexical_anchor_late_weight"]) * (position - anchor) / source_count
            if position in rescue_positions and semantic_gap > margin:
                cost += 0.10 * semantic_gap
            row_costs[position] = cost
        costs.append(row_costs)
        meta.append(
            {
                **anchors[gloss_index],
                "empty_source_mean_logprob": round(empty_lp, 6),
                "full_source_mean_logprob": round(full_lp, 6),
                "source_conditioning_gain_nats": round(full_lp - empty_lp, 6),
                "stable_candidate_source_positions": stable_positions(values, full_lp, margin, min_prob, window),
                "bounded_delay_rescue_source_positions": rescue_positions,
            }
        )

    required = monotonic_dynamic_programming(costs)
    scheduled = schedule_positions(required, source_count)
    late_negation_violations = 0
    for index, item in enumerate(meta):
        position = scheduled[index]
        full_lp = float(item["full_source_mean_logprob"])
        scheduled_lp = float(matrix[position][index])
        anchor = item["source_position"]
        if gloss_tokens[index].upper() in NEGATION_GLOSSES and anchor is not None and position > anchor + int(config["bounded_delay_negation"]):
            late_negation_violations += 1
        item.update(
            {
                "semantic_required_source_position": required[index],
                "scheduled_emit_source_position": position,
                "scheduled_position_mean_logprob": round(scheduled_lp, 6),
                "scheduled_geometric_mean_probability": round(math.exp(scheduled_lp), 8),
                "scheduled_to_full_logprob_gap_nats": round(max(0.0, full_lp - scheduled_lp), 6),
                "bounded_delay_rescue_used": position in item["bounded_delay_rescue_source_positions"]
                    and position not in item["stable_candidate_source_positions"],
            }
        )
    if late_negation_violations:
        exclusion_reasons.append("negation_order_incompatible_with_streaming")
    al, normalized_al = average_lagging(scheduled, source_count, len(gloss_tokens))
    sufficient = sum(
        item["scheduled_to_full_logprob_gap_nats"] <= margin
        and item["scheduled_geometric_mean_probability"] >= min_prob
        for item in meta
    )
    return {
        "eligible_for_finetune2_v2_1": not exclusion_reasons,
        "exclusion_reasons": exclusion_reasons,
        "lexical_order_diagnostics": order,
        "semantic_required_source_positions": required,
        "scheduled_emit_source_positions": scheduled,
        "gloss_token_alignment": meta,
        "metrics": {
            "word_position_average_lagging_source_tokens": al,
            "word_position_normalized_average_lagging": normalized_al,
            "mean_scheduled_to_full_logprob_gap_nats": round(mean(item["scheduled_to_full_logprob_gap_nats"] for item in meta), 6),
            "semantic_sufficient_token_count": sufficient,
            "semantic_sufficiency_ratio": round(sufficient / max(1, len(meta)), 6),
            "full_source_emit_token_count": sum(position == source_count for position in scheduled),
            "bounded_delay_rescue_token_count": sum(item["bounded_delay_rescue_used"] for item in meta),
            "late_negation_violation_count": late_negation_violations,
        },
    }


def build_timeline(parent: dict[str, Any], alignment: dict[str, Any], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_tokens = parent["source_tokens"]
    gloss_tokens = parent["target_pseudo_asl_gloss_tokens"]
    scheduled = alignment["scheduled_emit_source_positions"]
    required = alignment["semantic_required_source_positions"]
    source_count = len(source_tokens)
    timeline = []
    examples = []
    target_index = 0

    def add(source_position: int, action: str, action_type: str) -> None:
        action_index = len(timeline)
        committed = [item["action"] for item in timeline if str(item["action_type"]).startswith("WRITE")]
        gloss_index = target_index if action_type.startswith("WRITE") else None
        timeline.append(
            {
                "action_index": action_index,
                "source_prefix_length": source_position,
                "action": action,
                "action_type": action_type,
                "target_gloss_index": gloss_index,
                "semantic_required_source_position": required[target_index] if gloss_index is not None else None,
                "scheduled_emit_source_position": scheduled[target_index] if gloss_index is not None else None,
            }
        )
        examples.append(
            {
                "id": f"{parent['id']}__semantic_v2_1_action_{action_index:03d}",
                "parent_id": parent["id"],
                "split": parent.get("split", "train"),
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
                "parent_target_pseudo_asl_gloss": parent["target_pseudo_asl_gloss"],
                "target_is_gold": False,
                "alignment_is_gold": False,
                "alignment_version": config["dataset_version"],
            }
        )

    for source_position in range(1, source_count + 1):
        if target_index < len(gloss_tokens) and scheduled[target_index] <= source_position:
            add(source_position, gloss_tokens[target_index], "WRITE_GLOSS")
            target_index += 1
        elif source_position == source_count and target_index == len(gloss_tokens):
            add(source_position, config["end_token"], "END")
        else:
            add(source_position, config["wait_token"], "READ_WAIT")
    while target_index < len(gloss_tokens):
        add(source_count, gloss_tokens[target_index], "WRITE_AFTER_SOURCE_END")
        target_index += 1
    if not timeline or timeline[-1]["action_type"] != "END":
        add(source_count, config["end_token"], "END")
    return timeline, examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parents", type=Path, required=True)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-pairs", type=Path, required=True)
    parser.add_argument("--output-actions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    parents = read_jsonl(args.parents)
    scores = [row for path in args.scores for row in read_jsonl(path)]
    score_by_id = {row["id"]: row for row in scores}
    pair_rows = []
    action_rows = []
    preview_rows = []
    for parent in parents:
        alignment = build_alignment(parent, score_by_id[parent["id"]], config)
        timeline, examples = build_timeline(parent, alignment, config)
        if alignment["eligible_for_finetune2_v2_1"]:
            action_rows.extend(examples)
        v2_positions = [
            int(item["scheduled_emit_source_position"])
            for item in parent.get("semantic_gloss_token_alignment", [])
        ]
        if not v2_positions:
            v2_positions = [
                int(item["source_prefix_length"])
                for item in parent["causal_timeline"]
                if str(item["action_type"]).startswith("WRITE")
            ]
        pair = {
            "id": parent["id"],
            "source_text_en": parent["source_text_en"],
            "source_tokens": parent["source_tokens"],
            "target_pseudo_asl_gloss": parent["target_pseudo_asl_gloss"],
            "target_pseudo_asl_gloss_tokens": parent["target_pseudo_asl_gloss_tokens"],
            "target_pseudo_gloss_is_gold": False,
            "source_dataset": parent.get("source_dataset"),
            "how2sign_video_ids": parent.get("how2sign_video_ids", []),
            "split": parent.get("split", "train"),
            "alignment_method": "finetune1_prefix_semantics_occurrence_aware_anchor_bounded_delay_monotonic_dp_v2_1",
            "alignment_model": config["base_model_id"],
            "alignment_model_revision": config["base_model_revision"],
            "alignment_adapter": config["finetune1_adapter"],
            "semantic_gloss_token_alignment": alignment["gloss_token_alignment"],
            "lexical_order_diagnostics": alignment["lexical_order_diagnostics"],
            "causal_action_sequence": [item["action"] for item in timeline],
            "causal_action_source_prefix_lengths": [item["source_prefix_length"] for item in timeline],
            "causal_timeline": timeline,
            "semantic_v2_scheduled_emit_source_positions": v2_positions,
            "metrics": alignment["metrics"],
            "eligible_for_finetune2_v2_1": alignment["eligible_for_finetune2_v2_1"],
            "v2_1_exclusion_reasons": alignment["exclusion_reasons"],
            "alignment_is_gold": False,
            "asl_linguistic_review_status": "PENDING_QUALIFIED_ASL_REVIEW",
        }
        pair_rows.append(pair)
        preview_rows.append(
            {
                "id": pair["id"],
                "source_text_en": pair["source_text_en"],
                "target_pseudo_asl_gloss": pair["target_pseudo_asl_gloss"],
                "v2_positions": json.dumps(v2_positions),
                "v2_1_positions": json.dumps(alignment["scheduled_emit_source_positions"]),
                "eligible": alignment["eligible_for_finetune2_v2_1"],
                "exclusion_reasons": json.dumps(alignment["exclusion_reasons"]),
                "normalized_al": alignment["metrics"]["word_position_normalized_average_lagging"],
                "semantic_gap": alignment["metrics"]["mean_scheduled_to_full_logprob_gap_nats"],
                "full_source_emits": alignment["metrics"]["full_source_emit_token_count"],
                "rescue_tokens": alignment["metrics"]["bounded_delay_rescue_token_count"],
            }
        )

    write_jsonl(args.output_pairs, pair_rows)
    write_jsonl(args.output_actions, action_rows)
    eligible = [row for row in pair_rows if row["eligible_for_finetune2_v2_1"]]
    token_count = sum(len(row["target_pseudo_asl_gloss_tokens"]) for row in eligible)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": config["scientific_status"],
        "parent_count": len(pair_rows),
        "eligible_parent_count": len(eligible),
        "excluded_parent_count": len(pair_rows) - len(eligible),
        "action_count": len(action_rows),
        "gloss_token_count": token_count,
        "exclusion_reason_counts": dict(Counter(reason for row in pair_rows for reason in row["v2_1_exclusion_reasons"])),
        "v2_1": {
            "mean_normalized_average_lagging": round(mean(row["metrics"]["word_position_normalized_average_lagging"] for row in eligible), 6),
            "mean_semantic_gap_nats": round(mean(row["metrics"]["mean_scheduled_to_full_logprob_gap_nats"] for row in eligible), 6),
            "semantic_sufficiency_ratio": round(sum(row["metrics"]["semantic_sufficient_token_count"] for row in eligible) / max(1, token_count), 6),
            "mean_full_source_emit_tokens_per_parent": round(mean(row["metrics"]["full_source_emit_token_count"] for row in eligible), 6),
            "mean_bounded_delay_rescue_tokens_per_parent": round(mean(row["metrics"]["bounded_delay_rescue_token_count"] for row in eligible), 6),
            "late_negation_violation_count": sum(row["metrics"]["late_negation_violation_count"] for row in eligible),
        },
        "important_limitations": config["important_limitations"],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    with args.preview.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(preview_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(preview_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
