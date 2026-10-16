from __future__ import annotations

import math
from collections import Counter
from difflib import SequenceMatcher
from statistics import mean
from typing import Any


def token_overlap(target: list[str], prediction: list[str]) -> tuple[float, float, float]:
    target_counter = Counter(target)
    prediction_counter = Counter(prediction)
    overlap = sum((target_counter & prediction_counter).values())
    precision = overlap / max(1, len(prediction))
    recall = overlap / max(1, len(target))
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return precision, recall, f1


def length_adaptive_average_lagging(
    write_positions: list[int], source_count: int, reference_count: int
) -> float | None:
    if not write_positions or source_count <= 0 or reference_count <= 0:
        return None
    ratio = max(reference_count, len(write_positions)) / source_count
    tau = next(
        (index for index, position in enumerate(write_positions, 1) if position >= source_count),
        len(write_positions),
    )
    return round(
        mean(
            write_positions[index - 1] - (index - 1) / ratio
            for index in range(1, tau + 1)
        ),
        8,
    )


def ngrams(tokens: list[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def corpus_bleu(rows: list[dict[str, Any]], max_order: int = 4) -> dict[str, float]:
    matches = [0] * max_order
    totals = [0] * max_order
    hypothesis_length = 0
    reference_length = 0
    for row in rows:
        hypothesis = row["predicted_pseudo_asl_gloss_tokens"]
        reference = row["target_pseudo_asl_gloss_tokens"]
        hypothesis_length += len(hypothesis)
        reference_length += len(reference)
        for order in range(1, max_order + 1):
            hypothesis_counts = ngrams(hypothesis, order)
            reference_counts = ngrams(reference, order)
            matches[order - 1] += sum(
                min(count, reference_counts[gram])
                for gram, count in hypothesis_counts.items()
            )
            totals[order - 1] += sum(hypothesis_counts.values())

    brevity_penalty = (
        1.0
        if hypothesis_length >= reference_length
        else math.exp(1.0 - reference_length / max(hypothesis_length, 1))
    )
    result = {
        "hypothesis_length": float(hypothesis_length),
        "reference_length": float(reference_length),
        "length_ratio": hypothesis_length / max(1, reference_length),
        "brevity_penalty": brevity_penalty,
    }
    for order in range(1, max_order + 1):
        precisions = [
            matches[index] / totals[index] if totals[index] else 0.0
            for index in range(order)
        ]
        score = (
            0.0
            if any(value <= 0 for value in precisions)
            else 100.0
            * brevity_penalty
            * math.exp(sum(math.log(value) for value in precisions) / order)
        )
        result[f"bleu_{order}"] = score
    return result


def finalize_state(state: dict[str, Any], stage: str) -> dict[str, Any]:
    target = list(state["target_tokens"])
    prediction = list(state["committed"])
    precision, recall, f1 = token_overlap(target, prediction)
    similarity = SequenceMatcher(None, target, prediction).ratio()
    laal = length_adaptive_average_lagging(
        state["write_positions"], len(state["source_tokens"]), len(target)
    )
    action_counts = Counter(event["predicted_action_class"] for event in state["events"])
    adjacent_repetitions = sum(
        prediction[index] == prediction[index - 1] for index in range(1, len(prediction))
    )
    return {
        "stage": stage,
        "id": state["id"],
        "source_text_en": state["source_text_en"],
        "source_tokens": state["source_tokens"],
        "target_pseudo_asl_gloss": " ".join(target),
        "target_pseudo_asl_gloss_tokens": target,
        "predicted_pseudo_asl_gloss": " ".join(prediction),
        "predicted_pseudo_asl_gloss_tokens": prediction,
        "target_is_gold": False,
        "alignment_is_gold": False,
        "rollout_history_uses_gold_committed_gloss": False,
        "wait_k": state["wait_k"],
        "termination_reason": state["termination_reason"],
        "completed_with_end": state["termination_reason"] == "end_after_full_source",
        "completed_after_full_source": state["termination_reason"] == "end_after_full_source",
        "premature_end": state["termination_reason"] == "premature_end",
        "source_token_count": len(state["source_tokens"]),
        "target_gloss_token_count": len(target),
        "predicted_gloss_token_count": len(prediction),
        "action_count": len(state["events"]),
        "max_action_count": state["max_actions"],
        "action_class_counts": dict(action_counts),
        "write_source_positions": state["write_positions"],
        "pseudo_gloss_exact_match": target == prediction,
        "pseudo_gloss_sequence_similarity": round(similarity, 8),
        "pseudo_gloss_token_precision": round(precision, 8),
        "pseudo_gloss_token_recall": round(recall, 8),
        "pseudo_gloss_token_f1": round(f1, 8),
        "hallucinated_token_ratio": round(1.0 - precision, 8),
        "adjacent_repetition_ratio": round(
            adjacent_repetitions / max(1, len(prediction) - 1), 8
        ),
        "wait_action_ratio": round(action_counts["WAIT"] / max(1, len(state["events"])), 8),
        "invalid_action_ratio": round(
            action_counts["INVALID"] / max(1, len(state["events"])), 8
        ),
        "length_adaptive_average_lagging_source_tokens": laal,
        "relative_length_error": abs(len(prediction) - len(target)) / max(1, len(target)),
        "over_generated": len(prediction) > len(target),
        "rollout_events": state["events"],
    }


def safe_mean(values: list[float | int | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    return round(mean(filtered), 8) if filtered else None


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    termination_counts = Counter(row["termination_reason"] for row in rows)
    summary = {
        "parent_count": len(rows),
        "termination_reason_counts": dict(termination_counts),
        "completion_rate": safe_mean([int(row["completed_with_end"]) for row in rows]),
        "maximum_actions_reached_rate": safe_mean(
            [int(row["termination_reason"] == "maximum_actions_reached") for row in rows]
        ),
        "mean_pseudo_gloss_token_precision": safe_mean(
            [row["pseudo_gloss_token_precision"] for row in rows]
        ),
        "mean_pseudo_gloss_token_recall": safe_mean(
            [row["pseudo_gloss_token_recall"] for row in rows]
        ),
        "mean_pseudo_gloss_token_f1": safe_mean(
            [row["pseudo_gloss_token_f1"] for row in rows]
        ),
        "mean_pseudo_gloss_sequence_similarity": safe_mean(
            [row["pseudo_gloss_sequence_similarity"] for row in rows]
        ),
        "mean_length_adaptive_average_lagging_source_tokens": safe_mean(
            [row["length_adaptive_average_lagging_source_tokens"] for row in rows]
        ),
        "mean_invalid_action_ratio": safe_mean([row["invalid_action_ratio"] for row in rows]),
        "mean_relative_length_error": safe_mean([row["relative_length_error"] for row in rows]),
        "over_generation_rate": safe_mean([int(row["over_generated"]) for row in rows]),
        "mean_predicted_gloss_token_count": safe_mean(
            [row["predicted_gloss_token_count"] for row in rows]
        ),
        "mean_target_gloss_token_count": safe_mean(
            [row["target_gloss_token_count"] for row in rows]
        ),
        "mean_action_count": safe_mean([row["action_count"] for row in rows]),
    }
    summary.update(corpus_bleu(rows))
    return summary
