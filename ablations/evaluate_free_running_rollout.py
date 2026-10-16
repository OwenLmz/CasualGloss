#!/usr/bin/env python3
"""Run sentence-level causal-action inference using the model's own history."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

import torch
from nltk.tokenize import TreebankWordDetokenizer

from causal_action_common import (
    action_class,
    normalize_action,
    read_json,
    read_jsonl,
    set_seed,
    write_json,
    write_jsonl,
)


def configured_dtype(config: dict[str, Any]) -> torch.dtype:
    precision = str(config.get("precision", "bfloat16")).lower()
    if precision in {"float16", "fp16"}:
        return torch.float16
    if precision in {"bfloat16", "bf16"}:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bfloat16 was requested on a GPU without bf16 support")
        return torch.bfloat16
    raise ValueError(f"Unsupported precision: {precision}")


DETOKENIZER = TreebankWordDetokenizer()


def state_prompt(source_tokens: list[str], prefix_length: int, committed: list[str]) -> str:
    source = DETOKENIZER.detokenize(source_tokens[:prefix_length])
    committed_text = " ".join(committed) if committed else "<EMPTY>"
    status = "COMPLETE" if prefix_length >= len(source_tokens) else "INCOMPLETE"
    return (
        "Observed English prefix:\n"
        f"{source}\n\n"
        "Already committed pseudo ASL gloss:\n"
        f"{committed_text}\n\n"
        f"Source status: {status}\n"
        "Return exactly one next action."
    )


def average_lagging(write_positions: list[int], source_count: int) -> tuple[float | None, float | None]:
    if not write_positions or source_count <= 0:
        return None, None
    target_count = len(write_positions)
    ratio = target_count / source_count
    tau = next(
        (index for index, position in enumerate(write_positions, 1) if position >= source_count),
        target_count,
    )
    values = [
        write_positions[target_index - 1] - (target_index - 1) / ratio
        for target_index in range(1, tau + 1)
    ]
    value = mean(values)
    return round(value, 8), round(value / source_count, 8)


def reference_average_lagging(write_positions: list[int], source_count: int, reference_count: int) -> float | None:
    if not write_positions or source_count <= 0 or reference_count <= 0:
        return None
    ratio = reference_count / source_count
    tau = next(
        (index for index, position in enumerate(write_positions, 1) if position >= source_count),
        len(write_positions),
    )
    return round(
        mean(write_positions[index - 1] - (index - 1) / ratio for index in range(1, tau + 1)),
        8,
    )


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
        mean(write_positions[index - 1] - (index - 1) / ratio for index in range(1, tau + 1)),
        8,
    )


def average_proportion(write_positions: list[int], source_count: int) -> float | None:
    if not write_positions or source_count <= 0:
        return None
    return round(sum(write_positions) / (source_count * len(write_positions)), 8)


def differentiable_average_lagging(
    write_positions: list[int], source_count: int, reference_count: int
) -> float | None:
    if not write_positions or source_count <= 0 or reference_count <= 0:
        return None
    ratio = reference_count / source_count
    adjusted = [float(write_positions[0])]
    for position in write_positions[1:]:
        adjusted.append(max(float(position), adjusted[-1] + 1.0 / ratio))
    return round(mean(value - index / ratio for index, value in enumerate(adjusted)), 8)


def token_overlap(target: list[str], prediction: list[str]) -> tuple[float, float, float]:
    target_counter = Counter(target)
    prediction_counter = Counter(prediction)
    overlap = sum((target_counter & prediction_counter).values())
    precision = overlap / max(1, len(prediction))
    recall = overlap / max(1, len(target))
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return precision, recall, f1


def initialize_state(pair: dict[str, Any]) -> dict[str, Any]:
    source_tokens = pair["source_tokens"]
    target_tokens = pair["target_pseudo_asl_gloss_tokens"]
    return {
        "id": pair["id"],
        "source_text_en": pair["source_text_en"],
        "source_tokens": source_tokens,
        "target_tokens": target_tokens,
        "prefix_length": 1,
        "committed": [],
        "write_positions": [],
        "events": [],
        "done": False,
        "termination_reason": None,
        "premature_end": False,
        "max_actions": len(source_tokens) + max(20, 3 * len(target_tokens) + 10),
    }


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts, config, device):
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=int(config["max_length"]),
        return_tensors="pt",
    ).to(device)
    generated = model.generate(
        **encoded,
        max_new_tokens=int(config["max_new_tokens"]),
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = generated[:, encoded["input_ids"].shape[1] :]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def finalize_state(state: dict[str, Any], stage: str) -> dict[str, Any]:
    target = state["target_tokens"]
    prediction = state["committed"]
    similarity = SequenceMatcher(None, target, prediction).ratio()
    precision, recall, f1 = token_overlap(target, prediction)
    al, normalized_al = average_lagging(state["write_positions"], len(state["source_tokens"]))
    reference_al = reference_average_lagging(
        state["write_positions"], len(state["source_tokens"]), len(target)
    )
    laal = length_adaptive_average_lagging(
        state["write_positions"], len(state["source_tokens"]), len(target)
    )
    ap = average_proportion(state["write_positions"], len(state["source_tokens"]))
    dal = differentiable_average_lagging(
        state["write_positions"], len(state["source_tokens"]), len(target)
    )
    action_counts = Counter(event["predicted_action_class"] for event in state["events"])
    adjacent_repetitions = sum(
        prediction[index] == prediction[index - 1] for index in range(1, len(prediction))
    )
    ended = state["termination_reason"] in {"end_after_full_source", "premature_end"}
    completed_after_full_source = state["termination_reason"] == "end_after_full_source"
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
        "termination_reason": state["termination_reason"],
        "completed_with_end": ended,
        "completed_after_full_source": completed_after_full_source,
        "premature_end": state["premature_end"],
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
        "invalid_action_ratio": round(action_counts["INVALID"] / max(1, len(state["events"])), 8),
        "word_position_average_lagging_source_tokens": al,
        "word_position_normalized_average_lagging": normalized_al,
        "reference_length_average_lagging_source_tokens": reference_al,
        "reference_length_normalized_average_lagging": (
            round(reference_al / len(state["source_tokens"]), 8)
            if reference_al is not None and state["source_tokens"]
            else None
        ),
        "length_adaptive_average_lagging_source_tokens": laal,
        "length_adaptive_normalized_average_lagging": (
            round(laal / len(state["source_tokens"]), 8)
            if laal is not None and state["source_tokens"]
            else None
        ),
        "average_proportion": ap,
        "differentiable_average_lagging_source_tokens": dal,
        "rollout_events": state["events"],
    }


def safe_mean(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    return round(mean(filtered), 8) if filtered else None


def safe_median(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    return round(median(filtered), 8) if filtered else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-splits", type=Path, required=True)
    parser.add_argument("--causal-pairs", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--generation-batch-size", type=int, default=8)
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = read_json(args.config)
    set_seed(int(config["seed"]))
    parent_rows = read_jsonl(args.parent_splits)
    selected_ids = [
        row["id"]
        for row in parent_rows
        if row["split"] == args.split and row.get("eligible_for_finetune2", True)
    ]
    pair_by_id = {row["id"]: row for row in read_jsonl(args.causal_pairs)}
    states = [initialize_state(pair_by_id[parent_id]) for parent_id in selected_ids]

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["model_revision"], trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        dtype=configured_dtype(config),
        attn_implementation=config["attention_implementation"],
        device_map={"": args.device},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(base_model, str(args.adapter_dir), is_trainable=False)
    model.eval()
    model.config.use_cache = True

    start_time = perf_counter()
    cycle = 0
    while any(not state["done"] for state in states):
        active = [state for state in states if not state["done"]]
        for batch_start in range(0, len(active), args.generation_batch_size):
            batch_states = active[batch_start : batch_start + args.generation_batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": config["system_prompt"]},
                        {
                            "role": "user",
                            "content": state_prompt(
                                state["source_tokens"],
                                state["prefix_length"],
                                state["committed"],
                            ),
                        },
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for state in batch_states
            ]
            raw_outputs = generate_batch(model, tokenizer, prompts, config, args.device)
            for state, raw_output in zip(batch_states, raw_outputs):
                action = normalize_action(raw_output)
                action_type = action_class(action, config["wait_token"], config["end_token"])
                event = {
                    "action_index": len(state["events"]),
                    "source_prefix_length": state["prefix_length"],
                    "source_complete": state["prefix_length"] >= len(state["source_tokens"]),
                    "committed_gloss_before": " ".join(state["committed"]),
                    "raw_prediction": raw_output.strip(),
                    "predicted_action": action,
                    "predicted_action_class": action_type,
                }
                state["events"].append(event)
                if action_type == "END":
                    state["premature_end"] = state["prefix_length"] < len(state["source_tokens"])
                    state["termination_reason"] = (
                        "premature_end" if state["premature_end"] else "end_after_full_source"
                    )
                    state["done"] = True
                elif action_type == "WRITE":
                    state["committed"].append(action)
                    state["write_positions"].append(state["prefix_length"])
                elif action_type == "WAIT":
                    if state["prefix_length"] < len(state["source_tokens"]):
                        state["prefix_length"] += 1
                elif action_type == "INVALID" and state["prefix_length"] < len(state["source_tokens"]):
                    # Fail-safe progress prevents malformed outputs from creating an infinite loop.
                    state["prefix_length"] += 1
                if not state["done"] and len(state["events"]) >= state["max_actions"]:
                    state["termination_reason"] = "maximum_actions_reached"
                    state["done"] = True
        cycle += 1
        print(
            f"[rollout:{args.stage}] cycle={cycle} completed="
            f"{sum(state['done'] for state in states)}/{len(states)}",
            flush=True,
        )

    elapsed = perf_counter() - start_time
    output_rows = [finalize_state(state, args.stage) for state in states]
    write_jsonl(args.output_dir / "sentence_rollouts.jsonl", output_rows)

    termination_counts = Counter(row["termination_reason"] for row in output_rows)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "free_running_pseudo_gloss_rollout_not_gold_asl",
        "stage": args.stage,
        "adapter_dir": str(args.adapter_dir),
        "parent_count": len(output_rows),
        "rollout_history_uses_gold_committed_gloss": False,
        "policy": {
            "initial_source_prefix_tokens": 1,
            "before_source_end": (
                "WRITE commits one gloss token without consuming new source; WAIT reads exactly "
                "one new source token, so multiple WRITEs may form a burst at one prefix"
            ),
            "after_source_end": "continue actions at full source until END or limit",
            "invalid_action": "record and consume one source token as a fail-safe before source end",
            "maximum_actions": "source_length + max(20, 3 * target_length + 10)",
        },
        "termination_reason_counts": dict(termination_counts),
        "end_completion_rate": round(
            sum(row["completed_with_end"] for row in output_rows) / len(output_rows), 8
        ),
        "valid_end_completion_rate": round(
            sum(row["completed_after_full_source"] for row in output_rows)
            / len(output_rows),
            8,
        ),
        "premature_end_rate": round(
            sum(row["premature_end"] for row in output_rows) / len(output_rows), 8
        ),
        "maximum_actions_reached_rate": round(
            termination_counts["maximum_actions_reached"] / len(output_rows), 8
        ),
        "pseudo_gloss_exact_match_rate": round(
            sum(row["pseudo_gloss_exact_match"] for row in output_rows) / len(output_rows), 8
        ),
        "mean_pseudo_gloss_sequence_similarity": safe_mean(
            [row["pseudo_gloss_sequence_similarity"] for row in output_rows]
        ),
        "mean_pseudo_gloss_token_precision": safe_mean(
            [row["pseudo_gloss_token_precision"] for row in output_rows]
        ),
        "mean_pseudo_gloss_token_recall": safe_mean(
            [row["pseudo_gloss_token_recall"] for row in output_rows]
        ),
        "mean_pseudo_gloss_token_f1": safe_mean(
            [row["pseudo_gloss_token_f1"] for row in output_rows]
        ),
        "mean_hallucinated_token_ratio": safe_mean(
            [row["hallucinated_token_ratio"] for row in output_rows]
        ),
        "mean_adjacent_repetition_ratio": safe_mean(
            [row["adjacent_repetition_ratio"] for row in output_rows]
        ),
        "mean_wait_action_ratio": safe_mean([row["wait_action_ratio"] for row in output_rows]),
        "mean_invalid_action_ratio": safe_mean(
            [row["invalid_action_ratio"] for row in output_rows]
        ),
        "mean_word_position_average_lagging_source_tokens": safe_mean(
            [row["word_position_average_lagging_source_tokens"] for row in output_rows]
        ),
        "median_word_position_average_lagging_source_tokens": safe_median(
            [row["word_position_average_lagging_source_tokens"] for row in output_rows]
        ),
        "mean_word_position_normalized_average_lagging": safe_mean(
            [row["word_position_normalized_average_lagging"] for row in output_rows]
        ),
        "mean_reference_length_average_lagging_source_tokens": safe_mean(
            [row["reference_length_average_lagging_source_tokens"] for row in output_rows]
        ),
        "mean_reference_length_normalized_average_lagging": safe_mean(
            [row["reference_length_normalized_average_lagging"] for row in output_rows]
        ),
        "mean_length_adaptive_average_lagging_source_tokens": safe_mean(
            [row["length_adaptive_average_lagging_source_tokens"] for row in output_rows]
        ),
        "mean_length_adaptive_normalized_average_lagging": safe_mean(
            [row["length_adaptive_normalized_average_lagging"] for row in output_rows]
        ),
        "mean_average_proportion": safe_mean([row["average_proportion"] for row in output_rows]),
        "mean_differentiable_average_lagging_source_tokens": safe_mean(
            [row["differentiable_average_lagging_source_tokens"] for row in output_rows]
        ),
        "mean_action_count": safe_mean([row["action_count"] for row in output_rows]),
        "elapsed_seconds": round(elapsed, 4),
        "important_limitations": [
            "The target gloss is an LLM-generated pseudo label.",
            "Causal timing is pseudo-aligned rather than video-verified gold timing.",
            "The rollout starts from English text prefixes, not streaming speech chunks.",
        ],
    }
    write_json(args.output_dir / "rollout_metrics.json", report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "rollout_preview.tsv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "id",
            "source_text_en",
            "target_pseudo_asl_gloss",
            "predicted_pseudo_asl_gloss",
            "termination_reason",
            "pseudo_gloss_sequence_similarity",
            "pseudo_gloss_token_f1",
            "wait_action_ratio",
            "word_position_average_lagging_source_tokens",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
