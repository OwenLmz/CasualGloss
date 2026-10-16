#!/usr/bin/env python3
"""Evaluate a trained causal-action model with TransLLaMA-style decoding."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import torch


DATA_ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = DATA_ROOT / "fine_tuning_v2_semantic_v2_2_full" / "scripts"
sys.path.insert(0, str(COMMON_DIR))

from causal_action_common import (  # noqa: E402
    action_class,
    normalize_action,
    read_json,
    read_jsonl,
    set_seed,
    write_json,
    write_jsonl,
)
from evaluate_free_running_rollout import (  # noqa: E402
    finalize_state,
    generate_batch,
    safe_mean,
    safe_median,
    state_prompt,
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


def parse_k_values(value: str) -> list[int]:
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not values or values[0] < 1:
        raise argparse.ArgumentTypeError("k values must be positive integers")
    return values


def initialize_state(pair: dict[str, Any], wait_k: int) -> dict[str, Any]:
    source_tokens = pair["source_tokens"]
    target_tokens = pair["target_pseudo_asl_gloss_tokens"]
    if not source_tokens:
        raise ValueError(f"Empty source tokens for {pair['id']}")
    return {
        "id": pair["id"],
        "source_text_en": pair["source_text_en"],
        "source_tokens": source_tokens,
        "target_tokens": target_tokens,
        "wait_k": wait_k,
        "prefix_length": min(wait_k, len(source_tokens)),
        "committed": [],
        "write_positions": [],
        "events": [],
        "done": False,
        "termination_reason": None,
        "premature_end": False,
        "max_actions": len(source_tokens) + max(20, 3 * len(target_tokens) + 10),
    }


def apply_action(
    state: dict[str, Any],
    action: str,
    action_type: str,
    raw_output: str,
) -> None:
    source_count = len(state["source_tokens"])
    prefix_before = state["prefix_length"]
    event = {
        "action_index": len(state["events"]),
        "wait_k": state["wait_k"],
        "source_prefix_length": prefix_before,
        "source_complete": prefix_before >= source_count,
        "committed_gloss_before": " ".join(state["committed"]),
        "raw_prediction": raw_output.strip(),
        "predicted_action": action,
        "predicted_action_class": action_type,
        "source_advanced_after_action": False,
    }

    if action_type == "END":
        state["premature_end"] = prefix_before < source_count
        state["termination_reason"] = (
            "premature_end" if state["premature_end"] else "end_after_full_source"
        )
        state["done"] = True
    elif action_type == "WRITE":
        state["committed"].append(action)
        state["write_positions"].append(prefix_before)
    elif action_type == "WAIT":
        if prefix_before < source_count:
            state["prefix_length"] += 1
            event["source_advanced_after_action"] = True
        else:
            event["wait_at_full_source"] = True
    elif action_type == "INVALID":
        if prefix_before < source_count:
            state["prefix_length"] += 1
            event["source_advanced_after_action"] = True

    event["source_prefix_length_after"] = state["prefix_length"]
    event["committed_gloss_after"] = " ".join(state["committed"])
    state["events"].append(event)
    if not state["done"] and len(state["events"]) >= state["max_actions"]:
        state["termination_reason"] = "maximum_actions_reached"
        state["done"] = True


def summarize_rows(
    rows: list[dict[str, Any]],
    stage: str,
    adapter_dir: Path,
    wait_k: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    termination_counts = Counter(row["termination_reason"] for row in rows)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "free_running_pseudo_gloss_rollout_not_gold_asl",
        "stage": stage,
        "adapter_dir": str(adapter_dir),
        "parent_count": len(rows),
        "rollout_history_uses_gold_committed_gloss": False,
        "policy": {
            "name": "transllama_style_wait_k_inference",
            "initial_source_prefix_tokens": wait_k,
            "normal_gloss_action": "WRITE one gloss token; source prefix is unchanged",
            "wait_action": "READ one source token without emitting gloss",
            "after_source_end": "continue model actions until END or action limit",
            "invalid_action": (
                "record and READ one source token as a finite-progress safeguard"
            ),
            "maximum_actions": "source_length + max(20, 3 * target_length + 10)",
        },
        "termination_reason_counts": dict(termination_counts),
        "end_completion_rate": round(
            sum(row["completed_with_end"] for row in rows) / max(1, len(rows)), 8
        ),
        "valid_end_completion_rate": round(
            sum(row["completed_after_full_source"] for row in rows) / max(1, len(rows)),
            8,
        ),
        "premature_end_rate": round(
            sum(row["premature_end"] for row in rows) / max(1, len(rows)), 8
        ),
        "maximum_actions_reached_rate": round(
            termination_counts["maximum_actions_reached"] / max(1, len(rows)), 8
        ),
        "pseudo_gloss_exact_match_rate": round(
            sum(row["pseudo_gloss_exact_match"] for row in rows) / max(1, len(rows)), 8
        ),
        "mean_pseudo_gloss_sequence_similarity": safe_mean(
            [row["pseudo_gloss_sequence_similarity"] for row in rows]
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
        "mean_hallucinated_token_ratio": safe_mean(
            [row["hallucinated_token_ratio"] for row in rows]
        ),
        "mean_adjacent_repetition_ratio": safe_mean(
            [row["adjacent_repetition_ratio"] for row in rows]
        ),
        "mean_wait_action_ratio": safe_mean([row["wait_action_ratio"] for row in rows]),
        "mean_invalid_action_ratio": safe_mean(
            [row["invalid_action_ratio"] for row in rows]
        ),
        "mean_reference_length_average_lagging_source_tokens": safe_mean(
            [row["reference_length_average_lagging_source_tokens"] for row in rows]
        ),
        "median_reference_length_average_lagging_source_tokens": safe_median(
            [row["reference_length_average_lagging_source_tokens"] for row in rows]
        ),
        "mean_length_adaptive_average_lagging_source_tokens": safe_mean(
            [row["length_adaptive_average_lagging_source_tokens"] for row in rows]
        ),
        "median_length_adaptive_average_lagging_source_tokens": safe_median(
            [row["length_adaptive_average_lagging_source_tokens"] for row in rows]
        ),
        "mean_average_proportion": safe_mean([row["average_proportion"] for row in rows]),
        "mean_differentiable_average_lagging_source_tokens": safe_mean(
            [row["differentiable_average_lagging_source_tokens"] for row in rows]
        ),
        "mean_action_count": safe_mean([row["action_count"] for row in rows]),
        "elapsed_seconds": round(elapsed_seconds, 4),
        "important_limitations": [
            "The target gloss is an LLM-generated pseudo label.",
            "Causal timing is pseudo-aligned rather than video-verified gold timing.",
            "The rollout starts from English text prefixes, not streaming speech chunks.",
            "This run uses a learned causal-alignment adapter and TransLLaMA-style inference control; it is not an official TransLLaMA reproduction.",
        ],
    }


def write_preview(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "id",
        "source_text_en",
        "target_pseudo_asl_gloss",
        "predicted_pseudo_asl_gloss",
        "termination_reason",
        "pseudo_gloss_sequence_similarity",
        "pseudo_gloss_token_f1",
        "wait_action_ratio",
        "reference_length_average_lagging_source_tokens",
        "length_adaptive_average_lagging_source_tokens",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-splits", type=Path, required=True)
    parser.add_argument("--causal-pairs", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--k-values", type=parse_k_values, default=parse_k_values("1,2,3,4,5,6,7,8"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
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
    if args.limit is not None:
        selected_ids = selected_ids[: args.limit]
    pair_by_id = {row["id"]: row for row in read_jsonl(args.causal_pairs)}
    missing_ids = [parent_id for parent_id in selected_ids if parent_id not in pair_by_id]
    if missing_ids:
        raise ValueError(f"Missing causal pairs for {len(missing_ids)} selected parents")

    states = [
        initialize_state(pair_by_id[parent_id], wait_k)
        for wait_k in args.k_values
        for parent_id in selected_ids
    ]

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
                apply_action(state, action, action_type, raw_output)
        cycle += 1
        per_k_done = {
            wait_k: sum(
                state["done"] for state in states if state["wait_k"] == wait_k
            )
            for wait_k in args.k_values
        }
        print(
            f"[transllama-wait-k:{args.stage}] cycle={cycle} done={per_k_done}",
            flush=True,
        )

    total_elapsed = perf_counter() - start_time
    sweep_summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "split": args.split,
        "parent_count_per_k": len(selected_ids),
        "k_values": args.k_values,
        "adapter_dir": str(args.adapter_dir),
        "total_elapsed_seconds": round(total_elapsed, 4),
        "results": [],
    }
    for wait_k in args.k_values:
        k_states = [state for state in states if state["wait_k"] == wait_k]
        rows = []
        for state in k_states:
            row = finalize_state(state, f"{args.stage}_k{wait_k}")
            row["wait_k"] = wait_k
            row["inference_policy"] = "transllama_style_wait_k"
            rows.append(row)
        output_dir = args.output_dir / f"k{wait_k}"
        write_jsonl(output_dir / "sentence_rollouts.jsonl", rows)
        report = summarize_rows(
            rows,
            f"{args.stage}_k{wait_k}",
            args.adapter_dir,
            wait_k,
            total_elapsed / max(1, len(args.k_values)),
        )
        write_json(output_dir / "rollout_metrics.json", report)
        write_preview(output_dir / "rollout_preview.tsv", rows)
        sweep_summary["results"].append(report)
    write_json(args.output_dir / "sweep_summary.json", sweep_summary)
    print(json.dumps(sweep_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
