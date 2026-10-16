#!/usr/bin/env python3
"""Run an existing causal-action adapter on CLERC Epee English prompts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
    average_proportion,
    generate_batch,
    length_adaptive_average_lagging,
    state_prompt,
)


def initialize_state(source: dict[str, Any]) -> dict[str, Any]:
    source_tokens = source["source_tokens"]
    max_ref_length = max(
        [len(ref["content_tokens"]) for ref in source["references"]] or [1]
    )
    return {
        "id": source["id"],
        "phrase_id": source["phrase_id"],
        "evaluation_subset": source["evaluation_subset"],
        "source_text_en": source["source_text_en"],
        "source_tokens": source_tokens,
        "reference_count": source["reference_count"],
        "median_reference_length": source["median_content_reference_length"],
        "prefix_length": 1,
        "committed": [],
        "write_positions": [],
        "events": [],
        "done": False,
        "termination_reason": None,
        "premature_end": False,
        "max_actions": len(source_tokens) + max(30, 4 * max_ref_length + 10),
    }


def finalize(state: dict[str, Any], stage: str) -> dict[str, Any]:
    event_types = Counter(event["predicted_action_class"] for event in state["events"])
    source_count = len(state["source_tokens"])
    laal = length_adaptive_average_lagging(
        state["write_positions"], source_count, state["median_reference_length"]
    )
    return {
        "stage": stage,
        "id": state["id"],
        "phrase_id": state["phrase_id"],
        "evaluation_subset": state["evaluation_subset"],
        "source_text_en": state["source_text_en"],
        "source_tokens": state["source_tokens"],
        "reference_count": state["reference_count"],
        "median_content_reference_length": state["median_reference_length"],
        "predicted_gloss": " ".join(state["committed"]),
        "predicted_gloss_tokens": state["committed"],
        "write_source_positions": state["write_positions"],
        "termination_reason": state["termination_reason"],
        "completed_after_full_source": state["termination_reason"] == "end_after_full_source",
        "premature_end": state["premature_end"],
        "action_count": len(state["events"]),
        "action_class_counts": dict(event_types),
        "invalid_action_ratio": event_types["INVALID"] / max(1, len(state["events"])),
        "length_adaptive_average_lagging_source_tokens": laal,
        "average_proportion": average_proportion(state["write_positions"], source_count),
        "rollout_events": state["events"],
        "target_is_external_human_reference": True,
        "rollout_history_uses_reference_gloss": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--generation-batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = read_json(args.config)
    sources = read_jsonl(args.manifest)
    if len(sources) != 401 or len({row["id"] for row in sources}) != 401:
        raise ValueError("Expected exactly 401 unique Epee source phrases")
    set_seed(int(config["seed"]))

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["model_revision"], trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        dtype=torch.bfloat16,
        attn_implementation=config["attention_implementation"],
        device_map={"": args.device},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(base_model, str(args.adapter_dir), is_trainable=False)
    model.eval()
    model.config.use_cache = True

    states = [initialize_state(source) for source in sources]
    started = perf_counter()
    cycle = 0
    while any(not state["done"] for state in states):
        active = [state for state in states if not state["done"]]
        for start in range(0, len(active), args.generation_batch_size):
            batch = active[start : start + args.generation_batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": config["system_prompt"]},
                        {
                            "role": "user",
                            "content": state_prompt(
                                state["source_tokens"], state["prefix_length"], state["committed"]
                            ),
                        },
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for state in batch
            ]
            raw_outputs = generate_batch(model, tokenizer, prompts, config, args.device)
            for state, raw_output in zip(batch, raw_outputs):
                action = normalize_action(raw_output)
                action_type = action_class(action, config["wait_token"], config["end_token"])
                prefix_before = state["prefix_length"]
                event = {
                    "action_index": len(state["events"]),
                    "source_prefix_length": prefix_before,
                    "source_complete": prefix_before >= len(state["source_tokens"]),
                    "committed_gloss_before": " ".join(state["committed"]),
                    "raw_prediction": raw_output.strip(),
                    "predicted_action": action,
                    "predicted_action_class": action_type,
                    "source_advanced_after_action": False,
                }
                if action_type == "END":
                    state["premature_end"] = prefix_before < len(state["source_tokens"])
                    state["termination_reason"] = (
                        "premature_end" if state["premature_end"] else "end_after_full_source"
                    )
                    state["done"] = True
                elif action_type == "WRITE":
                    state["committed"].append(action)
                    state["write_positions"].append(prefix_before)
                elif action_type in {"WAIT", "INVALID"} and prefix_before < len(state["source_tokens"]):
                    state["prefix_length"] += 1
                    event["source_advanced_after_action"] = True
                event["source_prefix_length_after"] = state["prefix_length"]
                event["committed_gloss_after"] = " ".join(state["committed"])
                state["events"].append(event)
                if not state["done"] and len(state["events"]) >= state["max_actions"]:
                    state["termination_reason"] = "maximum_actions_reached"
                    state["done"] = True
        cycle += 1
        print(
            f"[epee:{args.stage}] cycle={cycle} completed={sum(s['done'] for s in states)}/401",
            flush=True,
        )

    rows = [finalize(state, args.stage) for state in states]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "sentence_rollouts.jsonl", rows)
    termination = Counter(row["termination_reason"] for row in rows)
    report = {
        "scientific_status": "external_human_asl_reference_free_running_rollout",
        "stage": args.stage,
        "source_count": len(rows),
        "adapter_dir": str(args.adapter_dir),
        "policy": "native learned WAIT/WRITE, initial English prefix=1",
        "termination_reason_counts": dict(termination),
        "valid_completion_rate": sum(row["completed_after_full_source"] for row in rows) / len(rows),
        "mean_invalid_action_ratio": mean(row["invalid_action_ratio"] for row in rows),
        "mean_laal_source_tokens": mean(
            row["length_adaptive_average_lagging_source_tokens"]
            for row in rows
            if row["length_adaptive_average_lagging_source_tokens"] is not None
        ),
        "elapsed_seconds": round(perf_counter() - started, 4),
        "important_limitations": [
            "Epee is external and out of domain relative to How2Sign training.",
            "English token latency is reported, not speech-time latency.",
            "No Epee reference is used in model history or decoding.",
        ],
    }
    write_json(args.output_dir / "rollout_metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

