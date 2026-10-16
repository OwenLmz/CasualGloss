#!/usr/bin/env python3
"""Run a frozen trained Fixed Wait-k checkpoint on Epee under the shared protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import torch

from epee_common import (
    file_sha256,
    finalize_external_state,
    initialize_external_state,
    load_manifest,
    read_json,
    summarize_rollouts,
    write_json,
    write_jsonl,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ACTION_SCRIPTS = REPOSITORY_ROOT / "method" / "causalgloss"
sys.path.insert(0, str(ACTION_SCRIPTS))

from causal_action_common import action_class, normalize_action, set_seed  # noqa: E402
from evaluate_free_running_rollout import generate_batch, state_prompt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--system-family", choices=("fixed_wait_k",), default="fixed_wait_k")
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--generation-batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output path: {args.output_dir}")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = read_json(args.config)
    if int(config.get("wait_k", 0)) not in {1, 3, 5, 7}:
        raise ValueError("Fixed Wait-k config must declare wait_k in {1,3,5,7}")
    sources = load_manifest(args.manifest, args.expected_count)
    training_seed = int(config["seed"])
    set_seed(training_seed)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    device = torch.device("cuda", 0)
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        local_files_only=True,
        dtype=torch.float16,
        attn_implementation=config["attention_implementation"],
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(
        base_model, str(args.adapter_dir), is_trainable=False
    )
    model.eval()
    model.config.use_cache = True

    states = [initialize_external_state(source) for source in sources]
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
                                state["source_tokens"],
                                state["prefix_length"],
                                state["committed"],
                            ),
                        },
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for state in batch
            ]
            raw_outputs = generate_batch(model, tokenizer, prompts, config, device)
            for state, raw_output in zip(batch, raw_outputs):
                action = normalize_action(raw_output)
                action_type = action_class(
                    action, config["wait_token"], config["end_token"]
                )
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
                    state["termination_reason"] = (
                        "end_after_full_source"
                        if prefix_before >= len(state["source_tokens"])
                        else "premature_end"
                    )
                    state["done"] = True
                elif action_type == "WRITE":
                    state["committed"].append(action)
                    state["write_positions"].append(prefix_before)
                elif action_type in {"WAIT", "INVALID"} and prefix_before < len(
                    state["source_tokens"]
                ):
                    state["prefix_length"] += 1
                    event["source_advanced_after_action"] = True
                event["source_prefix_length_after"] = state["prefix_length"]
                event["committed_gloss_after"] = " ".join(state["committed"])
                state["events"].append(event)
                if not state["done"] and len(state["events"]) >= state["max_actions"]:
                    state["termination_reason"] = "maximum_actions_reached"
                    state["done"] = True
        cycle += 1
        if cycle == 1 or cycle % 10 == 0:
            print(
                f"[{args.stage}] cycle={cycle} completed="
                f"{sum(state['done'] for state in states)}/{len(states)}",
                flush=True,
            )

    rows = [
        finalize_external_state(
            state,
            stage=args.stage,
            system_family=args.system_family,
            training_seed=training_seed,
            adapter_dir=args.adapter_dir,
        )
        for state in states
    ]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output_dir / "sentence_rollouts.jsonl", rows)
    report = {
        "scientific_status": "zero_shot_external_human_gloss_reference_rollout",
        "stage": args.stage,
        "system_family": args.system_family,
        "wait_k": int(config["wait_k"]),
        "training_seed": training_seed,
        "adapter_dir": str(args.adapter_dir),
        "inference_dtype": "float16",
        "hardware_precision_reason": "shared V100-compatible replication stack",
        "reference_exposed_to_model": False,
        "shared_action_budget": "source_length + max(30, 4 * max_human_reference_length + 10)",
        "manifest_sha256": file_sha256(args.manifest),
        "config_sha256": file_sha256(args.config),
        "elapsed_seconds": round(perf_counter() - started, 3),
        **summarize_rollouts(rows),
    }
    write_json(args.output_dir / "rollout_metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
