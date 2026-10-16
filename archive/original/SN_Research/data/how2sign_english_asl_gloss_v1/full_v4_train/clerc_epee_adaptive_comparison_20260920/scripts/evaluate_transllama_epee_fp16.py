#!/usr/bin/env python3
"""Run the frozen TransLLaMA-style policy on Epee without reference exposure."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
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


DATA_ROOT = Path(__file__).resolve().parents[2]
TRANS_SCRIPTS = DATA_ROOT / "transllama_learned_adaptive_20260918" / "scripts"
sys.path.insert(0, str(TRANS_SCRIPTS))

from evaluate_transllama import (  # noqa: E402
    apply_action,
    generate_actions,
    native_context_window,
    parse_k_values,
)
from transllama_training import set_seed, wait_token_ids  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--k-values", type=parse_k_values, default=parse_k_values("1,2,3,4,5,6,7,8"))
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--action-max-new-tokens", type=int, default=16)
    return parser.parse_args()


def initialize_learned_state(source: dict, wait_k: int) -> dict:
    state = initialize_external_state(source, wait_k=wait_k)
    state["target_token_ids"] = []
    state["pending_token_ids"] = []
    return state


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output path: {args.output_dir}")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = read_json(args.config)
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
    waits = wait_token_ids(tokenizer)
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
    context_window = native_context_window(base_model, tokenizer)
    model = PeftModel.from_pretrained(
        base_model, str(args.adapter_dir), is_trainable=False
    )
    model.eval()
    model.config.use_cache = True

    states = [
        initialize_learned_state(source, wait_k)
        for wait_k in args.k_values
        for source in sources
    ]
    started = perf_counter()
    cycle = 0
    context_stats = {"max_observed_prompt_tokens": 0}
    while any(not state["done"] for state in states):
        active = [state for state in states if not state["done"]]
        results = generate_actions(
            model,
            tokenizer,
            active,
            config["system_prompt"],
            waits,
            args.generation_batch_size,
            args.action_max_new_tokens,
            context_window,
            context_stats,
            device,
        )
        for (
            state,
            action,
            action_type,
            raw,
            accepted_ids,
            pending_ids,
            finish_after_write,
        ) in results:
            apply_action(
                state,
                action,
                action_type,
                raw,
                accepted_ids,
                pending_ids,
                finish_after_write,
                tokenizer,
            )
        cycle += 1
        if cycle == 1 or cycle % 10 == 0:
            print(
                f"[{args.stage}] cycle={cycle} completed="
                f"{sum(state['done'] for state in states)}/{len(states)}",
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    sweep = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "zero_shot_external_human_gloss_reference_rollout",
        "stage": args.stage,
        "system_family": "transllama_style",
        "training_seed": training_seed,
        "adapter_dir": str(args.adapter_dir),
        "k_values": args.k_values,
        "confirmatory_k": 5,
        "inference_dtype": "float16",
        "reference_exposed_to_model": False,
        "shared_action_budget": "source_length + max(30, 4 * max_human_reference_length + 10)",
        "generation": {
            "decoding": "greedy",
            "modified_wait_k": "minimum initial read only",
            "normal_gloss": "buffer subtokens and commit one complete gloss without reading source",
            "wait": "read exactly one source token",
            "eos_before_source_end": "suppressed",
            "wait_after_source_end": "suppressed",
            "invalid_action": "record and read one source token before completion",
            "native_inference_context_window": context_window,
            "max_observed_prompt_tokens": context_stats["max_observed_prompt_tokens"],
        },
        "manifest_sha256": file_sha256(args.manifest),
        "config_sha256": file_sha256(args.config),
        "elapsed_seconds": round(perf_counter() - started, 3),
        "results": [],
    }
    for wait_k in args.k_values:
        rows = [
            finalize_external_state(
                state,
                stage=f"{args.stage}_k{wait_k}",
                system_family="transllama_style",
                training_seed=training_seed,
                adapter_dir=args.adapter_dir,
            )
            for state in states
            if state["wait_k"] == wait_k
        ]
        output = args.output_dir / f"k{wait_k}"
        write_jsonl(output / "sentence_rollouts.jsonl", rows)
        metrics = {
            "stage": f"{args.stage}_k{wait_k}",
            "wait_k": wait_k,
            "training_seed": training_seed,
            "inference_dtype": "float16",
            "reference_exposed_to_model": False,
            **summarize_rollouts(rows),
        }
        write_json(output / "rollout_metrics.json", metrics)
        sweep["results"].append(metrics)
    write_json(args.output_dir / "sweep_summary.json", sweep)
    print(json.dumps(sweep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
