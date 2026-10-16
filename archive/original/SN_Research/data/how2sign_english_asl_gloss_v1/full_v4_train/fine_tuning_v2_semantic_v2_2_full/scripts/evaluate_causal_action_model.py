#!/usr/bin/env python3
"""Evaluate a LoRA adapter on fixed held-out pseudo causal actions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from causal_action_common import (
    ActionCollator,
    ActionDataset,
    action_class,
    file_sha256,
    generation_metrics,
    normalize_action,
    read_json,
    read_jsonl,
    set_seed,
    teacher_forced_metrics,
    write_json,
    write_jsonl,
)


@torch.inference_mode()
def generate_actions(
    model,
    tokenizer,
    rows,
    config,
    device,
    batch_size,
    stage,
):
    model.eval()
    model.config.use_cache = True
    tokenizer.padding_side = "left"
    predictions = []
    wait_token = config["wait_token"]
    end_token = config["end_token"]
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": config["system_prompt"]},
                    {"role": "user", "content": row["state_prompt"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch_rows
        ]
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
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for row, raw in zip(batch_rows, decoded):
            predicted = normalize_action(raw)
            target = normalize_action(row["target_action"])
            target_class = action_class(target, wait_token, end_token)
            predicted_class = action_class(predicted, wait_token, end_token)
            predictions.append(
                {
                    "stage": stage,
                    "id": row["id"],
                    "parent_id": row["parent_id"],
                    "source_prefix_length": row["source_prefix_length"],
                    "source_token_count": row["source_token_count"],
                    "source_complete": row["source_complete"],
                    "target_action": target,
                    "predicted_action": predicted,
                    "raw_prediction": raw.strip(),
                    "target_action_class": target_class,
                    "predicted_action_class": predicted_class,
                    "exact_action_match": predicted == target,
                    "target_is_gold": False,
                    "alignment_is_gold": False,
                }
            )
        if (start // batch_size + 1) % 20 == 0 or start + batch_size >= len(rows):
            print(f"[evaluate:{stage}] {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--generation-batch-size", type=int, default=8)
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = read_json(args.config)
    rows = read_jsonl(args.manifest)
    set_seed(int(config["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["model_revision"], trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
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
    dataset = ActionDataset(rows, tokenizer, config["system_prompt"], int(config["max_length"]))
    collator = ActionCollator(int(tokenizer.pad_token_id))
    teacher = teacher_forced_metrics(
        model,
        dataset,
        collator,
        args.device,
        int(config.get("evaluation_batch_size_per_gpu", config["micro_batch_size_per_gpu"])),
    )
    predictions = generate_actions(
        model,
        tokenizer,
        rows,
        config,
        args.device,
        args.generation_batch_size,
        args.stage,
    )
    generation = generation_metrics(
        predictions, config["wait_token"], config["end_token"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "test_action_predictions.jsonl", predictions)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "pseudo_action_evaluation_not_gold_asl_or_gold_timing",
        "stage": args.stage,
        "adapter_dir": str(args.adapter_dir),
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "record_count": len(rows),
        "parent_count": len({row["parent_id"] for row in rows}),
        "dataset_tokenization": dataset.statistics(),
        "teacher_forced": teacher,
        "greedy_generation_on_gold_states": generation,
        "evaluation_scope": (
            "Each prediction uses the gold observed prefix and gold committed pseudo gloss. "
            "This is not free-running sentence rollout evaluation."
        ),
        "important_limitations": [
            "Action targets come from pseudo gloss and pseudo alignment.",
            "Metrics do not measure ASL linguistic correctness.",
            "Latency-related errors are measured on gold states, not autonomous rollout.",
        ],
    }
    write_json(args.output_dir / "metrics_summary.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
