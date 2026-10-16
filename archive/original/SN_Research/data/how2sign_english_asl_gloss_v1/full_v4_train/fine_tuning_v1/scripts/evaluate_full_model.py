#!/usr/bin/env python3
"""Evaluate the base model or a saved LoRA adapter on fixed held-out splits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch

from finetune_common import (
    CausalCollator,
    GlossDataset,
    batched_generation_metrics,
    file_sha256,
    read_json,
    read_jsonl,
    set_seed,
    teacher_forced_metrics_single,
    write_json,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--validation-generation-manifest", type=Path, required=True)
    parser.add_argument("--test-generation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = read_json(args.config)
    set_seed(int(config["seed"]))
    validation_rows = read_jsonl(args.validation_manifest)
    test_rows = read_jsonl(args.test_manifest)
    generation_rows = {
        "validation": read_jsonl(args.validation_generation_manifest),
        "test": read_jsonl(args.test_generation_manifest),
    }
    for row in validation_rows + test_rows:
        if row.get("pseudo_gloss_is_gold") is not False:
            raise SystemExit(f"Non-pseudo target flag for {row.get('id')}")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        dtype=torch.bfloat16,
        attn_implementation=config["attention_implementation"],
        device_map={"": args.device},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    if args.adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, str(args.adapter_dir), is_trainable=False
        )
    model.config.use_cache = True

    system_prompt = str(config["system_prompt"])
    collator = CausalCollator(int(tokenizer.pad_token_id))
    datasets = {
        "validation": GlossDataset(
            validation_rows, tokenizer, system_prompt, int(config["max_length"])
        ),
        "test": GlossDataset(
            test_rows, tokenizer, system_prompt, int(config["max_length"])
        ),
    }
    teacher_forced = {
        split: teacher_forced_metrics_single(
            model,
            datasets[split],
            collator,
            args.device,
            batch_size=int(config["micro_batch_size_per_gpu"]),
        )
        for split in ("validation", "test")
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generation = {}
    for split in ("validation", "test"):
        metrics, predictions = batched_generation_metrics(
            model,
            tokenizer,
            generation_rows[split],
            system_prompt,
            args.device,
            int(config["max_new_tokens"]),
            args.generation_batch_size,
            stage=args.stage,
            split=split,
        )
        generation[split] = metrics
        write_jsonl(args.output_dir / f"{split}_generation_predictions.jsonl", predictions)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "pseudo_label_evaluation_not_gold_asl",
        "stage": args.stage,
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "device": args.device,
        "generation_batch_size": args.generation_batch_size,
        "counts": {
            "validation_teacher_forced": len(validation_rows),
            "test_teacher_forced": len(test_rows),
            "validation_generation": len(generation_rows["validation"]),
            "test_generation": len(generation_rows["test"]),
        },
        "dataset_tokenization": {
            split: datasets[split].statistics() for split in datasets
        },
        "manifest_sha256": {
            "validation": file_sha256(args.validation_manifest),
            "test": file_sha256(args.test_manifest),
            "validation_generation": file_sha256(
                args.validation_generation_manifest
            ),
            "test_generation": file_sha256(args.test_generation_manifest),
        },
        "teacher_forced": teacher_forced,
        "generation": generation,
        "important_limitations": [
            "Targets are generated pseudo ASL glosses, not gold annotations.",
            "Metrics quantify pseudo-label agreement, not ASL correctness.",
            "No qualified ASL or sign-video alignment evaluation is included.",
        ],
    }
    write_json(args.output_dir / "metrics_summary.json", report)
    print(report)


if __name__ == "__main__":
    main()

