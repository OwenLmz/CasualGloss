#!/usr/bin/env python3
"""Reproduce the first full-data optimizer updates and locate non-finite values."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--source-adapter-dir", type=Path, required=True)
    parser.add_argument("--shared-scripts-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention-implementation", choices=["sdpa", "eager"], required=True)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-updates", type=int, default=300)
    args = parser.parse_args()

    sys.path.insert(0, str(args.shared_scripts_dir.resolve()))
    from causal_action_common import (  # pylint: disable=import-error,import-outside-toplevel
        ActionCollator,
        ActionDataset,
        move_batch,
        read_json,
        read_jsonl,
        set_seed,
    )
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    config = read_json(args.config)
    seed = int(config["seed"])
    set_seed(seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    all_rows = read_jsonl(args.train_manifest)
    micro_batch = int(config["micro_batch_size_per_gpu"])
    grad_accum = int(config["gradient_accumulation_steps"])
    examples_needed = min(len(all_rows), args.max_updates * micro_batch * grad_accum)
    generator = torch.Generator()
    generator.manual_seed(seed + 1)  # DistributedSampler epoch=1.
    permutation = torch.randperm(len(all_rows), generator=generator).tolist()
    selected_indices = permutation[:examples_needed]
    selected_rows = [all_rows[index] for index in selected_indices]

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
        attn_implementation=args.attention_implementation,
        device_map={"": args.device},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    base_model.config.use_cache = False
    if config.get("gradient_checkpointing", True):
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base_model.enable_input_require_grads()
    model = PeftModel.from_pretrained(base_model, str(args.source_adapter_dir), is_trainable=True)
    model.train()

    dataset = ActionDataset(
        selected_rows, tokenizer, config["system_prompt"], int(config["max_length"])
    )
    loader = DataLoader(
        dataset,
        batch_size=micro_batch,
        shuffle=False,
        collate_fn=ActionCollator(int(tokenizer.pad_token_id)),
        num_workers=0,
        pin_memory=True,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    learning_rate = args.learning_rate or float(config["learning_rate"])
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=float(config["weight_decay"])
    )
    full_updates = math.ceil(math.ceil(len(all_rows) / micro_batch) / grad_accum)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(full_updates * float(config["warmup_ratio"])),
        num_training_steps=full_updates,
    )

    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "attention_implementation": args.attention_implementation,
        "learning_rate": learning_rate,
        "max_updates": args.max_updates,
        "examples_selected": len(selected_rows),
        "dataset_statistics": dataset.statistics(),
        "updates": [],
    }
    optimizer.zero_grad(set_to_none=True)
    group_losses: list[float] = []
    group_ids: list[str] = []
    global_step = 0
    for micro_index, batch in enumerate(loader):
        batch_ids = [str(value) for value in batch["ids"]]
        group_ids.extend(batch_ids)
        outputs = model(**move_batch(batch, device))
        raw_loss = outputs.loss
        if not torch.isfinite(raw_loss):
            report.update(
                status="failed",
                failure_stage="forward_loss",
                global_step_before_update=global_step,
                micro_index=micro_index,
                batch_ids=batch_ids,
                accumulation_group_ids=group_ids,
                raw_loss=float(raw_loss.detach().item()),
            )
            write_json(args.output_report, report)
            return
        (raw_loss / grad_accum).backward()
        group_losses.append(float(raw_loss.detach().item()))
        if (micro_index + 1) % grad_accum != 0:
            continue

        grad_norm = clip_grad_norm_(trainable, float(config["max_grad_norm"]))
        if not torch.isfinite(grad_norm):
            bad_gradients = [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
                and parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
            ]
            report.update(
                status="failed",
                failure_stage="gradient",
                global_step_before_update=global_step,
                micro_index=micro_index,
                accumulation_group_ids=group_ids,
                mean_raw_loss=sum(group_losses) / len(group_losses),
                grad_norm=float(grad_norm.item()),
                nonfinite_gradient_parameters=bad_gradients,
            )
            write_json(args.output_report, report)
            return

        optimizer.step()
        bad_parameters = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and not torch.isfinite(parameter).all()
        ]
        global_step += 1
        event = {
            "global_step": global_step,
            "mean_raw_loss": round(sum(group_losses) / len(group_losses), 8),
            "grad_norm": round(float(grad_norm.item()), 8),
            "learning_rate": scheduler.get_last_lr()[0],
        }
        report["updates"].append(event)
        if bad_parameters:
            report.update(
                status="failed",
                failure_stage="parameter_after_optimizer",
                global_step=global_step,
                accumulation_group_ids=group_ids,
                nonfinite_parameter_names=bad_parameters,
            )
            write_json(args.output_report, report)
            return
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        group_losses = []
        group_ids = []
        if global_step % 10 == 0 or global_step == 1:
            print(json.dumps(event), flush=True)
        if global_step >= args.max_updates:
            break

    report.update(status="completed", completed_updates=global_step)
    write_json(args.output_report, report)


if __name__ == "__main__":
    main()
