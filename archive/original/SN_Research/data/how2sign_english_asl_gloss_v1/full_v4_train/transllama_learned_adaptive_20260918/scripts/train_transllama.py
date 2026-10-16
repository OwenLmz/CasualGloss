#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from transllama_data import file_sha256, read_jsonl, write_json
from transllama_training import (
    AllPrefixDataset,
    PrefixCollator,
    SampledPrefixDataset,
    set_seed,
    tensor_batch,
    verify_chat_boundary,
    wait_token_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-alignments", type=Path, required=True)
    parser.add_argument("--validation-alignments", type=Path, required=True)
    parser.add_argument("--source-adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-validation-parents", type=int)
    parser.add_argument("--max-updates", type=int)
    return parser.parse_args()


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


@torch.inference_mode()
def validate(
    model: Any,
    loader: DataLoader,
    device: torch.device,
    wait_ids: set[int],
    eos_id: int,
) -> dict[str, Any]:
    model.eval()
    totals = {
        "loss_numerator": 0.0,
        "target_tokens": 0,
        "correct_tokens": 0,
        "sequences": 0,
        "exact_sequences": 0,
        "wait_tokens": 0,
        "correct_wait_tokens": 0,
        "eos_tokens": 0,
        "correct_eos_tokens": 0,
        "content_tokens": 0,
        "correct_content_tokens": 0,
        "truncated_sequences": 0,
    }
    for batch in loader:
        tensors = tensor_batch(batch, device)
        outputs = model(**tensors)
        logits = outputs.logits[:, :-1, :]
        labels = tensors["labels"][:, 1:]
        mask = labels.ne(-100)
        predictions = logits.argmax(dim=-1)
        correct = predictions.eq(labels) & mask
        count = int(mask.sum().item())
        totals["loss_numerator"] += float(outputs.loss.item()) * count
        totals["target_tokens"] += count
        totals["correct_tokens"] += int(correct.sum().item())
        totals["truncated_sequences"] += sum(batch["truncated"])

        wait_mask = torch.zeros_like(mask)
        for wait_id in wait_ids:
            wait_mask |= labels.eq(wait_id)
        wait_mask &= mask
        eos_mask = labels.eq(eos_id) & mask
        content_mask = mask & ~wait_mask & ~eos_mask
        totals["wait_tokens"] += int(wait_mask.sum().item())
        totals["correct_wait_tokens"] += int((correct & wait_mask).sum().item())
        totals["eos_tokens"] += int(eos_mask.sum().item())
        totals["correct_eos_tokens"] += int((correct & eos_mask).sum().item())
        totals["content_tokens"] += int(content_mask.sum().item())
        totals["correct_content_tokens"] += int((correct & content_mask).sum().item())
        for row_index in range(mask.shape[0]):
            row_mask = mask[row_index]
            totals["exact_sequences"] += int(
                predictions[row_index][row_mask].eq(labels[row_index][row_mask]).all().item()
            )
            totals["sequences"] += 1

    def accuracy(correct_key: str, count_key: str) -> float:
        return round(totals[correct_key] / max(1, totals[count_key]), 8)

    return {
        "target_only_loss": round(
            totals["loss_numerator"] / max(1, totals["target_tokens"]), 8
        ),
        "target_token_accuracy_teacher_forced": accuracy("correct_tokens", "target_tokens"),
        "sequence_exact_match_teacher_forced": accuracy("exact_sequences", "sequences"),
        "wait_token_accuracy_teacher_forced": accuracy(
            "correct_wait_tokens", "wait_tokens"
        ),
        "content_token_accuracy_teacher_forced": accuracy(
            "correct_content_tokens", "content_tokens"
        ),
        "eos_token_accuracy_teacher_forced": accuracy("correct_eos_tokens", "eos_tokens"),
        **{key: value for key, value in totals.items() if key != "loss_numerator"},
    }


def main() -> None:
    args = parse_args()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    import peft
    import transformers

    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    set_seed(seed)
    device = torch.device("cuda", 0)
    start_time = perf_counter()
    train_rows = read_jsonl(args.train_alignments)
    validation_rows = read_jsonl(args.validation_alignments)
    if args.max_train_rows is not None:
        train_rows = train_rows[: args.max_train_rows]
    if args.max_validation_parents is not None:
        validation_rows = validation_rows[: args.max_validation_parents]
    if any(row["split"] != "train" for row in train_rows):
        raise SystemExit("non-train row in training alignments")
    if any(row["split"] != "validation" for row in validation_rows):
        raise SystemExit("non-validation row in validation alignments")
    if any(row.get("alignment_method") != "official_transllama_simalign_itermax_adapted_to_gloss" for row in train_rows + validation_rows):
        raise SystemExit("unexpected alignment provenance")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    history_path = args.output_dir / "training_history.jsonl"
    history_path.write_text("", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    verify_chat_boundary(tokenizer, config["system_prompt"])
    wait_ids = wait_token_ids(tokenizer)

    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=config["attention_implementation"],
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    base_model.config.use_cache = False
    if config.get("gradient_checkpointing", True):
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base_model.enable_input_require_grads()
    model = PeftModel.from_pretrained(
        base_model, str(args.source_adapter_dir), is_trainable=True
    )
    model.print_trainable_parameters()

    max_length = int(config["max_length"])
    train_dataset = SampledPrefixDataset(
        train_rows, tokenizer, config["system_prompt"], max_length, seed
    )
    validation_dataset = AllPrefixDataset(
        validation_rows, tokenizer, config["system_prompt"], max_length
    )
    collator = PrefixCollator(int(tokenizer.pad_token_id))
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["validation_batch_size"]),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    micro_batch = int(config["micro_batch_size"])
    grad_accum = int(config["gradient_accumulation_steps"])
    epochs = int(config["epochs"])
    micro_steps_per_epoch = math.ceil(len(train_dataset) / micro_batch)
    updates_per_epoch = math.ceil(micro_steps_per_epoch / grad_accum)
    planned_updates = updates_per_epoch * epochs
    total_updates = min(planned_updates, args.max_updates) if args.max_updates else planned_updates
    warmup_steps = int(total_updates * float(config["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates
    )

    baseline_validation = validate(
        model, validation_loader, device, wait_ids, int(tokenizer.eos_token_id)
    )
    append_jsonl(
        history_path,
        {"event": "validation", "epoch": 0, "global_step": 0, "validation": baseline_validation},
    )
    model.save_pretrained(args.output_dir / "adapter_best")
    write_json(
        args.output_dir / "adapter_best" / "selection.json",
        {
            "selected_by": "minimum_all_prefix_validation_target_only_loss",
            "epoch": 0,
            "global_step": 0,
            "validation": baseline_validation,
        },
    )
    print(f"[validation baseline] {baseline_validation}", flush=True)

    write_json(
        args.output_dir / "run_metadata.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_status": "independent_transllama_style_sequence_sft_not_gold_asl",
            "config": config,
            "source_adapter_dir": str(args.source_adapter_dir),
            "train_parent_count": len(train_rows),
            "validation_parent_count": len(validation_rows),
            "validation_all_prefix_count": len(validation_dataset),
            "effective_batch_size": micro_batch * grad_accum,
            "wait_token_ids": sorted(wait_ids),
            "test_manifest_accessed": False,
            "sha256": {
                "config": file_sha256(args.config),
                "train_alignments": file_sha256(args.train_alignments),
                "validation_alignments": file_sha256(args.validation_alignments),
            },
            "versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "peft": peft.__version__,
            },
        },
    )

    best_loss = float(baseline_validation["target_only_loss"])
    best_epoch = 0
    best_step = 0
    global_step = 0
    stop = False
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        train_dataset.set_epoch(epoch)
        generator = torch.Generator()
        generator.manual_seed(seed + epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=micro_batch,
            shuffle=True,
            generator=generator,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True,
        )
        model.train()
        group_losses: list[float] = []
        for micro_index, batch in enumerate(train_loader):
            group_start = (micro_index // grad_accum) * grad_accum
            group_size = min(grad_accum, len(train_loader) - group_start)
            outputs = model(**tensor_batch(batch, device))
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}, micro {micro_index}")
            (loss / group_size).backward()
            group_losses.append(float(loss.detach().item()))
            is_update = (micro_index + 1) % grad_accum == 0 or micro_index + 1 == len(train_loader)
            if not is_update:
                continue
            grad_norm = clip_grad_norm_(trainable_parameters, float(config["max_grad_norm"]))
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient at update {global_step + 1}")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            mean_loss = sum(group_losses) / len(group_losses)
            group_losses = []
            if global_step == 1 or global_step % int(config["logging_steps"]) == 0:
                event = {
                    "event": "train_update",
                    "epoch": epoch,
                    "global_step": global_step,
                    "target_only_train_loss": round(mean_loss, 8),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "grad_norm": round(float(grad_norm.item()), 8),
                }
                append_jsonl(history_path, event)
                print(f"[train] {event}", flush=True)
            if global_step >= total_updates:
                stop = True
                break

        validation = validate(
            model, validation_loader, device, wait_ids, int(tokenizer.eos_token_id)
        )
        append_jsonl(
            history_path,
            {
                "event": "validation",
                "epoch": epoch,
                "global_step": global_step,
                "validation": validation,
            },
        )
        print(f"[validation epoch={epoch}] {validation}", flush=True)
        if float(validation["target_only_loss"]) < best_loss:
            best_loss = float(validation["target_only_loss"])
            best_epoch = epoch
            best_step = global_step
            model.save_pretrained(args.output_dir / "adapter_best")
            write_json(
                args.output_dir / "adapter_best" / "selection.json",
                {
                    "selected_by": "minimum_all_prefix_validation_target_only_loss",
                    "epoch": best_epoch,
                    "global_step": best_step,
                    "validation": validation,
                },
            )
        if stop:
            break
        model.train()

    model.save_pretrained(args.output_dir / "adapter_final")
    tokenizer.save_pretrained(args.output_dir / "adapter_final")
    write_json(
        args.output_dir / "training_summary.json",
        {
            "scientific_status": "independent_transllama_style_sequence_sft_not_gold_asl",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(perf_counter() - start_time, 3),
            "global_update_steps": global_step,
            "baseline_validation": baseline_validation,
            "best_checkpoint": {
                "epoch": best_epoch,
                "global_step": best_step,
                "validation_target_only_loss": best_loss,
                "path": str(args.output_dir / "adapter_best"),
            },
        },
    )


if __name__ == "__main__":
    main()
