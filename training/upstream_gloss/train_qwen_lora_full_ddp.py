#!/usr/bin/env python3
"""Four-GPU DDP LoRA training on the full How2Sign pseudo-gloss dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from finetune_common import (
    CausalCollator,
    GlossDataset,
    file_sha256,
    metrics_from_totals,
    move_batch,
    read_json,
    read_jsonl,
    set_seed,
    write_json,
)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@torch.inference_mode()
def distributed_teacher_forced_metrics(
    module: Any,
    dataset: GlossDataset,
    collator: CausalCollator,
    device: torch.device,
    rank: int,
    world_size: int,
    batch_size: int,
) -> dict[str, float | int]:
    module.eval()
    indices = list(range(rank, len(dataset), world_size))
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=True,
    )
    totals = torch.zeros(5, dtype=torch.float64, device=device)
    for batch in loader:
        tensor_batch = move_batch(batch, device)
        outputs = module(**tensor_batch)
        shifted_logits = outputs.logits[:, :-1, :]
        shifted_labels = tensor_batch["labels"][:, 1:]
        mask = shifted_labels.ne(-100)
        target_count = int(mask.sum().item())
        predictions = shifted_logits.argmax(dim=-1)
        totals[0] += float(outputs.loss.item()) * target_count
        totals[1] += target_count
        totals[2] += int((predictions.eq(shifted_labels) & mask).sum().item())
        for row_index in range(mask.shape[0]):
            row_mask = mask[row_index]
            if int(row_mask.sum().item()) == 0:
                continue
            totals[3] += int(
                predictions[row_index][row_mask]
                .eq(shifted_labels[row_index][row_mask])
                .all()
                .item()
            )
            totals[4] += 1
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return metrics_from_totals(totals.tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--evaluation-steps", type=int)
    args = parser.parse_args()

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
    import peft
    import transformers

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    config = read_json(args.config)
    seed = int(config["seed"])
    set_seed(seed + rank)
    epochs = int(args.epochs if args.epochs is not None else config["epochs"])
    evaluation_steps = int(
        args.evaluation_steps
        if args.evaluation_steps is not None
        else config["evaluation_steps"]
    )
    train_rows = read_jsonl(args.train_manifest)
    validation_rows = read_jsonl(args.validation_manifest)
    test_rows = read_jsonl(args.test_manifest)
    if args.max_train_samples:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_validation_samples:
        validation_rows = validation_rows[: args.max_validation_samples]
    for row in train_rows + validation_rows + test_rows:
        if row.get("pseudo_gloss_is_gold") is not False:
            raise SystemExit(f"Non-pseudo target flag for {row.get('id')}")

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        history_path = args.output_dir / "training_history.jsonl"
        history_path.write_text("", encoding="utf-8")
    dist.barrier()
    history_path = args.output_dir / "training_history.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        dtype=torch.bfloat16,
        attn_implementation=config["attention_implementation"],
        device_map={"": local_rank},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    base_model.config.use_cache = False
    if config.get("gradient_checkpointing", True):
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base_model.enable_input_require_grads()
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(config["lora"]["rank"]),
        lora_alpha=int(config["lora"]["alpha"]),
        lora_dropout=float(config["lora"]["dropout"]),
        target_modules=list(config["lora"]["target_modules"]),
        bias="none",
    )
    module = get_peft_model(base_model, lora)
    if rank == 0:
        module.print_trainable_parameters()

    system_prompt = str(config["system_prompt"])
    max_length = int(config["max_length"])
    train_dataset = GlossDataset(train_rows, tokenizer, system_prompt, max_length)
    validation_dataset = GlossDataset(
        validation_rows, tokenizer, system_prompt, max_length
    )
    collator = CausalCollator(int(tokenizer.pad_token_id))
    micro_batch_size = int(config["micro_batch_size_per_gpu"])
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        sampler=train_sampler,
        collate_fn=collator,
        pin_memory=True,
        num_workers=0,
    )
    ddp_model = DistributedDataParallel(
        module,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        broadcast_buffers=False,
    )
    trainable_parameters = [
        parameter for parameter in ddp_model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    grad_accum = int(config["gradient_accumulation_steps"])
    updates_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_updates = max(1, updates_per_epoch * epochs)
    warmup_steps = int(total_updates * float(config["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    sampler_padding = len(train_sampler) * world_size - len(train_dataset)
    if rank == 0:
        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_name": config["experiment_name"],
            "scientific_status": "full_pseudo_label_lora_not_gold_asl",
            "config": config,
            "effective_epochs": epochs,
            "world_size": world_size,
            "cuda_devices": [
                torch.cuda.get_device_name(index) for index in range(world_size)
            ],
            "train_count": len(train_rows),
            "validation_count": len(validation_rows),
            "test_count_reserved_unseen_during_training": len(test_rows),
            "effective_batch_size": micro_batch_size * world_size * grad_accum,
            "updates_per_epoch": updates_per_epoch,
            "total_updates": total_updates,
            "distributed_sampler_padding_example_count": sampler_padding,
            "train_dataset_tokenization": train_dataset.statistics(),
            "validation_dataset_tokenization": validation_dataset.statistics(),
            "source_sha256": {
                "config": file_sha256(args.config),
                "train": file_sha256(args.train_manifest),
                "validation": file_sha256(args.validation_manifest),
                "test": file_sha256(args.test_manifest),
            },
            "versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "peft": peft.__version__,
            },
        }
        write_json(args.output_dir / "run_metadata.json", metadata)

    baseline_validation = distributed_teacher_forced_metrics(
        module,
        validation_dataset,
        collator,
        device,
        rank,
        world_size,
        micro_batch_size,
    )
    if rank == 0:
        append_jsonl(
            history_path,
            {
                "event": "validation",
                "stage": "base_before_training",
                "epoch": 0,
                "global_step": 0,
                "validation": baseline_validation,
            },
        )
        print(f"[baseline] {baseline_validation}", flush=True)

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    best_validation_loss = float("inf")
    best_step = 0
    best_epoch_fraction = 0.0
    last_evaluation_step = 0
    for epoch in range(1, epochs + 1):
        train_sampler.set_epoch(epoch)
        ddp_model.train()
        group_losses: list[float] = []
        for micro_index, batch in enumerate(train_loader):
            group_start = (micro_index // grad_accum) * grad_accum
            group_size = min(grad_accum, len(train_loader) - group_start)
            is_update = (micro_index + 1) % grad_accum == 0 or (
                micro_index + 1 == len(train_loader)
            )
            sync_context = nullcontext() if is_update else ddp_model.no_sync()
            with sync_context:
                outputs = ddp_model(**move_batch(batch, device))
                raw_loss = outputs.loss
                (raw_loss / group_size).backward()
            group_losses.append(float(raw_loss.detach().item()))
            if not is_update:
                continue
            grad_norm = clip_grad_norm_(
                trainable_parameters, float(config["max_grad_norm"])
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            local_loss = torch.tensor(
                sum(group_losses) / len(group_losses),
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
            mean_train_loss = float((local_loss / world_size).item())
            group_losses = []
            if rank == 0 and (
                global_step == 1
                or global_step % int(config["logging_steps"]) == 0
                or global_step == total_updates
            ):
                append_jsonl(
                    history_path,
                    {
                        "event": "train_update",
                        "epoch": epoch,
                        "epoch_fraction": round(
                            epoch - 1 + (micro_index + 1) / len(train_loader), 8
                        ),
                        "global_step": global_step,
                        "pseudo_label_train_loss": round(mean_train_loss, 8),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "grad_norm_rank0": round(float(grad_norm.item()), 8),
                    },
                )
                print(
                    f"[train] epoch={epoch} step={global_step}/{total_updates} "
                    f"loss={mean_train_loss:.5f}",
                    flush=True,
                )

            should_evaluate = global_step % evaluation_steps == 0 or (
                global_step == total_updates
            )
            if not should_evaluate:
                continue
            validation_metrics = distributed_teacher_forced_metrics(
                module,
                validation_dataset,
                collator,
                device,
                rank,
                world_size,
                micro_batch_size,
            )
            last_evaluation_step = global_step
            epoch_fraction = epoch - 1 + (micro_index + 1) / len(train_loader)
            if rank == 0:
                append_jsonl(
                    history_path,
                    {
                        "event": "validation",
                        "stage": "during_training",
                        "epoch": epoch,
                        "epoch_fraction": round(epoch_fraction, 8),
                        "global_step": global_step,
                        "validation": validation_metrics,
                    },
                )
                print(
                    f"[validation] step={global_step} "
                    f"loss={validation_metrics['pseudo_label_loss']:.5f} "
                    f"token_acc={validation_metrics['pseudo_label_token_accuracy_teacher_forced']:.4f}",
                    flush=True,
                )
                current_loss = float(validation_metrics["pseudo_label_loss"])
                if current_loss < best_validation_loss:
                    best_validation_loss = current_loss
                    best_step = global_step
                    best_epoch_fraction = epoch_fraction
                    module.save_pretrained(args.output_dir / "adapter_best")
                    write_json(
                        args.output_dir / "adapter_best" / "selection.json",
                        {
                            "selected_by": "minimum_validation_pseudo_label_loss",
                            "global_step": best_step,
                            "epoch_fraction": round(best_epoch_fraction, 8),
                            "validation": validation_metrics,
                        },
                    )
            dist.barrier()
            ddp_model.train()

    if last_evaluation_step != global_step:
        raise RuntimeError("Final training step was not evaluated")
    if rank == 0:
        module.save_pretrained(args.output_dir / "adapter_final")
        tokenizer.save_pretrained(args.output_dir / "adapter_final")
        summary = {
            "experiment_name": config["experiment_name"],
            "scientific_status": "full_pseudo_label_lora_not_gold_asl",
            "train_count": len(train_rows),
            "validation_count": len(validation_rows),
            "test_count_reserved": len(test_rows),
            "epochs": epochs,
            "global_update_steps": global_step,
            "baseline_validation": baseline_validation,
            "best_checkpoint": {
                "global_step": best_step,
                "epoch_fraction": round(best_epoch_fraction, 8),
                "validation_pseudo_label_loss": round(best_validation_loss, 8),
                "path": str(args.output_dir / "adapter_best"),
            },
            "important_limitations": [
                "Targets are deterministic pseudo ASL glosses, not gold annotations.",
                "Validation metrics measure pseudo-label agreement, not ASL correctness.",
                "The test split was reserved and was not evaluated during training.",
            ],
        }
        write_json(args.output_dir / "training_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
