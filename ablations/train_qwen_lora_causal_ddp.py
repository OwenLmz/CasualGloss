#!/usr/bin/env python3
"""Continue the Fine-tune 1 LoRA on pseudo causal-action supervision."""

from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from causal_action_common import (
    ActionCollator,
    ActionDataset,
    file_sha256,
    metrics_from_totals,
    move_batch,
    read_json,
    read_jsonl,
    set_seed,
    write_json,
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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_nonfinite_report(
    output_dir: Path,
    *,
    stage: str,
    global_step: int,
    micro_index: int,
    batch_ids: list[str],
    group_ids: list[str],
    raw_loss: float | None = None,
    grad_norm: float | None = None,
    nonfinite_gradient_parameters: list[str] | None = None,
    nonfinite_parameter_names: list[str] | None = None,
) -> None:
    write_json(
        output_dir / "nonfinite_failure_report.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "global_step_before_update": global_step,
            "micro_index": micro_index,
            "batch_ids": batch_ids,
            "accumulation_group_ids": group_ids,
            "raw_loss": raw_loss,
            "grad_norm": grad_norm,
            "nonfinite_gradient_parameters": nonfinite_gradient_parameters or [],
            "nonfinite_parameter_names": nonfinite_parameter_names or [],
        },
    )


@torch.inference_mode()
def distributed_metrics(
    module: Any,
    dataset: ActionDataset,
    collator: ActionCollator,
    device: torch.device,
    rank: int,
    world_size: int,
    batch_size: int,
) -> dict[str, Any]:
    module.eval()
    indices = list(range(rank, len(dataset), world_size))
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    totals = torch.zeros(5, dtype=torch.float64, device=device)
    for batch in loader:
        tensor_batch = move_batch(batch, device)
        outputs = module(**tensor_batch)
        logits = outputs.logits[:, :-1, :]
        labels = tensor_batch["labels"][:, 1:]
        mask = labels.ne(-100)
        count = int(mask.sum().item())
        predictions = logits.argmax(dim=-1)
        totals[0] += float(outputs.loss.item()) * count
        totals[1] += count
        totals[2] += int((predictions.eq(labels) & mask).sum().item())
        for index in range(mask.shape[0]):
            row_mask = mask[index]
            totals[3] += int(predictions[index][row_mask].eq(labels[index][row_mask]).all().item())
            totals[4] += 1
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return metrics_from_totals(totals.tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--source-adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    import peft
    import transformers

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    training_start_time = perf_counter()

    config = read_json(args.config)
    seed = int(config["seed"])
    set_seed(seed + rank)
    train_rows = read_jsonl(args.train_manifest)
    validation_rows = read_jsonl(args.validation_manifest)
    test_rows = read_jsonl(args.test_manifest)
    for row in train_rows + validation_rows + test_rows:
        if row.get("target_is_gold") is not False or row.get("alignment_is_gold") is not False:
            raise SystemExit(f"Unexpected gold flag in {row.get('id')}")

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        history_path = args.output_dir / "training_history.jsonl"
        history_path.write_text("", encoding="utf-8")
    dist.barrier()
    history_path = args.output_dir / "training_history.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["model_revision"], trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model_dtype = configured_dtype(config)
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        dtype=model_dtype,
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
    module = PeftModel.from_pretrained(
        base_model, str(args.source_adapter_dir), is_trainable=True
    )
    if bool(config.get("trainable_parameters_fp32", False)):
        for parameter in module.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
    if rank == 0:
        module.print_trainable_parameters()

    system_prompt = config["system_prompt"]
    max_length = int(config["max_length"])
    train_dataset = ActionDataset(train_rows, tokenizer, system_prompt, max_length)
    validation_dataset = ActionDataset(validation_rows, tokenizer, system_prompt, max_length)
    collator = ActionCollator(int(tokenizer.pad_token_id))
    micro_batch = int(config["micro_batch_size_per_gpu"])
    evaluation_batch = int(config.get("evaluation_batch_size_per_gpu", micro_batch))
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
        batch_size=micro_batch,
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    ddp_model = DistributedDataParallel(
        module,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        broadcast_buffers=False,
    )
    trainable_parameters = [parameter for parameter in ddp_model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    grad_accum = int(config["gradient_accumulation_steps"])
    epochs = int(config["epochs"])
    updates_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_updates = updates_per_epoch * epochs
    warmup_steps = int(total_updates * float(config["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates
    )

    baseline_validation = distributed_metrics(
        module, validation_dataset, collator, device, rank, world_size, evaluation_batch
    )
    if rank == 0:
        append_jsonl(
            history_path,
            {
                "event": "validation",
                "stage": "finetune1_before_causal_training",
                "epoch": 0,
                "global_step": 0,
                "validation": baseline_validation,
            },
        )
        module.save_pretrained(args.output_dir / "adapter_best")
        write_json(
            args.output_dir / "adapter_best" / "selection.json",
            {
                "selected_by": "minimum_validation_pseudo_action_loss",
                "global_step": 0,
                "epoch_fraction": 0.0,
                "validation": baseline_validation,
            },
        )
        write_json(
            args.output_dir / "run_metadata.json",
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "scientific_status": "pseudo_causal_action_lora_not_gold_asl",
                "config": config,
                "source_adapter_dir": str(args.source_adapter_dir),
                "world_size": world_size,
                "gpu_names": [torch.cuda.get_device_name(index) for index in range(world_size)],
                "train_action_count": len(train_rows),
                "validation_action_count": len(validation_rows),
                "test_action_count_reserved": len(test_rows),
                "effective_batch_size": micro_batch * world_size * grad_accum,
                "evaluation_batch_size_per_gpu": evaluation_batch,
                "updates_per_epoch": updates_per_epoch,
                "total_updates": total_updates,
                "train_tokenization": train_dataset.statistics(),
                "validation_tokenization": validation_dataset.statistics(),
                "sha256": {
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
            },
        )
        print(f"[baseline] {baseline_validation}", flush=True)
    dist.barrier()

    best_loss = float(baseline_validation["pseudo_action_loss"])
    best_step = 0
    best_epoch_fraction = 0.0
    early_stopping_patience = int(
        config.get("early_stopping_patience_evaluations", 0)
    )
    early_stopping_min_delta = float(config.get("early_stopping_min_delta", 0.0))
    early_stopping_minimum_updates = int(
        config.get("early_stopping_minimum_updates", 0)
    )
    early_stopping_reference_loss = best_loss
    early_stopping_bad_evaluations = 0
    stopped_early = False
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    accumulation_group_ids: list[str] = []
    for epoch in range(1, epochs + 1):
        train_sampler.set_epoch(epoch)
        ddp_model.train()
        group_losses: list[float] = []
        for micro_index, batch in enumerate(train_loader):
            batch_ids = [str(value) for value in batch["ids"]]
            accumulation_group_ids.extend(batch_ids)
            group_start = (micro_index // grad_accum) * grad_accum
            group_size = min(grad_accum, len(train_loader) - group_start)
            is_update = (micro_index + 1) % grad_accum == 0 or micro_index + 1 == len(train_loader)
            context = nullcontext() if is_update else ddp_model.no_sync()
            with context:
                outputs = ddp_model(**move_batch(batch, device))
                raw_loss = outputs.loss
                if not torch.isfinite(raw_loss):
                    if rank == 0:
                        write_nonfinite_report(
                            args.output_dir,
                            stage="forward_loss",
                            global_step=global_step,
                            micro_index=micro_index,
                            batch_ids=batch_ids,
                            group_ids=accumulation_group_ids,
                            raw_loss=float(raw_loss.detach().item()),
                        )
                    raise FloatingPointError(
                        f"Non-finite forward loss before update {global_step + 1}; ids={batch_ids}"
                    )
                (raw_loss / group_size).backward()
            group_losses.append(float(raw_loss.detach().item()))
            if not is_update:
                continue
            grad_norm = clip_grad_norm_(trainable_parameters, float(config["max_grad_norm"]))
            if not torch.isfinite(grad_norm):
                bad_gradients = [
                    name
                    for name, parameter in module.named_parameters()
                    if parameter.requires_grad
                    and parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                ]
                if rank == 0:
                    write_nonfinite_report(
                        args.output_dir,
                        stage="gradient",
                        global_step=global_step,
                        micro_index=micro_index,
                        batch_ids=batch_ids,
                        group_ids=accumulation_group_ids,
                        raw_loss=float(raw_loss.detach().item()),
                        grad_norm=float(grad_norm.item()),
                        nonfinite_gradient_parameters=bad_gradients,
                    )
                raise FloatingPointError(
                    f"Non-finite gradient before update {global_step + 1}; ids={accumulation_group_ids}"
                )
            optimizer.step()
            bad_parameters = [
                name
                for name, parameter in module.named_parameters()
                if parameter.requires_grad and not torch.isfinite(parameter).all()
            ]
            if bad_parameters:
                if rank == 0:
                    write_nonfinite_report(
                        args.output_dir,
                        stage="parameter_after_optimizer",
                        global_step=global_step,
                        micro_index=micro_index,
                        batch_ids=batch_ids,
                        group_ids=accumulation_group_ids,
                        raw_loss=float(raw_loss.detach().item()),
                        grad_norm=float(grad_norm.item()),
                        nonfinite_parameter_names=bad_parameters,
                    )
                raise FloatingPointError(
                    f"Non-finite parameter after update {global_step + 1}; params={bad_parameters[:3]}"
                )
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accumulation_group_ids = []
            global_step += 1
            reduced_loss = torch.tensor(
                sum(group_losses) / len(group_losses), dtype=torch.float64, device=device
            )
            dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
            mean_train_loss = float((reduced_loss / world_size).item())
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
                        "epoch_fraction": round(epoch - 1 + (micro_index + 1) / len(train_loader), 8),
                        "global_step": global_step,
                        "pseudo_action_train_loss": round(mean_train_loss, 8),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "grad_norm_rank0": round(float(grad_norm.item()), 8),
                    },
                )
                print(f"[train] step={global_step}/{total_updates} loss={mean_train_loss:.5f}", flush=True)

            if global_step % int(config["evaluation_steps"]) != 0 and global_step != total_updates:
                continue
            validation = distributed_metrics(
                module, validation_dataset, collator, device, rank, world_size, evaluation_batch
            )
            epoch_fraction = epoch - 1 + (micro_index + 1) / len(train_loader)
            if rank == 0:
                append_jsonl(
                    history_path,
                    {
                        "event": "validation",
                        "stage": "during_causal_training",
                        "epoch": epoch,
                        "epoch_fraction": round(epoch_fraction, 8),
                        "global_step": global_step,
                        "validation": validation,
                    },
                )
                print(
                    f"[validation] step={global_step} loss={validation['pseudo_action_loss']:.5f} "
                    f"seq_acc={validation['pseudo_action_sequence_exact_match_teacher_forced']:.4f}",
                    flush=True,
                )
                if float(validation["pseudo_action_loss"]) < best_loss:
                    best_loss = float(validation["pseudo_action_loss"])
                    best_step = global_step
                    best_epoch_fraction = epoch_fraction
                    module.save_pretrained(args.output_dir / "adapter_best")
                    write_json(
                        args.output_dir / "adapter_best" / "selection.json",
                        {
                            "selected_by": "minimum_validation_pseudo_action_loss",
                            "global_step": best_step,
                            "epoch_fraction": round(best_epoch_fraction, 8),
                            "validation": validation,
                        },
                    )
                validation_loss = float(validation["pseudo_action_loss"])
                if (
                    early_stopping_reference_loss - validation_loss
                    > early_stopping_min_delta
                ):
                    early_stopping_reference_loss = validation_loss
                    early_stopping_bad_evaluations = 0
                elif global_step >= early_stopping_minimum_updates:
                    early_stopping_bad_evaluations += 1
                stopped_early = (
                    early_stopping_patience > 0
                    and global_step >= early_stopping_minimum_updates
                    and early_stopping_bad_evaluations
                    >= early_stopping_patience
                )
                if stopped_early:
                    append_jsonl(
                        history_path,
                        {
                            "event": "early_stopping",
                            "global_step": global_step,
                            "patience_evaluations": early_stopping_patience,
                            "minimum_delta": early_stopping_min_delta,
                            "bad_evaluations": early_stopping_bad_evaluations,
                            "best_validation_loss": best_loss,
                        },
                    )
                    print(
                        f"[early-stopping] step={global_step} "
                        f"bad_evaluations={early_stopping_bad_evaluations}",
                        flush=True,
                    )
            stop_tensor = torch.tensor(
                int(stopped_early), dtype=torch.int64, device=device
            )
            dist.broadcast(stop_tensor, src=0)
            stopped_early = bool(stop_tensor.item())
            dist.barrier()
            if stopped_early:
                break
            ddp_model.train()

        if stopped_early:
            break

    if rank == 0:
        module.save_pretrained(args.output_dir / "adapter_final")
        tokenizer.save_pretrained(args.output_dir / "adapter_final")
        write_json(
            args.output_dir / "training_summary.json",
            {
                "experiment_name": config["experiment_name"],
                "scientific_status": "pseudo_causal_action_lora_not_gold_asl",
                "train_action_count": len(train_rows),
                "validation_action_count": len(validation_rows),
                "test_action_count_reserved": len(test_rows),
                "epochs": epochs,
                "global_update_steps": global_step,
                "stopped_early": stopped_early,
                "early_stopping": {
                    "patience_evaluations": early_stopping_patience,
                    "minimum_delta": early_stopping_min_delta,
                    "minimum_updates": early_stopping_minimum_updates,
                },
                "elapsed_seconds": round(perf_counter() - training_start_time, 4),
                "baseline_validation": baseline_validation,
                "best_checkpoint": {
                    "global_step": best_step,
                    "epoch_fraction": round(best_epoch_fraction, 8),
                    "validation_pseudo_action_loss": round(best_loss, 8),
                    "path": str(args.output_dir / "adapter_best"),
                },
            },
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
