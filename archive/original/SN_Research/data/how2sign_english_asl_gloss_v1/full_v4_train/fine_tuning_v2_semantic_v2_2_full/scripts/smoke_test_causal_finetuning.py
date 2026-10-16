#!/usr/bin/env python3
"""Load Fine-tune 1 and run a small causal-action forward pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from causal_action_common import (
    ActionCollator,
    ActionDataset,
    move_batch,
    read_json,
)


def read_first_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) >= limit:
                    break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-adapter-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = read_json(args.config)
    rows = read_first_rows(args.manifest, 4)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["model_revision"]
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        dtype=torch.bfloat16,
        attn_implementation=config["attention_implementation"],
        device_map={"": args.device},
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        model, str(args.source_adapter_dir), is_trainable=True
    )
    dataset = ActionDataset(
        rows, tokenizer, config["system_prompt"], int(config["max_length"])
    )
    batch = ActionCollator(int(tokenizer.pad_token_id))(
        [dataset[index] for index in range(len(dataset))]
    )
    outputs = model(**move_batch(batch, args.device))
    outputs.loss.backward()
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    result = {
        "row_count": len(rows),
        "loss": round(float(outputs.loss.detach().item()), 8),
        "trainable_parameter_count": trainable_count,
        "dataset_statistics": dataset.statistics(),
        "ok": bool(torch.isfinite(outputs.loss).item()) and trainable_count > 0,
    }
    print(result)
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
