#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transllama_data import read_jsonl
from transllama_training import AllPrefixDataset, verify_chat_boundary, wait_token_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignments", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit-parents", type=int)
    return parser.parse_args()


def main() -> None:
    from transformers import AutoTokenizer

    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = read_jsonl(args.alignments)
    if args.limit_parents is not None:
        rows = rows[: args.limit_parents]
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    verify_chat_boundary(tokenizer, config["system_prompt"])
    waits = wait_token_ids(tokenizer)
    dataset = AllPrefixDataset(
        rows, tokenizer, config["system_prompt"], int(config["max_length"])
    )
    truncated = 0
    max_input = 0
    max_target = 0
    incomplete_with_eos = 0
    complete_without_eos = 0
    samples = []
    for index in range(len(dataset)):
        example = dataset[index]
        target_ids = [value for value in example["labels"] if value != -100]
        truncated += int(example["truncated"])
        max_input = max(max_input, len(example["input_ids"]))
        max_target = max(max_target, len(target_ids))
        has_eos = int(tokenizer.eos_token_id) in target_ids
        incomplete_with_eos += int(not example["source_complete"] and has_eos)
        complete_without_eos += int(example["source_complete"] and not has_eos)
        if len(samples) < 10:
            samples.append(
                {
                    "id": example["id"],
                    "source_complete": example["source_complete"],
                    "target_ids": target_ids,
                    "decoded_target": tokenizer.decode(target_ids),
                }
            )
    report = {
        "parent_count": len(rows),
        "prefix_example_count": len(dataset),
        "wait_token_ids": sorted(waits),
        "truncated": truncated,
        "max_input_length": max_input,
        "max_target_length": max_target,
        "incomplete_with_eos": incomplete_with_eos,
        "complete_without_eos": complete_without_eos,
        "samples": samples,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if truncated or incomplete_with_eos or complete_without_eos:
        raise SystemExit("training example audit failed")


if __name__ == "__main__":
    main()
