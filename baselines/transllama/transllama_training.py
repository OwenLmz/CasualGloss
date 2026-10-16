from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from transllama_data import WAIT_SYMBOL, make_prefix_example


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def deterministic_prefix_length(parent_id: str, source_length: int, seed: int, epoch: int) -> int:
    digest = hashlib.sha256(f"{seed}:{epoch}:{parent_id}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    weights = np.logspace(0.0, 1.0, source_length, dtype=np.float64)
    weights /= weights.sum()
    return int(rng.choice(np.arange(1, source_length + 1), p=weights))


def build_user_content(source_prefix_text: str) -> str:
    return source_prefix_text


def encode_prefix_example(
    prefix: dict[str, Any],
    tokenizer: Any,
    system_prompt: str,
    max_length: int,
) -> dict[str, Any]:
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_content(prefix["source_prefix_text"])},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    target_ids = tokenizer.encode(prefix["assistant_target_text"], add_special_tokens=False)
    if prefix["source_complete"]:
        target_ids = target_ids + [int(tokenizer.eos_token_id)]
    if not target_ids:
        raise ValueError(f"empty target tokenization for {prefix['id']}")

    truncated = False
    if len(prompt_ids) + len(target_ids) > max_length:
        truncated = True
        prompt_budget = max_length - len(target_ids)
        if prompt_budget <= 0:
            raise ValueError(f"target exceeds maximum length for {prefix['id']}")
        prompt_ids = prompt_ids[-prompt_budget:]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return {
        "id": prefix["id"],
        "parent_id": prefix["parent_id"],
        "prefix_length": prefix["source_prefix_length"],
        "source_complete": prefix["source_complete"],
        "input_ids": input_ids,
        "labels": labels,
        "truncated": truncated,
    }


class SampledPrefixDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        system_prompt: str,
        max_length: int,
        seed: int,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        prefix_length = deterministic_prefix_length(
            str(row["id"]), len(row["source_words"]), self.seed, self.epoch
        )
        prefix = make_prefix_example(row, prefix_length)
        return encode_prefix_example(prefix, self.tokenizer, self.system_prompt, self.max_length)


class AllPrefixDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        system_prompt: str,
        max_length: int,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.indices = [
            (row_index, prefix_length)
            for row_index, row in enumerate(rows)
            for prefix_length in range(1, len(row["source_words"]) + 1)
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index, prefix_length = self.indices[index]
        prefix = make_prefix_example(self.rows[row_index], prefix_length)
        return encode_prefix_example(prefix, self.tokenizer, self.system_prompt, self.max_length)


@dataclass
class PrefixCollator:
    pad_token_id: int

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        maximum = max(len(example["input_ids"]) for example in examples)
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for example in examples:
            padding = maximum - len(example["input_ids"])
            input_ids.append(example["input_ids"] + [self.pad_token_id] * padding)
            labels.append(example["labels"] + [-100] * padding)
            attention_mask.append([1] * len(example["input_ids"]) + [0] * padding)
        return {
            "ids": [example["id"] for example in examples],
            "parent_ids": [example["parent_id"] for example in examples],
            "prefix_lengths": [example["prefix_length"] for example in examples],
            "truncated": [bool(example["truncated"]) for example in examples],
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in ("input_ids", "attention_mask", "labels")
    }


def wait_token_ids(tokenizer: Any) -> set[int]:
    ids = {
        tuple(tokenizer.encode(WAIT_SYMBOL, add_special_tokens=False)),
        tuple(tokenizer.encode(" " + WAIT_SYMBOL, add_special_tokens=False)),
    }
    if any(len(value) != 1 for value in ids):
        raise ValueError(f"WAIT is not single-token in every context: {ids}")
    return {value[0] for value in ids}


def verify_chat_boundary(tokenizer: Any, system_prompt: str) -> None:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "So we are going down."},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    for target in (WAIT_SYMBOL, "GO DOWN " + WAIT_SYMBOL, "GO DOWN"):
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        combined_ids = tokenizer.encode(prompt_text + target, add_special_tokens=False)
        if combined_ids != prompt_ids + target_ids:
            raise ValueError(f"chat/target token boundary is unstable for {target!r}")
