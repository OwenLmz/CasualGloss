#!/usr/bin/env python3
"""Shared data and metric utilities for causal-action fine-tuning."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


WRITE_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9'_-]*$")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ActionDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, system_prompt: str, max_length: int) -> None:
        self.rows = rows
        self.examples: list[dict[str, Any]] = []
        self.truncated_example_count = 0
        self.max_original_length = 0
        self.max_target_length = 0
        for row in rows:
            prompt_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row["state_prompt"]},
            ]
            full_messages = prompt_messages + [
                {"role": "assistant", "content": row["target_action"]}
            ]
            prompt_ids = tokenizer.apply_chat_template(
                prompt_messages, tokenize=True, add_generation_prompt=True
            )
            full_ids = tokenizer.apply_chat_template(
                full_messages, tokenize=True, add_generation_prompt=False
            )
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError(f"Chat prefix mismatch for {row['id']}")
            target_ids = full_ids[len(prompt_ids) :]
            if not target_ids:
                raise ValueError(f"Empty target for {row['id']}")
            self.max_original_length = max(self.max_original_length, len(full_ids))
            self.max_target_length = max(self.max_target_length, len(target_ids))
            if len(full_ids) > max_length:
                self.truncated_example_count += 1
                prompt_budget = max_length - len(target_ids)
                prompt_ids = prompt_ids[-prompt_budget:]
                full_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
            self.examples.append({"id": row["id"], "input_ids": full_ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]

    def statistics(self) -> dict[str, int]:
        return {
            "example_count": len(self.examples),
            "truncated_example_count": self.truncated_example_count,
            "max_original_token_length": self.max_original_length,
            "max_target_token_length": self.max_target_length,
        }


class ActionCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        maximum = max(len(row["input_ids"]) for row in examples)
        input_ids, attention_mask, labels, ids = [], [], [], []
        for row in examples:
            padding = maximum - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append([1] * len(row["input_ids"]) + [0] * padding)
            labels.append(row["labels"] + [-100] * padding)
            ids.append(row["id"])
        return {
            "ids": ids,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def move_batch(batch: dict[str, Any], device: torch.device | str) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask", "labels"}
    }


def metrics_from_totals(totals: list[float | int]) -> dict[str, float | int]:
    loss_numerator, target_tokens, correct_tokens, exact_sequences, sequences = totals
    return {
        "pseudo_action_loss": round(float(loss_numerator) / max(1, int(target_tokens)), 8),
        "pseudo_action_token_accuracy_teacher_forced": round(
            int(correct_tokens) / max(1, int(target_tokens)), 8
        ),
        "pseudo_action_sequence_exact_match_teacher_forced": round(
            int(exact_sequences) / max(1, int(sequences)), 8
        ),
        "target_token_count": int(target_tokens),
        "sequence_count": int(sequences),
    }


@torch.inference_mode()
def teacher_forced_metrics(model: Any, dataset: Dataset, collator: ActionCollator, device: Any, batch_size: int) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    totals = [0.0, 0, 0, 0, 0]
    for batch in loader:
        tensor_batch = move_batch(batch, device)
        outputs = model(**tensor_batch)
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
    return metrics_from_totals(totals)


def normalize_action(text: str) -> str:
    return " ".join(text.strip().split()).upper()


def action_class(action: str, wait_token: str, end_token: str) -> str:
    if action == wait_token:
        return "WAIT"
    if action == end_token:
        return "END"
    if WRITE_PATTERN.fullmatch(action):
        return "WRITE"
    return "INVALID"


def generation_metrics(predictions: list[dict[str, Any]], wait_token: str, end_token: str) -> dict[str, Any]:
    total = len(predictions)
    exact = sum(row["exact_action_match"] for row in predictions)
    confusion: Counter[tuple[str, str]] = Counter(
        (row["target_action_class"], row["predicted_action_class"]) for row in predictions
    )
    classes = ("WAIT", "WRITE", "END")
    per_class = {}
    for name in classes:
        tp = confusion[(name, name)]
        fp = sum(confusion[(target, name)] for target in classes if target != name)
        fn = sum(confusion[(name, predicted)] for predicted in (*classes, "INVALID") if predicted != name)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        per_class[name] = {
            "precision": round(precision, 8),
            "recall": round(recall, 8),
            "f1": round(2 * precision * recall / max(1e-12, precision + recall), 8),
            "support": sum(confusion[(name, predicted)] for predicted in (*classes, "INVALID")),
        }
    write_rows = [row for row in predictions if row["target_action_class"] == "WRITE"]
    wait_rows = [row for row in predictions if row["target_action_class"] == "WAIT"]
    parent_scores: dict[str, list[bool]] = defaultdict(list)
    for row in predictions:
        parent_scores[row["parent_id"]].append(row["exact_action_match"])
    return {
        "record_count": total,
        "pseudo_action_exact_accuracy": round(exact / max(1, total), 8),
        "action_type_accuracy": round(
            sum(row["target_action_class"] == row["predicted_action_class"] for row in predictions)
            / max(1, total),
            8,
        ),
        "write_token_exact_accuracy": round(
            sum(row["exact_action_match"] for row in write_rows) / max(1, len(write_rows)), 8
        ),
        "premature_write_on_wait_rate": round(
            sum(row["predicted_action_class"] == "WRITE" for row in wait_rows)
            / max(1, len(wait_rows)),
            8,
        ),
        "over_wait_on_write_rate": round(
            sum(row["predicted_action_class"] == "WAIT" for row in write_rows)
            / max(1, len(write_rows)),
            8,
        ),
        "invalid_output_rate": round(
            sum(row["predicted_action_class"] == "INVALID" for row in predictions)
            / max(1, total),
            8,
        ),
        "target_wait_ratio": round(len(wait_rows) / max(1, total), 8),
        "predicted_wait_ratio": round(
            sum(row["predicted_action_class"] == "WAIT" for row in predictions)
            / max(1, total),
            8,
        ),
        "parent_all_actions_exact_rate": round(
            sum(all(values) for values in parent_scores.values()) / max(1, len(parent_scores)), 8
        ),
        "parent_macro_action_accuracy": round(
            float(np.mean([sum(values) / len(values) for values in parent_scores.values()])), 8
        ),
        "per_class": per_class,
        "confusion": {
            f"{target}->{predicted}": count
            for (target, predicted), count in sorted(confusion.items())
        },
    }
