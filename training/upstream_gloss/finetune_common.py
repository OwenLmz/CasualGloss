#!/usr/bin/env python3
"""Shared utilities for full pseudo-gloss LoRA training and evaluation."""

from __future__ import annotations

import hashlib
import json
import random
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


GLOSS_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9'_-]*(?: [A-Z0-9][A-Z0-9'_-]*)*$")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
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


def normalize_prediction(text: str) -> str:
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return " ".join(first_line.split()).upper()


class GlossDataset(Dataset):
    """Tokenize chat examples and mask every token except the assistant target."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        system_prompt: str,
        max_length: int,
    ) -> None:
        self.rows = rows
        self.examples: list[dict[str, Any]] = []
        self.truncated_example_count = 0
        self.max_original_length = 0
        self.max_target_length = 0
        for row in rows:
            prompt_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(row["source_text_en"])},
            ]
            full_messages = prompt_messages + [
                {
                    "role": "assistant",
                    "content": str(row["target_pseudo_asl_gloss"]),
                }
            ]
            prompt_ids = tokenizer.apply_chat_template(
                prompt_messages, tokenize=True, add_generation_prompt=True
            )
            full_ids = tokenizer.apply_chat_template(
                full_messages, tokenize=True, add_generation_prompt=False
            )
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError(f"Chat-template prefix mismatch for {row['id']}")
            target_ids = full_ids[len(prompt_ids) :]
            if not target_ids:
                raise ValueError(f"Empty assistant target for {row['id']}")
            self.max_original_length = max(self.max_original_length, len(full_ids))
            self.max_target_length = max(self.max_target_length, len(target_ids))
            if len(full_ids) > max_length:
                self.truncated_example_count += 1
                if len(target_ids) >= max_length:
                    target_ids = target_ids[: max_length - 1]
                prompt_budget = max_length - len(target_ids)
                prompt_ids = prompt_ids[-prompt_budget:]
                full_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
            if len(full_ids) != len(labels):
                raise ValueError(f"Input/label length mismatch for {row['id']}")
            self.examples.append(
                {
                    "id": row["id"],
                    "input_ids": full_ids,
                    "labels": labels,
                }
            )

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


class CausalCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        max_length = max(len(example["input_ids"]) for example in examples)
        input_ids, attention_mask, labels, ids = [], [], [], []
        for example in examples:
            padding = max_length - len(example["input_ids"])
            input_ids.append(example["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append([1] * len(example["input_ids"]) + [0] * padding)
            labels.append(example["labels"] + [-100] * padding)
            ids.append(example["id"])
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


@torch.inference_mode()
def teacher_forced_metrics_single(
    model: Any,
    dataset: Dataset,
    collator: CausalCollator,
    device: torch.device | str,
    batch_size: int = 1,
) -> dict[str, float | int]:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=True,
    )
    totals = [0.0, 0, 0, 0, 0]
    for batch in loader:
        tensor_batch = move_batch(batch, device)
        outputs = model(**tensor_batch)
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
    return metrics_from_totals(totals)


def metrics_from_totals(totals: list[float | int]) -> dict[str, float | int]:
    loss_numerator, target_tokens, correct_tokens, exact_sequences, sequences = totals
    return {
        "pseudo_label_loss": round(float(loss_numerator) / max(1, int(target_tokens)), 8),
        "pseudo_label_token_accuracy_teacher_forced": round(
            int(correct_tokens) / max(1, int(target_tokens)), 8
        ),
        "pseudo_label_sequence_exact_match_teacher_forced": round(
            int(exact_sequences) / max(1, int(sequences)), 8
        ),
        "target_token_count": int(target_tokens),
        "sequence_count": int(sequences),
    }


@torch.inference_mode()
def batched_generation_metrics(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    system_prompt: str,
    device: torch.device | str,
    max_new_tokens: int,
    batch_size: int,
    stage: str,
    split: str,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    model.eval()
    old_use_cache = bool(model.config.use_cache)
    old_padding_side = tokenizer.padding_side
    model.config.use_cache = True
    tokenizer.padding_side = "left"
    prediction_rows: list[dict[str, Any]] = []
    similarities: list[float] = []
    exact_count = 0
    compliant_count = 0
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        prompt_texts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": str(row["source_text_en"])},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch_rows
        ]
        encoded = tokenizer(
            prompt_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        raw_predictions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for row, raw_prediction in zip(batch_rows, raw_predictions):
            raw_prediction = raw_prediction.strip()
            prediction = normalize_prediction(raw_prediction)
            target = str(row["target_pseudo_asl_gloss"]).strip()
            similarity = SequenceMatcher(
                None, prediction.split(), target.split()
            ).ratio()
            exact = prediction == target
            format_compliant = bool(GLOSS_PATTERN.fullmatch(prediction))
            exact_count += int(exact)
            compliant_count += int(format_compliant)
            similarities.append(similarity)
            prediction_rows.append(
                {
                    "stage": stage,
                    "split": split,
                    "id": row["id"],
                    "source_text_en": row["source_text_en"],
                    "target_pseudo_asl_gloss": target,
                    "predicted_pseudo_asl_gloss": prediction,
                    "raw_prediction": raw_prediction,
                    "pseudo_label_exact_match": exact,
                    "pseudo_label_token_sequence_similarity": round(similarity, 8),
                    "gloss_format_compliant": format_compliant,
                    "target_is_gold": False,
                }
            )
    model.config.use_cache = old_use_cache
    tokenizer.padding_side = old_padding_side
    return (
        {
            "record_count": len(rows),
            "pseudo_label_generation_exact_match": round(
                exact_count / max(1, len(rows)), 8
            ),
            "pseudo_label_generation_token_sequence_similarity": round(
                float(np.mean(similarities)) if similarities else 0.0, 8
            ),
            "gloss_format_compliance": round(
                compliant_count / max(1, len(rows)), 8
            ),
        },
        prediction_rows,
    )

