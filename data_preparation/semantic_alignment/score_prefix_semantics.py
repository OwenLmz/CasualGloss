#!/usr/bin/env python3
"""Score pseudo-gloss tokens under every incremental English prefix."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from nltk.tokenize import TreebankWordDetokenizer
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DETOKENIZER = TreebankWordDetokenizer()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def target_encoding(tokenizer: Any, gloss_tokens: list[str]) -> tuple[list[int], list[int]]:
    target_text = " ".join(gloss_tokens)
    encoded = tokenizer(
        target_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    target_ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    spans = [(match.start(), match.end()) for match in re.finditer(r"\S+", target_text)]
    owners: list[int] = []
    for start, end in offsets:
        candidates = [
            (max(0, min(end, span_end) - max(start, span_start)), index)
            for index, (span_start, span_end) in enumerate(spans)
        ]
        overlap, owner = max(candidates, default=(0, -1))
        if overlap <= 0:
            # Qwen occasionally emits whitespace as a standalone subtoken. The
            # whitespace is part of predicting the following gloss word, so it
            # belongs to the next non-whitespace span (or the final span at EOS).
            following = [index for index, (span_start, _) in enumerate(spans) if span_start >= end]
            if target_text[start:end].isspace() and following:
                owner = following[0]
            elif target_text[start:end].isspace() and spans:
                owner = len(spans) - 1
            else:
                raise ValueError(f"Could not map target subtoken offset {(start, end)} in {target_text!r}")
        owners.append(owner)
    missing = sorted(set(range(len(gloss_tokens))) - set(owners))
    if missing:
        raise ValueError(f"Gloss tokens without target subtokens: {missing} in {target_text!r}")
    return target_ids, owners


def prompt_ids(tokenizer: Any, system_prompt: str, source_text: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": source_text},
        ],
        tokenize=True,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def score_parent(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    system_prompt: str,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    source_tokens = list(row["source_tokens"])
    gloss_tokens = list(row["target_pseudo_asl_gloss_tokens"])
    target_ids, target_owners = target_encoding(tokenizer, gloss_tokens)
    prefix_texts = [""] + [DETOKENIZER.detokenize(source_tokens[:index]) for index in range(1, len(source_tokens) + 1)]
    sequences: list[dict[str, Any]] = []
    for source_position, source_text in enumerate(prefix_texts):
        prefix_ids = prompt_ids(tokenizer, system_prompt, source_text)
        full_ids = prefix_ids + target_ids
        if len(full_ids) > max_length:
            raise ValueError(
                f"Sequence length {len(full_ids)} exceeds max_length={max_length} for {row['id']}"
            )
        sequences.append(
            {
                "source_position": source_position,
                "source_text": source_text,
                "input_ids": full_ids,
                "target_start": len(prefix_ids),
            }
        )

    matrix_mean: list[list[float]] = []
    matrix_min: list[list[float]] = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        longest = max(len(item["input_ids"]) for item in batch)
        ids = []
        masks = []
        for item in batch:
            padding = longest - len(item["input_ids"])
            ids.append(item["input_ids"] + [tokenizer.pad_token_id] * padding)
            masks.append([1] * len(item["input_ids"]) + [0] * padding)
        input_ids = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        for batch_index, item in enumerate(batch):
            positions = torch.arange(
                item["target_start"] - 1,
                item["target_start"] - 1 + len(target_ids),
                device=device,
            )
            token_logits = logits[batch_index, positions].float()
            target_tensor = torch.tensor(target_ids, dtype=torch.long, device=device)
            chosen = token_logits.gather(1, target_tensor[:, None]).squeeze(1)
            logprobs = chosen - torch.logsumexp(token_logits, dim=1)
            values = logprobs.cpu().tolist()
            gloss_means = []
            gloss_mins = []
            for gloss_index in range(len(gloss_tokens)):
                owned = [value for value, owner in zip(values, target_owners) if owner == gloss_index]
                gloss_means.append(round(sum(owned) / len(owned), 6))
                gloss_mins.append(round(min(owned), 6))
            matrix_mean.append(gloss_means)
            matrix_min.append(gloss_mins)

    return {
        "id": row["id"],
        "semantic_v2_pilot_index": row["semantic_v2_pilot_index"],
        "semantic_v2_shard_id": row["semantic_v2_shard_id"],
        "source_tokens": source_tokens,
        "source_prefix_texts": prefix_texts,
        "target_pseudo_asl_gloss_tokens": gloss_tokens,
        "target_subtoken_count": len(target_ids),
        "target_subtoken_owner_gloss_indices": target_owners,
        "gloss_token_mean_logprob_by_source_position": matrix_mean,
        "gloss_token_min_subtoken_logprob_by_source_position": matrix_min,
        "empty_source_geometric_mean_probabilities": [round(math.exp(value), 8) for value in matrix_mean[0]],
        "full_source_geometric_mean_probabilities": [round(math.exp(value), 8) for value in matrix_mean[-1]],
        "scoring_status": "completed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    rows = read_jsonl(args.input)
    torch.manual_seed(int(config["selection_seed"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    adapter_path = Path(config["finetune1_adapter"])
    adapter_model_path = adapter_path / "adapter_model.safetensors"
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model_id"],
        revision=config["base_model_revision"],
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model_id"],
        revision=config["base_model_revision"],
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": device.index or 0},
        attn_implementation=config["attention_implementation"],
    )
    model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model.eval()

    outputs = []
    failures = []
    for index, row in enumerate(rows, 1):
        try:
            outputs.append(
                score_parent(
                    model,
                    tokenizer,
                    row,
                    config["system_prompt"],
                    device,
                    int(config["batch_size"]),
                    int(config["max_length"]),
                )
            )
        except Exception as error:
            failures.append({"id": row.get("id"), "error": f"{type(error).__name__}: {error}"})
        if index % 10 == 0 or index == len(rows):
            print(f"[semantic-score] {index}/{len(rows)} completed={len(outputs)} failed={len(failures)}", flush=True)
    write_jsonl(args.output, outputs)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(args.input),
        "output_scores": str(args.output),
        "requested_parent_count": len(rows),
        "completed_parent_count": len(outputs),
        "failed_parent_count": len(failures),
        "failures": failures[:20],
        "elapsed_seconds": round(time.time() - started, 3),
        "device": str(device),
        "visible_gpu": torch.cuda.get_device_name(device),
        "base_model_id": config["base_model_id"],
        "base_model_revision": config["base_model_revision"],
        "adapter_path": str(adapter_path),
        "adapter_model_sha256": file_sha256(adapter_model_path),
        "system_prompt": config["system_prompt"],
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "tokenizers", "nltk")
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ok": len(outputs) == len(rows) and not failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
