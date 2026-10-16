#!/usr/bin/env python3
"""Generate deterministic Fine-tune 1 translations for every English prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from nltk.tokenize import TreebankWordDetokenizer


DETOKENIZER = TreebankWordDetokenizer()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--finetune1-scripts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-parents", type=int)
    args = parser.parse_args()

    sys.path.insert(0, str(args.finetune1_scripts.resolve()))
    from finetune_common import normalize_prediction, set_seed  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    config = read_json(args.config)
    set_seed(int(config["seed"]))
    pairs = [row for row in read_jsonl(args.pairs) if row.get("split") == args.split]
    if args.limit_parents is not None:
        pairs = pairs[: args.limit_parents]
    if not pairs:
        raise SystemExit(f"No parents found for split={args.split}")

    tasks: list[dict[str, Any]] = []
    for pair in pairs:
        source_tokens = [str(value) for value in pair["source_tokens"]]
        for source_position in range(1, len(source_tokens) + 1):
            tasks.append(
                {
                    "parent_id": pair["id"],
                    "split": args.split,
                    "source_position": source_position,
                    "source_token_count": len(source_tokens),
                    "source_prefix_text_en": DETOKENIZER.detokenize(
                        source_tokens[:source_position]
                    ),
                    "source_text_en": pair["source_text_en"],
                    "source_tokens": source_tokens,
                    "reference_pseudo_asl_gloss": pair["target_pseudo_asl_gloss"],
                    "reference_pseudo_asl_gloss_tokens": pair[
                        "target_pseudo_asl_gloss_tokens"
                    ],
                }
            )

    completed: set[tuple[str, int]] = set()
    if args.output.exists():
        for row in read_jsonl(args.output):
            completed.add((str(row["parent_id"]), int(row["source_position"])))
    pending = [
        row
        for row in tasks
        if (str(row["parent_id"]), int(row["source_position"])) not in completed
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        dtype=torch.bfloat16,
        attn_implementation=config["attention_implementation"],
        device_map={"": args.device},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(
        base_model, str(args.adapter_dir), is_trainable=False
    )
    model.eval()
    model.config.use_cache = True
    system_prompt = str(config["system_prompt"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.output.exists() else "w"
    with args.output.open(mode, encoding="utf-8", newline="\n") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": row["source_prefix_text_en"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row in batch
            ]
            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=int(config["max_length"]),
                return_tensors="pt",
            ).to(args.device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=int(config["max_new_tokens"]),
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[:, encoded["input_ids"].shape[1] :]
            raw_predictions = tokenizer.batch_decode(
                new_tokens, skip_special_tokens=True
            )
            for row, raw in zip(batch, raw_predictions):
                prediction = normalize_prediction(raw)
                output = {
                    **row,
                    "predicted_pseudo_asl_gloss": prediction,
                    "predicted_pseudo_asl_gloss_tokens": prediction.split(),
                    "raw_prediction": raw.strip(),
                    "generator": "finetune1_qwen2_5_7b_lora_greedy",
                    "target_is_gold": False,
                }
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            handle.flush()
            done = min(start + len(batch), len(pending))
            print(f"[prefix-v2.3] {done}/{len(pending)} pending prefixes", flush=True)

    final_rows = read_jsonl(args.output)
    expected_keys = {
        (str(row["parent_id"]), int(row["source_position"])) for row in tasks
    }
    actual_keys = {
        (str(row["parent_id"]), int(row["source_position"])) for row in final_rows
    }
    duplicates = len(final_rows) - len(actual_keys)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "finetune1_prefix_retranslation_for_local_agreement_v2_3",
        "scientific_status": "model_predictions_against_pseudo_gloss_not_gold_asl",
        "split": args.split,
        "parent_count": len(pairs),
        "expected_prefix_count": len(tasks),
        "record_count": len(final_rows),
        "missing_prefix_count": len(expected_keys - actual_keys),
        "unexpected_prefix_count": len(actual_keys - expected_keys),
        "duplicate_prefix_count": duplicates,
        "complete": actual_keys == expected_keys and duplicates == 0,
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "adapter_dir": str(args.adapter_dir),
        "detokenizer": "nltk.tokenize.TreebankWordDetokenizer",
        "decoding": {"do_sample": False, "max_new_tokens": config["max_new_tokens"]},
        "input_sha256": sha256(args.pairs),
        "output_sha256": sha256(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
