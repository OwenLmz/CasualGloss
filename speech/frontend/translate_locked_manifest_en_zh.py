#!/usr/bin/env python3
"""Deterministically translate a fixed English manifest into spoken Mandarin."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = """You translate English into natural spoken Mandarin Chinese for speech synthesis.
Preserve the complete meaning, especially negation, modality, names, quantities, and relations.
Do not summarize, explain, romanize, or add information.
Output only one Mandarin translation and nothing else."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def clean_translation(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:chinese|zh)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^(?:中文|翻译|译文)\s*[:：]\s*", "", text)
    return " ".join(text.split()).strip()


def quality_flags(source_en: str, translation_zh: str) -> list[str]:
    flags: list[str] = []
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", translation_zh))
    if cjk_count == 0:
        flags.append("no_cjk_character")
    if translation_zh == source_en:
        flags.append("unchanged_from_english")
    if len(translation_zh) > max(120, 5 * len(source_en)):
        flags.append("unexpectedly_long")
    if any(marker in translation_zh.lower() for marker in ("translation:", "mandarin:")):
        flags.append("meta_text_present")
    return flags


def render_prompt(tokenizer: Any, source: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"English: {source}\nMandarin:"},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    previous = {} if args.overwrite else {row["id"]: row for row in read_jsonl(args.output)}

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()

    translated_rows: list[dict[str, Any]] = []
    run_started = perf_counter()
    for index, original in enumerate(rows, 1):
        existing = previous.get(original["id"])
        if existing and existing.get("source_text_zh_status") == "completed":
            translated_rows.append(existing)
            print(f"[en-zh] {index}/{len(rows)} {original['id']} reused", flush=True)
            continue

        row = dict(original)
        started = perf_counter()
        try:
            rendered = render_prompt(tokenizer, str(row["reference_text_en"]))
            inputs = tokenizer(rendered, return_tensors="pt").to(args.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0, inputs["input_ids"].shape[1] :]
            translation = clean_translation(
                tokenizer.decode(new_tokens, skip_special_tokens=True)
            )
            flags = quality_flags(str(row["reference_text_en"]), translation)
            status = "completed" if translation and not flags else "flagged"
            row.update(
                {
                    "source_text_zh": translation,
                    "source_text_zh_status": status,
                    "source_text_zh_quality_flags": flags,
                    "source_text_zh_error": None,
                }
            )
        except Exception as exc:
            row.update(
                {
                    "source_text_zh": None,
                    "source_text_zh_status": "failed",
                    "source_text_zh_quality_flags": ["generation_failure"],
                    "source_text_zh_error": repr(exc),
                }
            )
        row.update(
            {
                "source_text_zh_model": "Qwen/Qwen3-4B-Instruct-2507",
                "source_text_zh_model_revision": args.model_revision,
                "source_text_zh_decoding": "greedy",
                "source_text_zh_processing_seconds": round(perf_counter() - started, 6),
                "source_text_zh_generated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        translated_rows.append(row)
        write_jsonl(args.output, translated_rows + rows[len(translated_rows) :])
        print(
            f"[en-zh] {index}/{len(rows)} {row['id']} {row['source_text_zh_status']}",
            flush=True,
        )

    write_jsonl(args.output, translated_rows)
    completed = sum(row.get("source_text_zh_status") == "completed" for row in translated_rows)
    flagged = sum(row.get("source_text_zh_status") == "flagged" for row in translated_rows)
    failed = len(translated_rows) - completed - flagged
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "synthetic_source_translation_for_locked_e2e_evaluation",
        "record_count": len(translated_rows),
        "completed_count": completed,
        "flagged_count": flagged,
        "failed_count": failed,
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "model_revision": args.model_revision,
        "decoding": "greedy",
        "run_wall_seconds": round(perf_counter() - run_started, 6),
        "flagged_rows": [
            {"id": row["id"], "flags": row.get("source_text_zh_quality_flags", [])}
            for row in translated_rows
            if row.get("source_text_zh_status") == "flagged"
        ],
        "failed_rows": [
            {"id": row["id"], "error": row.get("source_text_zh_error")}
            for row in translated_rows
            if row.get("source_text_zh_status") == "failed"
        ],
        "ok": completed == len(translated_rows),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
