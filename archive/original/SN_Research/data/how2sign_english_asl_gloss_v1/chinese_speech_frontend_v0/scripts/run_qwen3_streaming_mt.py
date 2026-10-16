#!/usr/bin/env python3
"""Translate committed Mandarin ASR prefixes with Local Agreement."""

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


SYSTEM_PROMPT = """You are a professional simultaneous Chinese-to-English translator.
Translate only the Chinese source that has already been observed.
Do not predict or add information that is not present.
Preserve names, negation, modality, quantities, and relations.
Output only the English translation, with no explanation.
If the incomplete source is too ambiguous to translate safely, output <WAIT>."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def english_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*|[^\w\s]", text)


def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%\)\]])", r"\1", text)
    text = re.sub(r"([\(\[])\s+", r"\1", text)
    text = re.sub(r"\s+(['’])\s+", r"\1", text)
    return text.strip()


def longest_common_prefix(left: list[str], right: list[str]) -> list[str]:
    result: list[str] = []
    for left_token, right_token in zip(left, right):
        if left_token.casefold() != right_token.casefold():
            break
        result.append(right_token)
    return result


def generate_translation(
    model: Any,
    tokenizer: Any,
    source_prefix: str,
    source_complete: bool,
    max_new_tokens: int,
) -> tuple[str, float]:
    status = "COMPLETE" if source_complete else "INCOMPLETE"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Source status: {status}\n"
                f"Observed Chinese source: {source_prefix}\n"
                "English translation:"
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    started = perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = perf_counter() - started
    new_tokens = generated[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--holdback-tokens", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl)
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.overwrite and args.output_jsonl.exists():
        args.output_jsonl.unlink()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        local_files_only=True,
    )
    model.eval()

    completed = 0
    failed = 0
    conflict_count = 0
    run_started = perf_counter()
    for index, row in enumerate(rows, 1):
        result = dict(row)
        mt_events: list[dict[str, Any]] = []
        committed_tokens: list[str] = []
        previous_candidate: list[str] = []
        source_parts: list[str] = []
        mt_processing_total = 0.0
        flags: list[str] = []
        try:
            if row.get("zh_asr_status") != "completed":
                raise ValueError("Mandarin streaming ASR did not complete")
            asr_events = row["zh_asr_commit_events"]
            if not asr_events:
                raise ValueError("Mandarin streaming ASR emitted no events")
            for event_index, event in enumerate(asr_events):
                source_parts.append(event["committed_text"])
                source_prefix = "".join(source_parts).strip()
                is_complete = event_index == len(asr_events) - 1
                candidate_text, processing = generate_translation(
                    model,
                    tokenizer,
                    source_prefix,
                    is_complete,
                    args.max_new_tokens,
                )
                mt_processing_total += processing
                candidate_tokens = (
                    [] if candidate_text == "<WAIT>" else english_tokens(candidate_text)
                )
                emitted_tokens: list[str] = []
                conflict = False
                if is_complete:
                    if [t.casefold() for t in candidate_tokens[: len(committed_tokens)]] == [
                        t.casefold() for t in committed_tokens
                    ]:
                        emitted_tokens = candidate_tokens[len(committed_tokens) :]
                    else:
                        conflict = True
                        stable = longest_common_prefix(committed_tokens, candidate_tokens)
                        emitted_tokens = candidate_tokens[
                            min(len(committed_tokens), len(candidate_tokens)) :
                        ]
                else:
                    stable = longest_common_prefix(previous_candidate, candidate_tokens)
                    committable_length = max(
                        len(committed_tokens), len(stable) - args.holdback_tokens
                    )
                    if committable_length > len(committed_tokens):
                        if [t.casefold() for t in stable[: len(committed_tokens)]] == [
                            t.casefold() for t in committed_tokens
                        ]:
                            emitted_tokens = stable[
                                len(committed_tokens) : committable_length
                            ]
                        else:
                            conflict = True
                if conflict:
                    flags.append("committed_prefix_conflict")
                    conflict_count += 1
                committed_tokens.extend(emitted_tokens)
                previous_candidate = candidate_tokens
                mt_events.append(
                    {
                        "audio_read_seconds": event["audio_read_seconds"],
                        "source_status": "COMPLETE" if is_complete else "INCOMPLETE",
                        "observed_chinese_prefix": source_prefix,
                        "candidate_translation": candidate_text,
                        "emitted_english_tokens": emitted_tokens,
                        "committed_english_prefix": detokenize(committed_tokens),
                        "mt_processing_seconds": round(processing, 6),
                        "prefix_conflict": conflict,
                    }
                )
            hypothesis = detokenize(committed_tokens)
            if not hypothesis:
                raise RuntimeError("Streaming MT emitted no English text")
            asr_processing = float(row.get("zh_asr_processing_seconds", 0.0))
            duration = float(row["audio_duration_seconds"])
            emission_events = [e for e in mt_events if e["emitted_english_tokens"]]
            result.update(
                {
                    "streaming_frontend_status": "completed",
                    "streaming_frontend_error": None,
                    "streaming_frontend_method": "AlignAtt zh ASR + Qwen3-4B-Instruct-2507 Local Agreement MT",
                    "streaming_frontend_model_revision": args.model_revision,
                    "streaming_frontend_hypothesis_en": hypothesis,
                    "streaming_frontend_commit_events": mt_events,
                    "streaming_frontend_quality_flags": sorted(set(flags)),
                    "streaming_frontend_first_commit_audio_seconds": (
                        emission_events[0]["audio_read_seconds"] if emission_events else None
                    ),
                    "streaming_frontend_last_commit_audio_seconds": (
                        emission_events[-1]["audio_read_seconds"] if emission_events else None
                    ),
                    "streaming_frontend_processing_seconds": round(
                        asr_processing + mt_processing_total, 6
                    ),
                    "streaming_frontend_real_time_factor": round(
                        (asr_processing + mt_processing_total) / duration, 6
                    ),
                    "streaming_frontend_commit_event_count": len(emission_events),
                }
            )
            completed += 1
            status = "completed"
        except Exception as exc:
            result.update(
                {
                    "streaming_frontend_status": "failed",
                    "streaming_frontend_error": repr(exc),
                    "streaming_frontend_hypothesis_en": None,
                    "streaming_frontend_commit_events": mt_events,
                }
            )
            failed += 1
            status = "failed"
        result["streaming_frontend_generated_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        append_jsonl(args.output_jsonl, result)
        print(f"[qwen-mt] {index}/{len(rows)} {row['id']} {status}", flush=True)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "model_revision": args.model_revision,
        "policy": "two-update Local Agreement plus forced finalization at source end",
        "holdback_tokens": args.holdback_tokens,
        "record_count": len(rows),
        "completed_count": completed,
        "failed_count": failed,
        "prefix_conflict_event_count": conflict_count,
        "run_wall_seconds": round(perf_counter() - run_started, 6),
        "ok": completed == len(rows) and failed == 0,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
