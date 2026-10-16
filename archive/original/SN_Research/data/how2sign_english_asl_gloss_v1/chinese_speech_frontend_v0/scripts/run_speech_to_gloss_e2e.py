#!/usr/bin/env python3
"""Connect committed English speech translations to Fine-tune 2 actions."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import torch
from nltk.tokenize import TreebankWordDetokenizer, TreebankWordTokenizer


WRITE_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9'_-]*$")
WORD_TOKENIZER = TreebankWordTokenizer()
WORD_DETOKENIZER = TreebankWordDetokenizer()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def state_prompt(source_tokens: list[str], committed: list[str], complete: bool) -> str:
    source = WORD_DETOKENIZER.detokenize(source_tokens)
    committed_text = " ".join(committed) if committed else "<EMPTY>"
    status = "COMPLETE" if complete else "INCOMPLETE"
    return (
        "Observed English prefix:\n"
        f"{source}\n\n"
        "Already committed pseudo ASL gloss:\n"
        f"{committed_text}\n\n"
        f"Source status: {status}\n"
        "Return exactly one next action."
    )


def token_overlap(target: list[str], prediction: list[str]) -> tuple[float, float, float]:
    target_counter = Counter(target)
    prediction_counter = Counter(prediction)
    overlap = sum((target_counter & prediction_counter).values())
    precision = overlap / max(1, len(prediction))
    recall = overlap / max(1, len(target))
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return precision, recall, f1


def average_lagging(
    delays: list[float], source_duration: float, target_length: int
) -> float | None:
    """Reference-length AL in seconds for speech-to-token translation."""
    if not delays or source_duration <= 0 or target_length <= 0:
        return None
    gamma = target_length / source_duration
    tau = next(
        (index for index, delay in enumerate(delays, 1) if delay >= source_duration),
        len(delays),
    )
    return mean(
        delays[index - 1] - (index - 1) / gamma
        for index in range(1, tau + 1)
    )


def length_adaptive_average_lagging(
    delays: list[float], source_duration: float, reference_length: int
) -> float | None:
    """LAAL in seconds, robust to over-generation relative to the reference."""
    if not delays or source_duration <= 0 or reference_length <= 0:
        return None
    adaptive_length = max(reference_length, len(delays))
    gamma = adaptive_length / source_duration
    tau = next(
        (index for index, delay in enumerate(delays, 1) if delay >= source_duration),
        len(delays),
    )
    return mean(
        delays[index - 1] - (index - 1) / gamma
        for index in range(1, tau + 1)
    )


def average_proportion(delays: list[float], source_duration: float) -> float | None:
    if not delays or source_duration <= 0:
        return None
    return sum(delays) / (source_duration * len(delays))


def differentiable_average_lagging(
    delays: list[float], source_duration: float, target_length: int
) -> float | None:
    if not delays or source_duration <= 0 or target_length <= 0:
        return None
    gamma = target_length / source_duration
    adjusted = [float(delays[0])]
    for delay in delays[1:]:
        adjusted.append(max(float(delay), adjusted[-1] + 1.0 / gamma))
    return mean(value - index / gamma for index, value in enumerate(adjusted))


def finite_round(value: float | None, digits: int = 8) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def serial_processing_interval(
    audio_time: float, prior_ready_time: float, processing_seconds: float
) -> tuple[float, float]:
    """Place one model call on a single serial worker after audio is available."""
    if audio_time < 0 or prior_ready_time < 0 or processing_seconds < 0:
        raise ValueError("Serial processing times must be non-negative")
    start = max(float(audio_time), float(prior_ready_time))
    return start, start + float(processing_seconds)


@torch.inference_mode()
def predict_action(model: Any, tokenizer: Any, prompt: str, config: dict, device: str):
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(
        rendered,
        truncation=True,
        max_length=int(config["max_length"]),
        return_tensors="pt",
    ).to(device)
    started = perf_counter()
    generated = model.generate(
        **encoded,
        max_new_tokens=int(config["max_new_tokens"]),
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    elapsed = perf_counter() - started
    new_tokens = generated[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True), elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend-jsonl", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-post-source-actions", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = read_json(args.config)
    rows = read_jsonl(args.frontend_jsonl)
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.overwrite and args.output_jsonl.exists():
        args.output_jsonl.unlink()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        dtype=torch.bfloat16,
        attn_implementation=config["attention_implementation"],
        device_map={"": args.device},
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(
        base_model, str(args.adapter_dir), is_trainable=False
    )
    model.eval()
    model.config.use_cache = True

    completed = 0
    failed = 0
    run_started = perf_counter()
    for index, row in enumerate(rows, 1):
        result = dict(row)
        observed_tokens: list[str] = []
        committed_gloss: list[str] = []
        action_events: list[dict[str, Any]] = []
        write_source_positions: list[int] = []
        quality_flags: list[str] = []
        done = False
        premature_end = False
        processing_total = 0.0
        simulated_serial_clock = 0.0

        target_tokens = row["reference_pseudo_asl_gloss"].split()
        source_token_count = sum(
            len(event.get("emitted_english_tokens", []))
            for event in row.get("streaming_frontend_commit_events", [])
        )
        max_actions = source_token_count + max(20, 3 * len(target_tokens) + 10)

        def run_one_action(audio_time: float, source_complete: bool, trigger: str) -> str:
            nonlocal done, premature_end, processing_total, simulated_serial_clock
            prompt = state_prompt(observed_tokens, committed_gloss, source_complete)
            raw, processing = predict_action(
                model, tokenizer, prompt, config, args.device
            )
            processing_total += processing
            processing_start, processing_finish = serial_processing_interval(
                float(audio_time), simulated_serial_clock, processing
            )
            simulated_serial_clock = processing_finish
            action = normalize_action(raw)
            kind = action_class(action, config["wait_token"], config["end_token"])
            before = list(committed_gloss)
            if kind == "WRITE":
                committed_gloss.append(action)
                write_source_positions.append(len(observed_tokens))
            elif kind == "END":
                premature_end = not source_complete
                done = True
            action_events.append(
                {
                    "action_index": len(action_events),
                    "audio_read_seconds": round(audio_time, 6),
                    "trigger": trigger,
                    "observed_english_token_count": len(observed_tokens),
                    "observed_english_prefix": WORD_DETOKENIZER.detokenize(
                        observed_tokens
                    ),
                    "source_status": "COMPLETE" if source_complete else "INCOMPLETE",
                    "committed_gloss_before": before,
                    "raw_prediction": raw.strip(),
                    "predicted_action": action,
                    "predicted_action_class": kind,
                    "committed_gloss_after": list(committed_gloss),
                    "processing_seconds": round(processing, 6),
                    "simulated_serial_processing_start_seconds": round(
                        processing_start, 6
                    ),
                    "simulated_serial_action_ready_seconds": round(
                        processing_finish, 6
                    ),
                }
            )
            return kind

        def run_at_current_prefix(
            audio_time: float, source_complete: bool, trigger: str
        ) -> None:
            """WRITE holds the prefix; WAIT consumes the next source token."""
            nonlocal done
            while not done and len(action_events) < max_actions:
                kind = run_one_action(audio_time, source_complete, trigger)
                if kind in {"WAIT", "INVALID"} and not source_complete:
                    break

        try:
            if row.get("streaming_frontend_status") != "completed":
                raise ValueError("Streaming English frontend did not complete")
            frontend_events = row["streaming_frontend_commit_events"]
            for frontend_index, frontend_event in enumerate(frontend_events):
                if done:
                    break
                committed_prefix = frontend_event["committed_english_prefix"]
                new_tokens = list(frontend_event.get("emitted_english_tokens", []))
                is_final_frontend_event = frontend_index == len(frontend_events) - 1
                event_is_complete = (
                    str(frontend_event.get("source_status", "")).upper() == "COMPLETE"
                )
                for token_index, token in enumerate(new_tokens):
                    observed_tokens.append(token)
                    complete_now = event_is_complete and token_index == len(new_tokens) - 1
                    run_at_current_prefix(
                        float(frontend_event["audio_read_seconds"]),
                        complete_now,
                        "new_english_token",
                    )
                    if done:
                        break
                reconstructed_prefix = WORD_DETOKENIZER.detokenize(observed_tokens)
                if reconstructed_prefix.casefold() != committed_prefix.casefold():
                    quality_flags.append("english_commit_audit_mismatch")
                if event_is_complete and not new_tokens and not done:
                    run_at_current_prefix(
                        float(frontend_event["audio_read_seconds"]),
                        True,
                        "source_completion_without_new_token",
                    )
                if is_final_frontend_event and not event_is_complete:
                    quality_flags.append("frontend_missing_complete_status")

            if not observed_tokens:
                raise RuntimeError("No committed English tokens reached Fine-tune 2")
            final_audio_time = float(row["audio_duration_seconds"])
            if not done and len(action_events) < max_actions:
                run_at_current_prefix(final_audio_time, True, "forced_source_completion")
            termination = (
                "premature_end"
                if premature_end
                else "end_after_full_source"
                if done
                else "maximum_actions_reached"
            )
            completed_after_full_source = termination == "end_after_full_source"
            precision, recall, f1 = token_overlap(target_tokens, committed_gloss)
            similarity = SequenceMatcher(
                None, target_tokens, committed_gloss
            ).ratio()
            write_events = [
                event
                for event in action_events
                if event["predicted_action_class"] == "WRITE"
            ]
            write_audio_delays = [
                float(event["audio_read_seconds"]) for event in write_events
            ]
            write_wall_delays = [
                float(event["simulated_serial_action_ready_seconds"])
                for event in write_events
            ]
            audio_duration = float(row["audio_duration_seconds"])
            audio_al = average_lagging(
                write_audio_delays, audio_duration, len(target_tokens)
            )
            audio_laal = length_adaptive_average_lagging(
                write_audio_delays, audio_duration, len(target_tokens)
            )
            audio_ap = average_proportion(write_audio_delays, audio_duration)
            audio_dal = differentiable_average_lagging(
                write_audio_delays, audio_duration, len(target_tokens)
            )
            wall_al = average_lagging(
                write_wall_delays, audio_duration, len(target_tokens)
            )
            wall_laal = length_adaptive_average_lagging(
                write_wall_delays, audio_duration, len(target_tokens)
            )
            wall_ap = average_proportion(write_wall_delays, audio_duration)
            wall_dal = differentiable_average_lagging(
                write_wall_delays, audio_duration, len(target_tokens)
            )
            result.update(
                {
                    "speech_to_gloss_status": "completed",
                    "speech_to_gloss_error": None,
                    "speech_to_gloss_predicted_pseudo_asl_gloss": " ".join(
                        committed_gloss
                    ),
                    "speech_to_gloss_predicted_tokens": committed_gloss,
                    "speech_to_gloss_action_events": action_events,
                    "speech_to_gloss_termination_reason": termination,
                    "speech_to_gloss_completed_with_end": done,
                    "speech_to_gloss_completed_after_full_source": (
                        completed_after_full_source
                    ),
                    "speech_to_gloss_premature_end": premature_end,
                    "speech_to_gloss_quality_flags": sorted(set(quality_flags)),
                    "speech_to_gloss_first_write_audio_seconds": (
                        write_events[0]["audio_read_seconds"] if write_events else None
                    ),
                    "speech_to_gloss_last_write_audio_seconds": (
                        write_events[-1]["audio_read_seconds"] if write_events else None
                    ),
                    "speech_to_gloss_processing_seconds": round(
                        processing_total, 6
                    ),
                    "speech_to_gloss_action_count": len(action_events),
                    "speech_to_gloss_maximum_actions": max_actions,
                    "speech_to_gloss_write_source_positions": write_source_positions,
                    "speech_to_gloss_write_audio_seconds": write_audio_delays,
                    "speech_to_gloss_write_compute_adjusted_seconds": (
                        write_wall_delays
                    ),
                    "speech_to_gloss_reference_length_al_seconds": finite_round(
                        audio_al
                    ),
                    "speech_to_gloss_laal_seconds": finite_round(audio_laal),
                    "speech_to_gloss_average_proportion": finite_round(audio_ap),
                    "speech_to_gloss_dal_seconds": finite_round(audio_dal),
                    "speech_to_gloss_compute_adjusted_reference_length_al_seconds": (
                        finite_round(wall_al)
                    ),
                    "speech_to_gloss_compute_adjusted_laal_seconds": finite_round(
                        wall_laal
                    ),
                    "speech_to_gloss_compute_adjusted_average_proportion": (
                        finite_round(wall_ap)
                    ),
                    "speech_to_gloss_compute_adjusted_dal_seconds": finite_round(
                        wall_dal
                    ),
                    "speech_to_gloss_processing_realtime_factor": round(
                        processing_total / max(audio_duration, 1e-12), 8
                    ),
                    "speech_to_gloss_simulated_serial_finish_seconds": round(
                        simulated_serial_clock, 6
                    ),
                    "pseudo_gloss_exact_match": target_tokens == committed_gloss,
                    "pseudo_gloss_sequence_similarity": round(similarity, 8),
                    "pseudo_gloss_token_precision": round(precision, 8),
                    "pseudo_gloss_token_recall": round(recall, 8),
                    "pseudo_gloss_token_f1": round(f1, 8),
                    "reference_pseudo_asl_gloss_is_gold": False,
                }
            )
            completed += 1
            status = "completed"
        except Exception as exc:
            result.update(
                {
                    "speech_to_gloss_status": "failed",
                    "speech_to_gloss_error": repr(exc),
                    "speech_to_gloss_predicted_pseudo_asl_gloss": None,
                    "speech_to_gloss_action_events": action_events,
                }
            )
            failed += 1
            status = "failed"
        result["speech_to_gloss_generated_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        append_jsonl(args.output_jsonl, result)
        print(f"[e2e] {index}/{len(rows)} {row['id']} {status}", flush=True)

    output_rows = read_jsonl(args.output_jsonl)
    valid = [row for row in output_rows if row.get("speech_to_gloss_status") == "completed"]
    termination_counts = Counter(
        row["speech_to_gloss_termination_reason"] for row in valid
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "end_to_end_synthetic_speech_to_weak_pseudo_gloss_not_gold_asl",
        "frontend_jsonl": str(args.frontend_jsonl),
        "adapter_dir": str(args.adapter_dir),
        "record_count": len(rows),
        "completed_count": completed,
        "failed_count": failed,
        "termination_reason_counts": dict(termination_counts),
        "end_completion_rate": round(
            sum(row["speech_to_gloss_completed_with_end"] for row in valid)
            / max(1, len(valid)),
            8,
        ),
        "valid_end_completion_rate": round(
            sum(
                row["speech_to_gloss_completed_after_full_source"] for row in valid
            )
            / max(1, len(valid)),
            8,
        ),
        "premature_end_rate": round(
            sum(row["speech_to_gloss_premature_end"] for row in valid)
            / max(1, len(valid)),
            8,
        ),
        "pseudo_gloss_exact_match_rate": round(
            sum(row["pseudo_gloss_exact_match"] for row in valid)
            / max(1, len(valid)),
            8,
        ),
        "mean_pseudo_gloss_sequence_similarity": round(
            mean(row["pseudo_gloss_sequence_similarity"] for row in valid), 8
        )
        if valid
        else None,
        "mean_pseudo_gloss_token_f1": round(
            mean(row["pseudo_gloss_token_f1"] for row in valid), 8
        )
        if valid
        else None,
        "mean_first_gloss_write_audio_seconds": round(
            mean(
                row["speech_to_gloss_first_write_audio_seconds"]
                for row in valid
                if row["speech_to_gloss_first_write_audio_seconds"] is not None
            ),
            8,
        )
        if any(
            row["speech_to_gloss_first_write_audio_seconds"] is not None
            for row in valid
        )
        else None,
        "mean_reference_length_al_seconds": round(
            mean(
                row["speech_to_gloss_reference_length_al_seconds"]
                for row in valid
                if row["speech_to_gloss_reference_length_al_seconds"] is not None
            ),
            8,
        )
        if any(
            row["speech_to_gloss_reference_length_al_seconds"] is not None
            for row in valid
        )
        else None,
        "mean_laal_seconds": round(
            mean(
                row["speech_to_gloss_laal_seconds"]
                for row in valid
                if row["speech_to_gloss_laal_seconds"] is not None
            ),
            8,
        )
        if any(row["speech_to_gloss_laal_seconds"] is not None for row in valid)
        else None,
        "mean_average_proportion": round(
            mean(
                row["speech_to_gloss_average_proportion"]
                for row in valid
                if row["speech_to_gloss_average_proportion"] is not None
            ),
            8,
        )
        if any(
            row["speech_to_gloss_average_proportion"] is not None for row in valid
        )
        else None,
        "mean_dal_seconds": round(
            mean(
                row["speech_to_gloss_dal_seconds"]
                for row in valid
                if row["speech_to_gloss_dal_seconds"] is not None
            ),
            8,
        )
        if any(row["speech_to_gloss_dal_seconds"] is not None for row in valid)
        else None,
        "mean_compute_adjusted_laal_seconds": round(
            mean(
                row["speech_to_gloss_compute_adjusted_laal_seconds"]
                for row in valid
                if row["speech_to_gloss_compute_adjusted_laal_seconds"] is not None
            ),
            8,
        )
        if any(
            row["speech_to_gloss_compute_adjusted_laal_seconds"] is not None
            for row in valid
        )
        else None,
        "mean_processing_seconds_per_sample": round(
            mean(row["speech_to_gloss_processing_seconds"] for row in valid), 8
        )
        if valid
        else None,
        "mean_processing_realtime_factor": round(
            mean(row["speech_to_gloss_processing_realtime_factor"] for row in valid),
            8,
        )
        if valid
        else None,
        "run_wall_seconds": round(perf_counter() - run_started, 6),
        "important_limitations": [
            "Mandarin speech is synthetic multi-speaker TTS.",
            "The ASL gloss reference is an LLM-generated weak pseudo label, not gold ASL.",
            "Fine-tune 2 timing supervision is text-derived pseudo alignment.",
        ],
        "rollout_policy": {
            "write": "commit one gloss token and keep the same English prefix",
            "wait": "consume the next committed English token",
            "source_end": "continue at the complete source until END or the action limit",
            "latency": "AL/LAAL/AP/DAL use audio-read timestamps, excluding model compute time",
            "compute_adjusted_latency": (
                "a serial online upper bound: each action starts after both its audio "
                "timestamp and the previous action finish, then adds measured inference time"
            ),
        },
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
