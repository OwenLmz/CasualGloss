#!/usr/bin/env python3
"""Run pinned SimulStreaming/Whisper on a JSONL speech manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import soundfile as sf
import torch


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def clean_text(parts: list[str]) -> str:
    return " ".join("".join(parts).split())


def serialize_event(
    output: dict[str, Any],
    phase: str,
    audio_read_seconds: float,
    processing_seconds: float,
    cumulative_processing_seconds: float,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "audio_read_seconds": round(audio_read_seconds, 6),
        "computationally_aware_time_seconds": round(
            max(audio_read_seconds, cumulative_processing_seconds), 6
        ),
        "processing_seconds": round(processing_seconds, 6),
        "estimated_source_start_seconds": round(float(output["start"]), 6),
        "estimated_source_end_seconds": round(float(output["end"]), 6),
        "committed_text": output["text"],
        "committed_words": [
            {
                "text": word["text"],
                "start": round(float(word["start"]), 6),
                "end": round(float(word["end"]), 6),
            }
            for word in output.get("words", [])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--simulstreaming-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(args.simulstreaming_root.resolve()))
    from simulstreaming_whisper import SimulWhisperASR, SimulWhisperOnline

    config = read_json(args.config)
    stream_config = config["simulstreaming"]
    rows = read_jsonl(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]

    if args.overwrite and args.output_jsonl.exists():
        args.output_jsonl.unlink()
    completed_ids: set[str] = set()
    if args.output_jsonl.exists():
        completed_ids = {
            row["id"]
            for row in read_jsonl(args.output_jsonl)
            if row.get("streaming_asr_status") == "completed"
        }

    expected_hash = stream_config["whisper_checkpoint_sha256"]
    actual_hash = sha256(args.model_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Whisper checkpoint SHA256 mismatch: {actual_hash} != {expected_hash}"
        )

    device = stream_config["device"]
    if device.startswith("cuda"):
        torch.cuda.set_device(int(device.split(":", maxsplit=1)[1]))
    asr = SimulWhisperASR(
        language=stream_config["source_language"],
        model_path=str(args.model_path),
        cif_ckpt_path=None,
        frame_threshold=int(stream_config["frame_threshold"]),
        audio_max_len=float(stream_config["audio_max_seconds"]),
        audio_min_len=0.0,
        segment_length=float(stream_config["minimum_chunk_seconds"]),
        beams=int(stream_config["beam_size"]),
        task=stream_config["task"],
        decoder_type=stream_config["decoder"],
        never_fire=False,
        init_prompt=None,
        static_init_prompt=None,
        max_context_tokens=None,
        logdir=None,
    )
    online = SimulWhisperOnline(asr)

    first_audio = args.root / rows[0]["tts_audio_16k_mono"]
    warmup_audio, warmup_sr = sf.read(first_audio, dtype="float32", always_2d=False)
    if warmup_sr != 16000:
        raise ValueError(f"Warmup audio must be 16 kHz, got {warmup_sr}")
    asr.warmup(np.asarray(warmup_audio[:16000], dtype=np.float32))

    run_started = perf_counter()
    attempted = 0
    completed = 0
    failed = 0
    for index, row in enumerate(rows, 1):
        if row["id"] in completed_ids:
            print(f"[simul] {index}/{len(rows)} {row['id']} skipped", flush=True)
            continue
        attempted += 1
        result = dict(row)
        sample_started = perf_counter()
        events: list[dict[str, Any]] = []
        committed_parts: list[str] = []
        cumulative_processing = 0.0
        try:
            audio_path = args.root / row["tts_audio_16k_mono"]
            audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
            if sample_rate != 16000:
                raise ValueError(f"Expected 16 kHz audio, got {sample_rate}")
            if audio.ndim != 1:
                raise ValueError(f"Expected mono audio, got shape {audio.shape}")
            audio = np.asarray(audio, dtype=np.float32)
            duration = len(audio) / float(sample_rate)
            chunk_samples = round(
                float(stream_config["minimum_chunk_seconds"]) * sample_rate
            )
            online.init(0)
            for begin in range(0, len(audio), chunk_samples):
                chunk = audio[begin : begin + chunk_samples]
                audio_read = min((begin + len(chunk)) / sample_rate, duration)
                online.insert_audio_chunk(chunk)
                inference_started = perf_counter()
                output = online.process_iter()
                processing = perf_counter() - inference_started
                cumulative_processing += processing
                if output:
                    committed_parts.append(output["text"])
                    events.append(
                        serialize_event(
                            output,
                            "stream",
                            audio_read,
                            processing,
                            cumulative_processing,
                        )
                    )

            inference_started = perf_counter()
            output = online.finish()
            processing = perf_counter() - inference_started
            cumulative_processing += processing
            if output:
                committed_parts.append(output["text"])
                events.append(
                    serialize_event(
                        output,
                        "finish",
                        duration,
                        processing,
                        cumulative_processing,
                    )
                )

            hypothesis = clean_text(committed_parts)
            if not hypothesis:
                raise RuntimeError("SimulStreaming emitted no committed English text")
            result.update(
                {
                    "streaming_asr_status": "completed",
                    "streaming_asr_error": None,
                    "streaming_asr_model": "Whisper large-v3",
                    "streaming_asr_policy": "AlignAtt",
                    "streaming_asr_model_sha256": actual_hash,
                    "streaming_asr_hypothesis_en": hypothesis,
                    "streaming_asr_commit_events": events,
                    "streaming_asr_commit_event_count": len(events),
                    "audio_duration_seconds": round(duration, 6),
                    "first_commit_audio_seconds": (
                        events[0]["audio_read_seconds"] if events else None
                    ),
                    "last_commit_audio_seconds": (
                        events[-1]["audio_read_seconds"] if events else None
                    ),
                    "streaming_asr_processing_seconds": round(
                        cumulative_processing, 6
                    ),
                    "streaming_asr_real_time_factor": round(
                        cumulative_processing / duration, 6
                    ),
                    "streaming_asr_sample_wall_seconds": round(
                        perf_counter() - sample_started, 6
                    ),
                }
            )
            completed += 1
            status = "completed"
        except Exception as exc:  # Keep failures auditable and resumable.
            result.update(
                {
                    "streaming_asr_status": "failed",
                    "streaming_asr_error": repr(exc),
                    "streaming_asr_hypothesis_en": None,
                    "streaming_asr_commit_events": events,
                }
            )
            failed += 1
            status = "failed"
            try:
                online.init(0)
            except Exception:
                pass
        result["streaming_asr_generated_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        append_jsonl(args.output_jsonl, result)
        print(f"[simul] {index}/{len(rows)} {row['id']} {status}", flush=True)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "input_manifest": str(args.manifest),
        "output_jsonl": str(args.output_jsonl),
        "simulstreaming_revision": config["simulstreaming"]["code_revision"],
        "model_sha256": actual_hash,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "requested_record_count": len(rows),
        "attempted_count": attempted,
        "completed_count": completed,
        "failed_count": failed,
        "previously_completed_count": len(completed_ids),
        "run_wall_seconds": round(perf_counter() - run_started, 6),
        "ok": failed == 0 and completed + len(completed_ids) >= len(rows),
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
