#!/usr/bin/env python3
"""Run pinned AlignAtt/Whisper as streaming Mandarin ASR."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import soundfile as sf
import torch

from run_simulstreaming_batch import (
    append_jsonl,
    clean_text,
    read_json,
    read_jsonl,
    serialize_event,
    sha256,
)


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

    actual_hash = sha256(args.model_path)
    expected_hash = stream_config["whisper_checkpoint_sha256"]
    if actual_hash != expected_hash:
        raise ValueError(f"Whisper SHA256 mismatch: {actual_hash} != {expected_hash}")

    asr = SimulWhisperASR(
        language=stream_config["source_language"],
        model_path=str(args.model_path),
        cif_ckpt_path=None,
        frame_threshold=int(stream_config["frame_threshold"]),
        audio_max_len=float(stream_config["audio_max_seconds"]),
        audio_min_len=0.0,
        segment_length=float(stream_config["minimum_chunk_seconds"]),
        beams=int(stream_config["beam_size"]),
        task="transcribe",
        decoder_type=stream_config["decoder"],
        never_fire=False,
        init_prompt=None,
        static_init_prompt=None,
        max_context_tokens=None,
        logdir=None,
    )
    online = SimulWhisperOnline(asr)
    warmup_audio, warmup_sr = sf.read(
        args.root / rows[0]["tts_audio_16k_mono"], dtype="float32"
    )
    if warmup_sr != 16000 or warmup_audio.ndim != 1:
        raise ValueError("Warmup audio must be 16 kHz mono")
    asr.warmup(np.asarray(warmup_audio[:16000], dtype=np.float32))

    completed = 0
    failed = 0
    run_started = perf_counter()
    for index, row in enumerate(rows, 1):
        result = dict(row)
        events: list[dict[str, Any]] = []
        parts: list[str] = []
        cumulative_processing = 0.0
        try:
            audio, sample_rate = sf.read(
                args.root / row["tts_audio_16k_mono"], dtype="float32"
            )
            if sample_rate != 16000 or audio.ndim != 1:
                raise ValueError(
                    f"Expected 16 kHz mono audio, got {sample_rate} Hz {audio.shape}"
                )
            audio = np.asarray(audio, dtype=np.float32)
            duration = len(audio) / sample_rate
            chunk_samples = round(
                float(stream_config["minimum_chunk_seconds"]) * sample_rate
            )
            online.init(0)
            for begin in range(0, len(audio), chunk_samples):
                chunk = audio[begin : begin + chunk_samples]
                audio_read = min((begin + len(chunk)) / sample_rate, duration)
                online.insert_audio_chunk(chunk)
                started = perf_counter()
                output = online.process_iter()
                processing = perf_counter() - started
                cumulative_processing += processing
                if output:
                    parts.append(output["text"])
                    events.append(
                        serialize_event(
                            output,
                            "stream",
                            audio_read,
                            processing,
                            cumulative_processing,
                        )
                    )
            started = perf_counter()
            output = online.finish()
            processing = perf_counter() - started
            cumulative_processing += processing
            if output:
                parts.append(output["text"])
                events.append(
                    serialize_event(
                        output,
                        "finish",
                        duration,
                        processing,
                        cumulative_processing,
                    )
                )
            transcript = clean_text(parts)
            if not transcript:
                raise RuntimeError("AlignAtt emitted no Mandarin transcript")
            result.update(
                {
                    "zh_asr_status": "completed",
                    "zh_asr_error": None,
                    "zh_asr_transcript": transcript,
                    "zh_asr_commit_events": events,
                    "zh_asr_commit_event_count": len(events),
                    "zh_asr_processing_seconds": round(cumulative_processing, 6),
                    "zh_asr_real_time_factor": round(
                        cumulative_processing / duration, 6
                    ),
                    "audio_duration_seconds": round(duration, 6),
                    "zh_asr_model_sha256": actual_hash,
                }
            )
            completed += 1
            status = "completed"
        except Exception as exc:
            result.update(
                {
                    "zh_asr_status": "failed",
                    "zh_asr_error": repr(exc),
                    "zh_asr_transcript": None,
                    "zh_asr_commit_events": events,
                }
            )
            failed += 1
            status = "failed"
            try:
                online.init(0)
            except Exception:
                pass
        result["zh_asr_generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        append_jsonl(args.output_jsonl, result)
        print(f"[zh-asr] {index}/{len(rows)} {row['id']} {status}", flush=True)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(args.manifest),
        "output_jsonl": str(args.output_jsonl),
        "method": "Whisper large-v3 AlignAtt Mandarin transcription",
        "record_count": len(rows),
        "completed_count": completed,
        "failed_count": failed,
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
