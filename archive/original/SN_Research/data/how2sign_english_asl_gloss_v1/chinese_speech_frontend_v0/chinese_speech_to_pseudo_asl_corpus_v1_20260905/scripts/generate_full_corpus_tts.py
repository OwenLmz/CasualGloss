#!/usr/bin/env python3
"""Generate checkpointed multi-speaker Mandarin TTS for one corpus shard."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def paths_exist(root: Path, row: dict[str, Any]) -> bool:
    return (root / row["tts_audio_original"]).exists() and (
        root / row["tts_audio_16k_mono"]
    ).exists()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from qwen_tts import Qwen3TTSModel

    config = read_json(args.config)
    originals = read_jsonl(args.manifest)
    if args.limit is not None:
        originals = originals[: args.limit]
    previous = {} if args.overwrite else {
        row["id"]: row for row in read_jsonl(args.output_manifest)
    }
    rows = [dict(row) for row in originals]
    pending: list[int] = []
    reused = 0
    for index, original in enumerate(originals):
        old = previous.get(original["id"])
        if old and old.get("tts_status") == "completed" and paths_exist(args.root, old):
            rows[index] = old
            reused += 1
        elif original.get("tts_status") == "completed" and paths_exist(args.root, original):
            reused += 1
        else:
            pending.append(index)

    model = Qwen3TTSModel.from_pretrained(
        args.model_path,
        device_map=args.device,
        dtype=torch.bfloat16,
        attn_implementation=config["tts"]["attention_implementation"],
    )

    generated_count = 0
    started_run = perf_counter()
    for pending_position, index in enumerate(pending, 1):
        row = rows[index]
        original_path = args.root / row["tts_audio_original"]
        normalized_path = args.root / row["tts_audio_16k_mono"]
        original_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        started = perf_counter()
        try:
            if row.get("source_text_zh_status") != "completed" or not row.get("source_text_zh"):
                raise ValueError("Mandarin translation is not completed")
            if args.overwrite or not paths_exist(args.root, row):
                set_seed(int(row["tts_seed"]))
                wavs, sample_rate = model.generate_custom_voice(
                    text=row["source_text_zh"],
                    language=config["tts"]["language"],
                    speaker=row["tts_speaker"],
                    instruct="",
                )
                waveform = np.asarray(wavs[0], dtype=np.float32).squeeze()
                sf.write(original_path, waveform, sample_rate, subtype="PCM_16")
                tensor = torch.from_numpy(waveform).float().reshape(1, -1)
                normalized = AF.resample(
                    tensor, int(sample_rate), int(config["tts"]["normalized_sample_rate_hz"])
                ).squeeze(0).cpu().numpy()
                sf.write(
                    normalized_path,
                    normalized,
                    int(config["tts"]["normalized_sample_rate_hz"]),
                    subtype="PCM_16",
                )
                generated_count += 1
            else:
                info = sf.info(original_path)
                sample_rate = info.samplerate
                waveform = np.empty(info.frames, dtype=np.float32)
            duration = len(waveform) / float(sample_rate)
            status = "completed"
            error = None
        except Exception as exc:
            duration = None
            sample_rate = None
            status = "failed"
            error = repr(exc)
        row.update(
            {
                "tts_status": status,
                "tts_error": error,
                "tts_original_sample_rate_hz": sample_rate,
                "tts_duration_seconds": round(duration, 6) if duration is not None else None,
                "tts_inference_seconds": round(perf_counter() - started, 6),
                "tts_generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "tts_torch_version": torch.__version__,
            }
        )
        if pending_position % args.checkpoint_every == 0 or pending_position == len(pending):
            write_jsonl_atomic(args.output_manifest, rows)
        print(f"[tts] generated {pending_position}/{len(pending)} {row['id']} {status}", flush=True)

    write_jsonl_atomic(args.output_manifest, rows)
    completed = sum(row.get("tts_status") == "completed" for row in rows)
    failed = len(rows) - completed
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "full_synthetic_mandarin_tts_shard",
        "record_count": len(rows),
        "completed_count": completed,
        "failed_count": failed,
        "generated_count": generated_count,
        "reused_count": reused,
        "model": config["tts"]["model_id"],
        "model_revision": config["tts"]["model_revision"],
        "device": args.device,
        "run_wall_seconds": round(perf_counter() - started_run, 6),
        "failed_rows": [
            {"id": row["id"], "error": row.get("tts_error")}
            for row in rows
            if row.get("tts_status") != "completed"
        ],
        "ok": completed == len(rows),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
