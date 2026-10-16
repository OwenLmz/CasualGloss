#!/usr/bin/env python3
"""Generate pinned multi-speaker Mandarin TTS and normalized 16 kHz WAV files."""

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
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from qwen_tts import Qwen3TTSModel

    config = read_json(args.config)
    rows = read_jsonl(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    device = args.device or config["tts"]["device"]
    model = Qwen3TTSModel.from_pretrained(
        args.model_path,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation=config["tts"]["attention_implementation"],
    )

    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        original_path = args.root / row["tts_audio_original"]
        normalized_path = args.root / row["tts_audio_16k_mono"]
        original_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        started = perf_counter()
        try:
            if args.overwrite or not (original_path.exists() and normalized_path.exists()):
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
            else:
                waveform, sample_rate = sf.read(original_path, dtype="float32")
            duration = len(waveform) / float(sample_rate)
            status = "completed"
            error = None
        except Exception as exc:  # Preserve failed rows for audit and retry.
            duration = None
            sample_rate = None
            status = "failed"
            error = repr(exc)
        result = dict(row)
        result.update(
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
        output_rows.append(result)
        write_jsonl(args.output_manifest, output_rows)
        print(f"[tts] {index}/{len(rows)} {row['id']} {status}", flush=True)

    failures = sum(row["tts_status"] != "completed" for row in output_rows)
    print(json.dumps({"record_count": len(output_rows), "failed_count": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
