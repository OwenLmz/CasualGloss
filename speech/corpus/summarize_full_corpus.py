#!/usr/bin/env python3
"""Audit and summarize the completed synthetic speech-to-pseudo-gloss corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": round(mean(values), 8) if values else None,
        "median": round(median(values), 8) if values else None,
        "min": round(min(values), 8) if values else None,
        "max": round(max(values), 8) if values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audio-validation", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset-card", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    audio_validation = json.loads(args.audio_validation.read_text(encoding="utf-8-sig"))
    durations = [float(row["tts_duration_seconds"]) for row in rows if row.get("tts_duration_seconds")]
    missing_audio = [
        row["id"]
        for row in rows
        if not (args.root / row["tts_audio_16k_mono"]).exists()
    ]
    video_ids = {
        video_id
        for row in rows
        for video_id in (row.get("how2sign_video_ids") or [])
        if video_id
    }
    split_counts = Counter(row["split"] for row in rows)
    speaker_counts = Counter(row["tts_speaker"] for row in rows)
    translation_counts = Counter(row.get("source_text_zh_status", "missing") for row in rows)
    tts_counts = Counter(row.get("tts_status", "missing") for row in rows)
    total_seconds = sum(durations)
    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_name": config["corpus_name"],
        "corpus_version": config["corpus_version"],
        "scientific_status": config["scientific_status"],
        "record_count": len(rows),
        "expected_count": args.expected_count,
        "unique_id_count": len({row["id"] for row in rows}),
        "unique_how2sign_video_count": len(video_ids),
        "split_counts": dict(sorted(split_counts.items())),
        "speaker_counts": dict(sorted(speaker_counts.items())),
        "translation_status_counts": dict(sorted(translation_counts.items())),
        "tts_status_counts": dict(sorted(tts_counts.items())),
        "total_duration_seconds": round(total_seconds, 6),
        "total_duration_hours": round(total_seconds / 3600.0, 6),
        "duration_seconds": stats(durations),
        "mean_english_whitespace_tokens": round(
            mean(len(str(row["reference_text_en"]).split()) for row in rows), 6
        ),
        "mean_pseudo_gloss_tokens": round(
            mean(len(str(row["reference_pseudo_asl_gloss"]).split()) for row in rows), 6
        ),
        "missing_audio_count": len(missing_audio),
        "missing_audio_ids": missing_audio,
        "audio_validation_ok": bool(audio_validation.get("ok")),
        "manifest_sha256": sha256(args.manifest),
        "config_sha256": sha256(args.config),
        "limitations": config["limitations"],
    }
    report["ok"] = (
        len(rows) == args.expected_count
        and report["unique_id_count"] == args.expected_count
        and translation_counts.get("completed", 0) == args.expected_count
        and tts_counts.get("completed", 0) == args.expected_count
        and not missing_audio
        and report["audio_validation_ok"]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    card = f"""# {config['corpus_name']}

## Scope

- Version: {config['corpus_version']}
- Utterances: {len(rows):,}
- Duration: {total_seconds / 3600.0:.2f} hours
- Speakers: {len(speaker_counts)} synthetic Mandarin voices
- Splits: {dict(sorted(split_counts.items()))}
- Target: weak pseudo-ASL gloss, not expert gold ASL

## Construction

The source text is derived from manually realigned How2Sign English translations. English
sentences are deterministically translated into spoken Mandarin with the pinned Qwen3-4B
model, then synthesized with four pinned Qwen3-TTS voices. Audio is retained at the original
sample rate and normalized to mono 16 kHz. The frozen 500-item test set is reused unchanged.

## Intended use

This corpus supports reproducible research on Chinese-speech-to-ASL-pseudo-gloss streaming
systems. It does not measure expert ASL translation quality and should not be represented as
a gold ASL corpus.

## Limitations

- Mandarin speech is synthetic and does not represent real-speaker or noisy conditions.
- ASL targets are LLM-generated weak pseudo labels.
- Redistribution remains subject to the licenses of the underlying source and model assets.
"""
    args.dataset_card.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_card.write_text(card, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
