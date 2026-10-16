#!/usr/bin/env python3
"""Build the locked 100-sample Task 5 text and TTS manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_english(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--legacy-tts-manifest", type=Path, required=True)
    parser.add_argument("--qwen-candidate-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    config = read_json(args.config)
    qwen_by_english: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(args.qwen_candidate_manifest):
        key = row.get("normalized_english_key") or normalize_english(row["source_text_en"])
        qwen_by_english.setdefault(key, row)

    speakers = config["tts"]["speakers"]
    legacy_rows = list(read_jsonl(args.legacy_tts_manifest))[: args.limit]
    output_rows: list[dict[str, Any]] = []
    missing_qwen: list[str] = []
    for index, source in enumerate(legacy_rows):
        english = source["bridge_text"]
        qwen = qwen_by_english.get(normalize_english(english))
        if qwen is None:
            missing_qwen.append(source["id"])
        speaker = speakers[index % len(speakers)]
        basename = f"{source['id']}__{speaker}"
        output_rows.append(
            {
                "id": source["id"],
                "pilot_index": index,
                "source_language": "zh",
                "source_text_zh": source["source_text_zh"],
                "reference_language": "en",
                "reference_text_en": english,
                "reference_text_source": "How2Sign English Translation manually realigned",
                "reference_pseudo_asl_gloss": (
                    qwen["target_pseudo_asl_gloss"] if qwen else None
                ),
                "reference_pseudo_asl_gloss_is_gold": False,
                "qwen_gloss_request_id": qwen.get("request_id") if qwen else None,
                "how2sign_sentence_name": source.get("how2sign_sentence_name"),
                "how2sign_video_id": source.get("how2sign_video_id"),
                "tts_model_id": config["tts"]["model_id"],
                "tts_model_revision": config["tts"]["model_revision"],
                "tts_speaker": speaker,
                "tts_seed": int(config["seed"]) + index,
                "tts_audio_original": f"audio/qwen3_tts_100/original/{basename}.wav",
                "tts_audio_16k_mono": f"audio/qwen3_tts_100/normalized_16k_mono/{basename}.wav",
                "tts_status": "pending",
            }
        )

    duplicate_ids = len(output_rows) - len({row["id"] for row in output_rows})
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": len(output_rows),
        "speaker_counts": {
            speaker: sum(row["tts_speaker"] == speaker for row in output_rows)
            for speaker in speakers
        },
        "duplicate_id_count": duplicate_ids,
        "missing_qwen_reference_count": len(missing_qwen),
        "missing_qwen_reference_ids": missing_qwen,
        "input_sha256": {
            "config": sha256(args.config),
            "legacy_tts_manifest": sha256(args.legacy_tts_manifest),
            "qwen_candidate_manifest": sha256(args.qwen_candidate_manifest),
        },
        "ok": duplicate_ids == 0 and not missing_qwen and len(output_rows) == args.limit,
    }
    write_jsonl(args.output, output_rows)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

