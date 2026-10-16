#!/usr/bin/env python3
"""Build a fixed synthetic-Mandarin manifest from the paper locked test IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locked-rollouts", type=Path, required=True)
    parser.add_argument("--all-records", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=500)
    parser.add_argument("--audio-subdir", default="paper_e2e_locked500_20260904")
    args = parser.parse_args()

    locked = read_jsonl(args.locked_rollouts)
    all_records = read_jsonl(args.all_records)
    config = read_json(args.config)
    records_by_request: dict[str, list[dict[str, Any]]] = {}
    for row in all_records:
        records_by_request.setdefault(str(row["request_id"]), []).append(row)

    speakers = list(config["tts"]["speakers"])
    rows: list[dict[str, Any]] = []
    missing_records: list[str] = []
    source_mismatches: list[str] = []
    for index, item in enumerate(locked):
        request_id = str(item["id"])
        matches = records_by_request.get(request_id, [])
        if not matches:
            missing_records.append(request_id)
            metadata: dict[str, Any] = {}
        else:
            exact = [
                row for row in matches
                if str(row.get("source_text_en", "")).strip() == str(item["source_text_en"]).strip()
            ]
            metadata = exact[0] if exact else matches[0]
            if not exact:
                source_mismatches.append(request_id)

        speaker = speakers[index % len(speakers)]
        basename = f"{index:04d}_{request_id}__{speaker}"
        rows.append(
            {
                "id": request_id,
                "locked_index": index,
                "paper_split": "locked_test_500",
                "source_language": "zh",
                "source_text_zh": None,
                "source_text_zh_status": "pending",
                "reference_language": "en",
                "reference_text_en": item["source_text_en"],
                "reference_text_source": "How2Sign English Translation manually realigned",
                "reference_pseudo_asl_gloss": item["target_pseudo_asl_gloss"],
                "reference_pseudo_asl_gloss_is_gold": False,
                "qwen_gloss_request_id": request_id,
                "how2sign_sentence_name": metadata.get("how2sign_sentence_name"),
                "how2sign_video_id": metadata.get("how2sign_video_id"),
                "how2sign_source_row_index": metadata.get("source_row_index"),
                "tts_model_id": config["tts"]["model_id"],
                "tts_model_revision": config["tts"]["model_revision"],
                "tts_speaker": speaker,
                "tts_seed": int(config["seed"]) + index,
                "tts_audio_original": (
                    f"audio/{args.audio_subdir}/original/{basename}.wav"
                ),
                "tts_audio_16k_mono": (
                    f"audio/{args.audio_subdir}/normalized_16k_mono/{basename}.wav"
                ),
                "tts_status": "pending",
            }
        )

    duplicate_ids = len(rows) - len({row["id"] for row in rows})
    speaker_counts = Counter(row["tts_speaker"] for row in rows)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "fixed_locked_test_synthetic_mandarin_manifest",
        "record_count": len(rows),
        "expected_count": args.expected_count,
        "duplicate_id_count": duplicate_ids,
        "missing_metadata_count": len(missing_records),
        "missing_metadata_ids": missing_records,
        "source_mismatch_count": len(source_mismatches),
        "source_mismatch_ids": source_mismatches,
        "speaker_counts": dict(sorted(speaker_counts.items())),
        "input_sha256": {
            "locked_rollouts": sha256(args.locked_rollouts),
            "all_records": sha256(args.all_records),
            "config": sha256(args.config),
        },
        "selection_rule": "all IDs in the pre-existing locked500 v2.3 rollout, original order",
        "locked_test_used_for_model_or_policy_selection": False,
        "ok": (
            len(rows) == args.expected_count
            and duplicate_ids == 0
            and not missing_records
            and not source_mismatches
        ),
    }
    write_jsonl(args.output, rows)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
