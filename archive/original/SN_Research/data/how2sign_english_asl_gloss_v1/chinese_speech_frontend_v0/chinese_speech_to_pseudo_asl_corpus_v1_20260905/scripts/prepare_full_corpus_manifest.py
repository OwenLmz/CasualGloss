#!/usr/bin/env python3
"""Build a fixed full-corpus manifest without changing the locked test set."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
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
    parser.add_argument("--parents", type=Path, required=True)
    parser.add_argument("--all-records", type=Path, required=True)
    parser.add_argument("--locked-test-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--audio-subdir", required=True)
    args = parser.parse_args()

    parents = read_jsonl(args.parents)
    all_records = read_jsonl(args.all_records)
    locked_rows = read_jsonl(args.locked_test_manifest)
    config = read_json(args.config)

    metadata_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_records:
        metadata_by_request[str(row["request_id"])].append(row)
    locked_by_id = {str(row["id"]): row for row in locked_rows}

    parent_test_ids = {str(row["id"]) for row in parents if row.get("split") == "test"}
    locked_ids = set(locked_by_id)
    locked_set_matches = parent_test_ids == locked_ids

    speakers = list(config["tts"]["speakers"])
    split_index: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    missing_metadata: list[str] = []
    source_mismatches: list[str] = []
    locked_reused = 0

    for corpus_index, parent in enumerate(parents):
        identifier = str(parent["id"])
        split = str(parent["split"])
        matches = metadata_by_request.get(identifier, [])
        exact = [
            row
            for row in matches
            if str(row.get("source_text_en", "")).strip()
            == str(parent["source_text_en"]).strip()
        ]
        if not matches:
            missing_metadata.append(identifier)
            metadata: dict[str, Any] = {}
        else:
            metadata = exact[0] if exact else matches[0]
            if not exact:
                source_mismatches.append(identifier)

        locked = locked_by_id.get(identifier)
        if split == "test" and locked is not None:
            speaker = str(locked["tts_speaker"])
            tts_seed = int(locked["tts_seed"])
            original_path = str(locked["tts_audio_original"])
            normalized_path = str(locked["tts_audio_16k_mono"])
            source_text_zh = locked.get("source_text_zh")
            source_text_zh_status = locked.get("source_text_zh_status", "pending")
            tts_status = locked.get("tts_status", "pending")
            locked_reused += 1
        else:
            local_index = split_index[split]
            speaker = speakers[local_index % len(speakers)]
            tts_seed = int(config["seed"]) + corpus_index
            basename = f"{corpus_index:05d}_{identifier}__{speaker}.wav"
            original_path = f"audio/{args.audio_subdir}/original/{split}/{basename}"
            normalized_path = f"audio/{args.audio_subdir}/normalized_16k_mono/{split}/{basename}"
            source_text_zh = None
            source_text_zh_status = "pending"
            tts_status = "pending"
        split_index[split] += 1

        row = {
            "id": identifier,
            "corpus_index": corpus_index,
            "corpus_name": config["corpus_name"],
            "corpus_version": config["corpus_version"],
            "split": split,
            "locked_evaluation_parent": bool(parent.get("locked_evaluation_parent", False)),
            "source_language": "zh",
            "source_text_zh": source_text_zh,
            "source_text_zh_status": source_text_zh_status,
            "reference_language": "en",
            "reference_text_en": parent["source_text_en"],
            "reference_text_source": "How2Sign English Translation manually realigned",
            "reference_pseudo_asl_gloss": parent["target_pseudo_asl_gloss"],
            "reference_pseudo_asl_gloss_is_gold": False,
            "qwen_gloss_request_id": identifier,
            "pseudo_gloss_generator": "Qwen/Qwen2.5-7B-Instruct",
            "how2sign_sentence_name": metadata.get("how2sign_sentence_name"),
            "how2sign_video_id": metadata.get("how2sign_video_id"),
            "how2sign_video_ids": parent.get("how2sign_video_ids", []),
            "how2sign_source_row_index": metadata.get("source_row_index"),
            "candidate_selection_bucket": parent.get("candidate_selection_bucket"),
            "candidate_source_length_bin": parent.get("candidate_source_length_bin"),
            "tts_model_id": config["tts"]["model_id"],
            "tts_model_revision": config["tts"]["model_revision"],
            "tts_speaker": speaker,
            "tts_seed": tts_seed,
            "tts_audio_original": original_path,
            "tts_audio_16k_mono": normalized_path,
            "tts_status": tts_status,
            "reuses_locked500_audio": bool(split == "test" and locked is not None),
        }
        if locked is not None:
            for key in (
                "source_text_zh_quality_flags",
                "source_text_zh_error",
                "source_text_zh_model",
                "source_text_zh_model_revision",
                "source_text_zh_decoding",
                "source_text_zh_processing_seconds",
                "source_text_zh_generated_at_utc",
                "tts_error",
                "tts_original_sample_rate_hz",
                "tts_duration_seconds",
                "tts_inference_seconds",
                "tts_generated_at_utc",
                "tts_torch_version",
            ):
                if key in locked:
                    row[key] = locked[key]
        rows.append(row)

    duplicate_ids = len(rows) - len({row["id"] for row in rows})
    split_counts = Counter(str(row["split"]) for row in rows)
    speaker_counts = Counter(str(row["tts_speaker"]) for row in rows)
    split_speaker_counts = {
        split: dict(sorted(Counter(r["tts_speaker"] for r in rows if r["split"] == split).items()))
        for split in sorted(split_counts)
    }
    expected = int(config["expected_count"])
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": config["scientific_status"],
        "corpus_name": config["corpus_name"],
        "corpus_version": config["corpus_version"],
        "record_count": len(rows),
        "expected_count": expected,
        "split_counts": dict(sorted(split_counts.items())),
        "speaker_counts": dict(sorted(speaker_counts.items())),
        "split_speaker_counts": split_speaker_counts,
        "duplicate_id_count": duplicate_ids,
        "missing_metadata_count": len(missing_metadata),
        "missing_metadata_ids": missing_metadata,
        "source_mismatch_count": len(source_mismatches),
        "source_mismatch_ids": source_mismatches,
        "locked_test_count": len(locked_rows),
        "locked_test_reused_count": locked_reused,
        "locked_test_id_set_matches_parent_test": locked_set_matches,
        "input_sha256": {
            "parents": sha256(args.parents),
            "all_records": sha256(args.all_records),
            "locked_test_manifest": sha256(args.locked_test_manifest),
            "config": sha256(args.config),
        },
        "selection_rule": "all usable parent records from the frozen causal-alignment split",
        "locked_test_used_for_model_or_policy_selection": False,
        "ok": (
            len(rows) == expected
            and duplicate_ids == 0
            and not missing_metadata
            and not source_mismatches
            and locked_set_matches
            and locked_reused == len(locked_rows) == 500
        ),
    }
    write_jsonl(args.output, rows)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
