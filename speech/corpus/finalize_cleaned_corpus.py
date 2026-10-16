#!/usr/bin/env python3
"""Freeze a cleaned corpus manifest by excluding pre-flagged acoustic outliers."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": round(mean(values), 8) if values else None,
        "median": round(median(values), 8) if values else None,
        "min": round(min(values), 8) if values else None,
        "max": round(max(values), 8) if values else None,
    }


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--raw-audio-validation", type=Path, required=True)
    parser.add_argument("--raw-config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-raw-count", type=int, default=19947)
    parser.add_argument("--expected-exclusion-count", type=int, default=23)
    args = parser.parse_args()

    raw_lines = [
        line
        for line in args.raw_manifest.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    rows = [json.loads(line) for line in raw_lines]
    raw_validation = read_json(args.raw_audio_validation)
    raw_config = read_json(args.raw_config)
    issues = raw_validation.get("issues", [])
    issue_by_id = {str(item["id"]): str(item["issue"]) for item in issues}

    if len(rows) != args.expected_raw_count:
        raise SystemExit(
            f"Raw row count mismatch: {len(rows)} != {args.expected_raw_count}"
        )
    if len(issue_by_id) != args.expected_exclusion_count:
        raise SystemExit(
            "Exclusion count mismatch: "
            f"{len(issue_by_id)} != {args.expected_exclusion_count}"
        )
    if len(issue_by_id) != len(issues):
        raise SystemExit("Audio validation contains duplicate issue IDs")

    row_by_id = {str(row["id"]): row for row in rows}
    if len(row_by_id) != len(rows):
        raise SystemExit("Raw manifest contains duplicate IDs")
    unknown_issue_ids = sorted(set(issue_by_id) - set(row_by_id))
    if unknown_issue_ids:
        raise SystemExit(f"Unknown issue IDs: {unknown_issue_ids}")

    excluded_rows = sorted(
        (row_by_id[item_id] for item_id in issue_by_id),
        key=lambda row: int(row["corpus_index"]),
    )
    locked_exclusions = [
        row["id"] for row in excluded_rows if row.get("locked_evaluation_parent")
    ]
    non_train_exclusions = [
        row["id"] for row in excluded_rows if row.get("split") != "train"
    ]
    if locked_exclusions or non_train_exclusions:
        raise SystemExit(
            "Refusing to alter validation/test data: "
            f"locked={locked_exclusions}, non_train={non_train_exclusions}"
        )

    clean_lines = [
        line for line, row in zip(raw_lines, rows) if str(row["id"]) not in issue_by_id
    ]
    clean_rows = [row for row in rows if str(row["id"]) not in issue_by_id]
    expected_clean_count = args.expected_raw_count - args.expected_exclusion_count
    if len(clean_rows) != expected_clean_count:
        raise SystemExit(
            f"Clean row count mismatch: {len(clean_rows)} != {expected_clean_count}"
        )

    manifests_dir = args.output_dir / "manifests"
    reports_dir = args.output_dir / "reports"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    clean_manifest = manifests_dir / "full.cleaned.jsonl"
    clean_manifest.write_text("\n".join(clean_lines) + "\n", encoding="utf-8")

    created_at = datetime.now(timezone.utc).isoformat()
    exclusions = {
        "created_at_utc": created_at,
        "policy": (
            "Exclude every training item flagged by the frozen pre-evaluation "
            "acoustic integrity audit; preserve all raw records and audio assets."
        ),
        "selection_uses_model_results": False,
        "raw_manifest": str(args.raw_manifest),
        "raw_manifest_sha256": sha256(args.raw_manifest),
        "raw_audio_validation": str(args.raw_audio_validation),
        "raw_audio_validation_sha256": sha256(args.raw_audio_validation),
        "raw_count": len(rows),
        "excluded_count": len(excluded_rows),
        "clean_count": len(clean_rows),
        "issue_counts": dict(sorted(Counter(issue_by_id.values()).items())),
        "split_counts": dict(
            sorted(Counter(str(row["split"]) for row in excluded_rows).items())
        ),
        "locked_evaluation_exclusion_count": len(locked_exclusions),
        "items": [
            {
                "id": row["id"],
                "corpus_index": row["corpus_index"],
                "split": row["split"],
                "issue": issue_by_id[str(row["id"])],
                "duration_seconds": row.get("tts_duration_seconds"),
                "tts_audio_original": row.get("tts_audio_original"),
                "tts_audio_16k_mono": row.get("tts_audio_16k_mono"),
            }
            for row in excluded_rows
        ],
    }
    exclusions_path = reports_dir / "audio_exclusions.cleaned.json"
    dump_json(exclusions_path, exclusions)

    missing_audio = [
        row["id"]
        for row in clean_rows
        if not (args.root / str(row["tts_audio_16k_mono"])).exists()
    ]
    durations = [float(row["tts_duration_seconds"]) for row in clean_rows]
    split_counts = Counter(str(row["split"]) for row in clean_rows)
    speaker_counts = Counter(str(row["tts_speaker"]) for row in clean_rows)
    translation_counts = Counter(
        str(row.get("source_text_zh_status", "missing")) for row in clean_rows
    )
    tts_counts = Counter(str(row.get("tts_status", "missing")) for row in clean_rows)
    video_ids = {
        str(video_id)
        for row in clean_rows
        for video_id in (row.get("how2sign_video_ids") or [])
        if video_id
    }
    total_seconds = sum(durations)

    clean_validation = {
        "created_at_utc": created_at,
        "record_count": len(clean_rows),
        "status_counts": {"valid": len(clean_rows), "flagged": 0},
        "excluded_by_frozen_raw_audit": len(excluded_rows),
        "missing_audio_count": len(missing_audio),
        "missing_audio_ids": missing_audio,
        "duration_seconds": numeric_stats(durations),
        "source_audio_validation_sha256": sha256(args.raw_audio_validation),
        "ok": not missing_audio,
        "note": (
            "All acoustic proxy flags from the frozen raw audit were excluded. "
            "This verifies engineering integrity, not human-perceived naturalness."
        ),
    }
    clean_validation_path = reports_dir / "audio_validation.cleaned.json"
    dump_json(clean_validation_path, clean_validation)

    clean_config = dict(raw_config)
    clean_config["corpus_version"] = "2026-09-06-cleaned-v1"
    clean_config["expected_count"] = expected_clean_count
    clean_config["parent_corpus_version"] = raw_config["corpus_version"]
    clean_config["quality_control"] = {
        "raw_count": len(rows),
        "excluded_count": len(excluded_rows),
        "excluded_fraction": round(len(excluded_rows) / len(rows), 8),
        "policy": exclusions["policy"],
        "validation_and_test_unchanged": True,
    }
    clean_config_path = args.output_dir / "config_cleaned_corpus.json"
    dump_json(clean_config_path, clean_config)

    summary = {
        "created_at_utc": created_at,
        "corpus_name": clean_config["corpus_name"],
        "corpus_version": clean_config["corpus_version"],
        "parent_corpus_version": raw_config["corpus_version"],
        "scientific_status": clean_config["scientific_status"],
        "raw_record_count": len(rows),
        "excluded_record_count": len(excluded_rows),
        "excluded_fraction": round(len(excluded_rows) / len(rows), 8),
        "record_count": len(clean_rows),
        "expected_count": expected_clean_count,
        "unique_id_count": len({str(row["id"]) for row in clean_rows}),
        "unique_how2sign_video_count": len(video_ids),
        "split_counts": dict(sorted(split_counts.items())),
        "speaker_counts": dict(sorted(speaker_counts.items())),
        "translation_status_counts": dict(sorted(translation_counts.items())),
        "tts_status_counts": dict(sorted(tts_counts.items())),
        "total_duration_seconds": round(total_seconds, 6),
        "total_duration_hours": round(total_seconds / 3600.0, 6),
        "duration_seconds": numeric_stats(durations),
        "mean_english_whitespace_tokens": round(
            mean(len(str(row["reference_text_en"]).split()) for row in clean_rows), 6
        ),
        "mean_pseudo_gloss_tokens": round(
            mean(
                len(str(row["reference_pseudo_asl_gloss"]).split())
                for row in clean_rows
            ),
            6,
        ),
        "missing_audio_count": len(missing_audio),
        "missing_audio_ids": missing_audio,
        "validation_and_test_unchanged": True,
        "clean_manifest_sha256": sha256(clean_manifest),
        "raw_manifest_sha256": sha256(args.raw_manifest),
        "clean_config_sha256": sha256(clean_config_path),
        "exclusions_sha256": sha256(exclusions_path),
        "limitations": clean_config["limitations"],
    }
    summary["ok"] = (
        len(clean_rows) == expected_clean_count
        and summary["unique_id_count"] == expected_clean_count
        and split_counts == Counter({"train": 18924, "validation": 500, "test": 500})
        and translation_counts.get("completed", 0) == expected_clean_count
        and tts_counts.get("completed", 0) == expected_clean_count
        and not missing_audio
        and clean_validation["ok"]
    )
    summary_path = reports_dir / "corpus_summary.cleaned.json"
    dump_json(summary_path, summary)

    card = f"""# {clean_config['corpus_name']} (cleaned release)

## Scope

- Version: {clean_config['corpus_version']}
- Utterances: {len(clean_rows):,}
- Duration: {total_seconds / 3600.0:.2f} hours
- Speakers: {len(speaker_counts)} synthetic Mandarin voices
- Splits: {dict(sorted(split_counts.items()))}
- Target: weak pseudo-ASL gloss, not expert gold ASL

## Construction

The source text is derived from manually realigned How2Sign English translations. English
sentences are deterministically translated into spoken Mandarin with the pinned Qwen3-4B
model, then synthesized with four pinned Qwen3-TTS voices. Audio is retained at the original
sample rate and normalized to mono 16 kHz. The frozen 500-item validation and 500-item test
sets are unchanged.

## Quality control

The raw corpus contained {len(rows):,} records. A frozen, model-result-independent acoustic
audit flagged {len(excluded_rows)} training records ({len(excluded_rows) / len(rows):.2%}) for
duration, silence-ratio, or possible-clipping anomalies. Those records remain in the raw
release but are excluded from this cleaned manifest. The exclusion IDs and reasons are stored
in `reports/audio_exclusions.cleaned.json`.

## Intended use

This corpus supports reproducible research on Chinese-speech-to-ASL-pseudo-gloss streaming
systems. It does not measure expert ASL translation quality and must not be represented as a
gold ASL corpus.

## Limitations

- Mandarin speech is synthetic and does not represent real-speaker or noisy conditions.
- ASL targets are LLM-generated weak pseudo labels rather than expert annotations.
- Acoustic proxy checks do not establish human-perceived speech naturalness.
- Redistribution remains subject to the licenses of the underlying source and model assets.
"""
    card_path = args.output_dir / "DATASET_CARD.cleaned.md"
    card_path.write_text(card, encoding="utf-8")

    hash_targets = [
        clean_manifest,
        clean_config_path,
        exclusions_path,
        clean_validation_path,
        summary_path,
        card_path,
    ]
    hashes_path = reports_dir / "core_sha256sums.cleaned.txt"
    hashes_path.write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(args.output_dir).as_posix()}\n"
            for path in hash_targets
        ),
        encoding="utf-8",
    )
    complete_path = reports_dir / "CLEANED_CORPUS_COMPLETE"
    complete_path.write_text(
        f"completed_at_utc={created_at}\nrecord_count={len(clean_rows)}\n",
        encoding="ascii",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
