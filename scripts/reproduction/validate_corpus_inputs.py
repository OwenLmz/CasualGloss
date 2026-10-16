#!/usr/bin/env python3
"""Check the frozen corpus cleaning/repair inputs without generating new audio.

This is a release acceptance check, not a replacement for the original waveform
QC in speech/frontend/validate_audio_manifest.py. Counts and durations are
computed from the supplied manifests, then compared with the supplied report.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def json_file(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def jsonl_file(path: Path):
    lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    return lines, [json.loads(line) for line in lines]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(raw_lines, raw, clean_lines, clean, qc, summary, repairs, repair_report,
             locked, config, audio_root=None):
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    raw_by_id = {str(row["id"]): row for row in raw}
    clean_by_id = {str(row["id"]): row for row in clean}
    issue_ids = [str(issue["id"]) for issue in qc["issues"]]
    excluded = set(issue_ids)
    require(len(raw_by_id) == len(raw), "duplicate raw ID")
    require(len(clean_by_id) == len(clean), "duplicate cleaned ID")
    require(len(excluded) == len(issue_ids), "duplicate raw QC issue ID")
    require(bool(excluded), "expected the original non-empty raw QC report, not cleaned QC")
    require(excluded <= set(raw_by_id), "QC ID missing from raw manifest")
    require(len(raw) == qc["record_count"] == qc["expected_count"], "raw QC record count mismatch")
    require(len(excluded) == qc["status_counts"].get("flagged", 0), "raw QC flag count mismatch")
    expected_lines = [line for line, row in zip(raw_lines, raw) if str(row["id"]) not in excluded]
    require(clean_lines == expected_lines, "cleaned manifest is not the exact ordered raw-minus-QC transform")
    require(all(raw_by_id[item]["split"] == "train" for item in excluded if item in raw_by_id),
            "QC excludes validation/test data")
    require(not any(raw_by_id[item].get("locked_evaluation_parent") for item in excluded if item in raw_by_id),
            "QC excludes locked evaluation data")

    total_seconds = sum(float(row["tts_duration_seconds"]) for row in clean)
    actual = {
        "raw_record_count": len(raw),
        "excluded_record_count": len(excluded),
        "record_count": len(clean),
        "unique_id_count": len(clean_by_id),
        "split_counts": dict(sorted(Counter(str(row["split"]) for row in clean).items())),
        "speaker_counts": dict(sorted(Counter(str(row["tts_speaker"]) for row in clean).items())),
        "translation_status_counts": dict(sorted(Counter(str(row.get("source_text_zh_status", "missing")) for row in clean).items())),
        "tts_status_counts": dict(sorted(Counter(str(row.get("tts_status", "missing")) for row in clean).items())),
        "total_duration_seconds": round(total_seconds, 6),
        "total_duration_hours": round(total_seconds / 3600.0, 6),
    }
    for key, value in actual.items():
        require(value == summary.get(key), f"saved corpus summary mismatch: {key}")
    require(actual["translation_status_counts"] == {"completed": len(clean)}, "incomplete translations")
    require(actual["tts_status_counts"] == {"completed": len(clean)}, "incomplete TTS")
    require(set(actual["speaker_counts"]) == set(config["tts"]["speakers"]), "voice inventory mismatch")

    applied = {str(item["id"]): item for item in repair_report["repairs"]}
    require(set(applied) == set(repairs), "repair report/specification IDs differ")
    require(repair_report.get("applied_repair_count") == len(repairs) and repair_report.get("ok") is True,
            "historical repair report incomplete")
    for identifier, repair in repairs.items():
        row = raw_by_id.get(identifier, {})
        item = applied.get(identifier, {})
        require(row.get("source_text_zh") == repair["translation"] == item.get("after"),
                f"final translation does not equal the frozen repair: {identifier}")
        require(row.get("source_text_zh_repair_reason") == repair["reason"] == item.get("reason"),
                f"repair reason mismatch: {identifier}")
        require(row.get("source_text_zh_model_output_before_repair") == item.get("before"),
                f"pre-repair model output mismatch: {identifier}")

    voices = config["tts"]["speakers"]
    locked_ids = [str(row["id"]) for row in locked]
    require(len(set(locked_ids)) == len(locked), "duplicate locked500 ID")
    require(set(locked_ids) == {str(row["id"]) for row in clean if row["split"] == "test"},
            "locked manifest does not cover exactly the cleaned test split")
    locked_mismatches = 0
    for index, row in enumerate(locked):
        expected = (int(config["seed"]) + index, voices[index % len(voices)],
                    config["tts"]["model_revision"], config["translation"]["model_revision"])
        observed = (int(row["tts_seed"]), row["tts_speaker"], row["tts_model_revision"],
                    row["source_text_zh_model_revision"])
        locked_mismatches += expected != observed
        full_row = clean_by_id.get(str(row["id"]), {})
        require(full_row.get("split") == "test", "locked test row not present in cleaned test split")
        for field in ("tts_audio_original", "tts_audio_16k_mono", "tts_seed", "tts_speaker", "tts_duration_seconds"):
            require(full_row.get(field) == row.get(field), f"locked test asset altered: {field}")
    require(locked_mismatches == 0, "locked500 seed/voice/model revision mismatch")
    actual["locked_record_count"] = len(locked)
    actual["locked_generation_parameter_mismatches"] = locked_mismatches
    actual["curated_translation_repair_count"] = len(repairs)
    actual["audio_reference_existence_checked"] = audio_root is not None
    if audio_root is not None:
        root = Path(audio_root).resolve()
        missing = unsafe = 0
        for row in clean:
            for field in ("tts_audio_original", "tts_audio_16k_mono"):
                path = (root / row[field]).resolve()
                if not path.is_relative_to(root):
                    unsafe += 1
                elif not path.is_file():
                    missing += 1
        actual["missing_clean_audio_reference_count"] = missing
        actual["unsafe_audio_reference_count"] = unsafe
        require(missing == 0 and unsafe == 0, "missing or escaping audio reference")
    return {"passed": not errors, "computed": actual, "errors": errors,
            "waveform_samples_revalidated": False,
            "note": "Manifest transform/repair integrity only. Run the original waveform QC separately."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw-manifest", "clean-manifest", "raw-audio-validation", "summary",
                 "repair-spec", "repair-report", "locked-manifest", "locked-config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--audio-root", type=Path)
    args = parser.parse_args()
    raw_lines, raw = jsonl_file(args.raw_manifest)
    clean_lines, clean = jsonl_file(args.clean_manifest)
    _, locked = jsonl_file(args.locked_manifest)
    report = validate(raw_lines, raw, clean_lines, clean, json_file(args.raw_audio_validation),
                      json_file(args.summary), json_file(args.repair_spec), json_file(args.repair_report),
                      locked, json_file(args.locked_config), args.audio_root)
    report["input_sha256"] = {name: sha256(value) for name, value in vars(args).items()
                              if isinstance(value, Path) and name not in {"audio_root", "output"}}
    summary = json_file(args.summary)
    for arg, key in ((args.raw_manifest, "raw_manifest_sha256"), (args.clean_manifest, "clean_manifest_sha256")):
        if sha256(arg) != summary.get(key):
            report["errors"].append(f"saved manifest hash mismatch: {key}")
    report["passed"] = not report["errors"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
