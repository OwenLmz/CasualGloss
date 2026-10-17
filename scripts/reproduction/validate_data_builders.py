#!/usr/bin/env python3
"""Rerun unchanged CPU-only paper data producers into a NEW external directory.

--data-root is the installed private bundle root containing data/, not data/
itself. No model is loaded, trained, or downloaded. Select independent checks
with repeated --check waveform-qc, --check finalizer, or --check recovery;
the default runs all three. Waveform/finalizer checks require installed audio
and authenticated installation metadata, not just an audio directory index.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from release_assets import (load_catalog, no_link_ancestors, parse_file_manifest,
                            read_metadata, relative_path)

ROOT = Path(__file__).resolve().parents[2]
BASE = "data/how2sign_english_asl_gloss_v1"
SPEECH = BASE + "/chinese_speech_frontend_v0"
CORPUS = SPEECH + "/chinese_speech_to_pseudo_asl_corpus_v1_20260905"
FULL = BASE + "/full_v4_train"
RECOVERY = FULL + "/state_recovery_v2_3_full"
CHECKS = ("waveform-qc", "finalizer", "recovery")
PRODUCERS = {
    "waveform-qc": "speech/frontend/validate_audio_manifest.py",
    "finalizer": "speech/corpus/finalize_cleaned_corpus.py",
    "recovery": "data_preparation/recovery/prepare_v2_3_manifests.py",
}
CORPUS_PATHS = {
    "raw": CORPUS + "/manifests/full.tts.jsonl",
    "qc": CORPUS + "/reports/audio_validation.json",
    "config": CORPUS + "/config_full_corpus.json",
    "clean": CORPUS + "/manifests/full.cleaned.jsonl",
    "summary": CORPUS + "/reports/corpus_summary.cleaned.json",
}
RECOVERY_INPUTS = {
    "causal-pairs": FULL + "/semantic_causal_alignment_v2_2_full/outputs/semantic_causal_aligned_pairs_v2_2_full.jsonl",
    "evaluation-pairs": FULL + "/causal_alignment_v1_corrected_full/outputs/causal_aligned_pairs_full.jsonl",
    "balanced-train-actions": FULL + "/fine_tuning_v2_semantic_v2_2_full/manifests_wait_balanced_v2_2_1/train_actions.jsonl",
    "validation-actions": FULL + "/semantic_causal_alignment_v2_2_full/manifests_treebank_v2_2_1/validation_actions.jsonl",
}
RECOVERY_OUTPUTS = tuple("manifests/" + name + ".jsonl" for name in
                         ("train_actions_full", "development_actions", "confirmation_actions",
                          "development_pairs", "confirmation_pairs"))
# Historical manifest_build_report.json and recovery_reference_closure.json.
# This intermediate is rebuilt, not redundantly stored in the private payload.
AUGMENTATION = {"relative_path": "augmentation/state_recovery_actions_train.jsonl",
                "row_count": 71836,
                "sha256": "eba874294e4694b9cd6531da873461dec5a7a2ca8a3cd45c51d83d1c1fb3159a"}
RECOVERY_COUNTS = {"source_train_parent_count": 18632,
                   "balanced_on_policy_action_count": 367267,
                   "recovery_action_count": 71836, "full_train_action_count": 439103}
SUMMARY_IGNORES = {"created_at_utc", "exclusions_sha256", "clean_config_sha256"}


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count_rows(path):
    with path.open(encoding="utf-8-sig") as handle:
        return sum(bool(line.strip()) for line in handle)


def safe_new_output(path, source, data_root):
    require(path.is_absolute() and ".." not in path.parts,
            "Use a normalized absolute NEW output directory")
    no_link_ancestors(path)
    output = path.resolve()
    require(output != Path(output.anchor) and len(output.parts) >= 3,
            "Choose a sufficiently narrow output directory, not a filesystem root")
    for protected in (source.resolve(), data_root.resolve()):
        require(not (output == protected or protected in output.parents or output in protected.parents),
                "Output must not overlap source or the entire installed bundle")
    require(not output.exists(), "Output already exists; preserve it and choose a NEW directory")
    return output


def checked_file(root, relative, entry):
    relative = relative_path(relative)
    target = root.joinpath(*relative.split("/"))
    no_link_ancestors(target)
    require(target.resolve().is_relative_to(root.resolve()), "Input escapes its protected root")
    require(target.is_file(), "Missing required input/source: " + str(target))
    require(isinstance(entry.get("sha256"), str) and len(entry["sha256"]) == 64,
            "Missing registered SHA-256: " + relative)
    require(target.stat().st_size == entry["size_bytes"] and sha256(target) == entry["sha256"],
            "Registered input/source SHA-256 or size mismatch: " + relative)
    return target


def required_assets(checks):
    required = set()
    if "waveform-qc" in checks:
        required.update(CORPUS_PATHS[name] for name in ("raw", "qc"))
    if "finalizer" in checks:
        required.update(CORPUS_PATHS.values())
    if "recovery" in checks:
        required.update(RECOVERY_INPUTS.values())
        required.update(RECOVERY + "/" + relative for relative in RECOVERY_OUTPUTS)
    return sorted(required)


def verify_audio(data_root, source, raw_path):
    """Anchor every referenced WAV to the outer-catalog-checked file manifest."""
    catalog = load_catalog(source / "provenance/release_assets.json")
    metadata = read_metadata(data_root, catalog)
    require(metadata is not None and isinstance(metadata.get("file_manifest_utf8"), str),
            "Waveform/finalizer checks require a verified Release installation with audio")
    files = parse_file_manifest(metadata["file_manifest_utf8"].encode("utf-8"), catalog)
    seen = set()
    receipt_digest = hashlib.sha256()
    with raw_path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            relative = SPEECH + "/" + relative_path(row["tts_audio_16k_mono"])
            require(relative not in seen, "Duplicate normalized waveform reference")
            seen.add(relative)
            entry = files.get(relative)
            require(entry is not None and entry["group"] == "audio",
                    "Normalized waveform is absent from the checked Release manifest: " + relative)
            checked_file(data_root, relative, entry)
            receipt_digest.update((relative + "\0" + entry["sha256"] + "\n").encode("utf-8"))
    return {"normalized_waveforms_hash_verified": len(seen),
            "ordered_path_and_sha256_digest": receipt_digest.hexdigest(),
            "release_file_manifest_sha256": catalog["manifest"]["sha256"],
            "only_referenced_normalized_waveforms_checked": True}


def preflight(data_root, output_root, checks, tts_python, policy_python, source=None):
    source = (source or ROOT).resolve(strict=True)
    no_link_ancestors(data_root)
    data_root = data_root.resolve(strict=True)
    require(data_root.is_dir(), "--data-root must be an installed bundle directory")
    checks = tuple(check for check in CHECKS if check in checks)
    require(bool(checks), "Select at least one data-builder check")
    output = safe_new_output(output_root, source, data_root)
    artifacts = read_json(source / "provenance/artifact_manifest.json")
    registry = {}
    for entry in artifacts["entries"]:
        name = entry["relative_to_remote_root"]
        require(name not in registry, "Duplicate artifact registry entry: " + name)
        registry[name] = entry
    inputs = {}
    for relative in required_assets(checks):
        entry = registry.get(relative)
        require(entry is not None and entry["type"] == "file", "Required file is not registered: " + relative)
        checked_file(data_root, relative, entry)
        inputs[relative] = {"sha256": entry["sha256"], "size_bytes": entry["size_bytes"]}
    source_entries = {}
    for item in read_json(source / "provenance/source_manifest.json")["files"]:
        key = item["repository_path"]
        require(key not in source_entries, "Duplicate producer source registry entry")
        source_entries[key] = {"sha256": item.get("repository_sha256", item["sha256"]),
                               "size_bytes": item.get("repository_size_bytes", item["size_bytes"])}
    producers = {}
    for check in checks:
        relative = PRODUCERS[check]
        require(relative in source_entries, "Producer has no source-provenance record: " + relative)
        checked_file(source, relative, source_entries[relative])
        producers[check] = source_entries[relative]
    for interpreter in {tts_python if check != "recovery" else policy_python for check in checks}:
        require(Path(interpreter).is_file(), "Interpreter does not exist: " + str(interpreter))
    audio = None
    if "waveform-qc" in checks or "finalizer" in checks:
        audio = verify_audio(data_root, source, data_root / CORPUS_PATHS["raw"])
    return {"source": source, "data_root": data_root, "output_root": output, "checks": checks,
            "inputs": inputs, "producers": producers, "audio": audio,
            "artifact_registry_sha256": sha256(source / "provenance/artifact_manifest.json"),
            "source_registry_sha256": sha256(source / "provenance/source_manifest.json")}


def run_original(report, plan, check, python, arguments):
    source = plan["source"]
    script = checked_file(source, PRODUCERS[check], plan["producers"][check])
    log = plan["output_root"] / (check + ".log")
    command = [str(python), "-B", str(script), *map(str, arguments)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1",
               PYTHONPATH=str(script.parent), CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1")
    with log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=source, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    report["producer_executions"].append({"check": check, "command": command,
        "returncode": completed.returncode, "source_sha256": sha256(script),
        "log": str(log), "log_sha256": sha256(log)})
    require(completed.returncode == 0, "Original producer failed: " + check + "; inspect " + str(log))


def differing_fields(actual, expected, ignored=()):
    keys = (set(actual) | set(expected)) - set(ignored)
    return sorted(key for key in keys if key not in actual or key not in expected or actual[key] != expected[key])


def validate(plan, tts_python, policy_python):
    output = plan["output_root"]
    output.mkdir(parents=True, exist_ok=False)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "passed": False,
              "source": str(plan["source"]), "data_root": str(plan["data_root"]), "output_root": str(output),
              "source_and_original_results_modified": False, "models_loaded_or_trained": False,
              "checks_requested": list(plan["checks"]), "registered_inputs": plan["inputs"],
              "artifact_registry_sha256": plan["artifact_registry_sha256"],
              "source_registry_sha256": plan["source_registry_sha256"],
              "audio_input_verification": plan["audio"], "producer_executions": [], "checks": {}, "errors": []}
    data = plan["data_root"]
    raw_qc_path = data / CORPUS_PATHS["qc"]
    try:
        if "waveform-qc" in plan["checks"] or "finalizer" in plan["checks"]:
            raw_qc = read_json(raw_qc_path)
            require(count_rows(data / CORPUS_PATHS["raw"]) == raw_qc["expected_count"] == raw_qc["record_count"],
                    "Frozen raw QC and manifest counts differ")
        if "waveform-qc" in plan["checks"]:
            new_qc_path = output / "audio_validation.rerun.json"
            run_original(report, plan, "waveform-qc", tts_python,
                         ["--manifest", data / CORPUS_PATHS["raw"], "--root", data / SPEECH,
                          "--report", new_qc_path, "--expected-count", raw_qc["expected_count"]])
            new_qc = read_json(new_qc_path)
            differences = differing_fields(new_qc, raw_qc, {"created_at_utc"})
            report["checks"]["waveform_qc"] = {"passed": not differences,
                "record_count": new_qc["record_count"], "status_counts": new_qc["status_counts"],
                "issue_count": len(new_qc["issues"]), "mismatched_fields": differences,
                "ignored_fields": ["created_at_utc"], "waveform_samples_revalidated": True,
                "original_qc_sha256": sha256(raw_qc_path), "new_qc_sha256": sha256(new_qc_path)}
            require(not differences, "Rerun waveform QC differs from frozen fields: " + repr(differences))
            raw_qc_path = new_qc_path
        if "finalizer" in plan["checks"]:
            final_output = output / "cleaned_corpus"
            run_original(report, plan, "finalizer", tts_python,
                         ["--raw-manifest", data / CORPUS_PATHS["raw"], "--raw-audio-validation", raw_qc_path,
                          "--raw-config", data / CORPUS_PATHS["config"], "--root", data / SPEECH,
                          "--output-dir", final_output, "--expected-raw-count", raw_qc["record_count"],
                          "--expected-exclusion-count", len(raw_qc["issues"])])
            new_clean = final_output / "manifests/full.cleaned.jsonl"
            expected_sha = plan["inputs"][CORPUS_PATHS["clean"]]["sha256"]
            actual_sha = sha256(new_clean)
            new_summary = read_json(final_output / "reports/corpus_summary.cleaned.json")
            differences = differing_fields(new_summary, read_json(data / CORPUS_PATHS["summary"]), SUMMARY_IGNORES)
            passed = actual_sha == expected_sha and not differences
            report["checks"]["cleaned_corpus"] = {"passed": passed,
                "rebuilt_manifest_sha256": actual_sha, "expected_manifest_sha256": expected_sha,
                "manifest_bytes_and_order_identical": actual_sha == expected_sha,
                "stable_summary_mismatched_fields": differences,
                "summary_comparison_ignores": sorted(SUMMARY_IGNORES),
                "waveform_qc_source": "freshly_revalidated" if "waveform-qc" in plan["checks"] else "frozen_registered_report",
                "ignored_metadata_reason": "Fresh report timestamps/paths affect dependent report hashes; no report byte-identity claim."}
            require(passed, "Rebuilt cleaned manifest or stable summary differs from frozen output")
        if "recovery" in plan["checks"]:
            recovery_output = output / "recovery_manifests"
            arguments = [item for name, value in RECOVERY_INPUTS.items() for item in ("--" + name, data / value)]
            run_original(report, plan, "recovery", policy_python, arguments +
                         ["--output-root", recovery_output, "--seed", 20260829, "--development-parents", 250])
            files = []
            for relative in RECOVERY_OUTPUTS:
                original, rebuilt = data / RECOVERY / relative, recovery_output / relative
                expected_sha = plan["inputs"][RECOVERY + "/" + relative]["sha256"]
                actual_sha = sha256(rebuilt)
                actual_count, expected_count = count_rows(rebuilt), count_rows(original)
                files.append({"relative_path": relative, "rebuilt_sha256": actual_sha, "expected_sha256": expected_sha,
                              "row_count": actual_count, "expected_row_count": expected_count,
                              "passed": actual_sha == expected_sha and actual_count == expected_count})
            augmentation = recovery_output / AUGMENTATION["relative_path"]
            actual_sha, actual_count = sha256(augmentation), count_rows(augmentation)
            files.append({"relative_path": AUGMENTATION["relative_path"], "rebuilt_sha256": actual_sha,
                          "expected_sha256": AUGMENTATION["sha256"], "row_count": actual_count,
                          "expected_row_count": AUGMENTATION["row_count"], "reference_payload_not_required": True,
                          "passed": actual_sha == AUGMENTATION["sha256"] and actual_count == AUGMENTATION["row_count"]})
            build_report = read_json(recovery_output / "reports/manifest_build_report.json")
            counts = {key: build_report.get(key) for key in RECOVERY_COUNTS}
            input_hashes = {name.replace("-", "_"): plan["inputs"][relative]["sha256"] for name, relative in RECOVERY_INPUTS.items()}
            passed = all(item["passed"] for item in files) and counts == RECOVERY_COUNTS and build_report.get("sha256_inputs") == input_hashes
            report["checks"]["recovery_manifests"] = {"passed": passed, "files": files,
                "computed_counts": counts, "expected_counts": RECOVERY_COUNTS,
                "producer_recorded_input_hashes_match": build_report.get("sha256_inputs") == input_hashes}
            require(passed, "Rebuilt recovery bytes, counts or producer input hashes differ from frozen references")
        report["passed"] = True
    except Exception as error:
        report["errors"].append(type(error).__name__ + ": " + str(error))
    report_path = output / "cpu_validation_report.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--check", action="append", choices=CHECKS, default=[])
    parser.add_argument("--tts-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--policy-python", type=Path, default=Path(sys.executable))
    args = parser.parse_args(argv)
    try:
        plan = preflight(args.data_root, args.output_root, args.check or CHECKS, args.tts_python, args.policy_python)
        report = validate(plan, args.tts_python, args.policy_python)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({"passed": report["passed"], "report": str(plan["output_root"] / "cpu_validation_report.json"),
                      "checks": list(plan["checks"]), "errors": report["errors"]}, indent=2))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
