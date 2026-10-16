#!/usr/bin/env python3
"""Read-only checks for a selected paper checkpoint suite and policy environment.

Passing preflight establishes available inputs, not successful model inference,
clean installation, numerical equivalence, or absence of original train/test
contamination. It never downloads packages/models or modifies archived results.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import sys

from experiment_matrix import (DEFAULT_MATRIX, EXPECTED_COUNTS, ROOT, adapter_hashes,
                               check_snapshot, data_path, load_matrix, model_snapshot,
                               select_entries, sha256)

POLICY_PACKAGES = {"torch": "2.11.0+cu128", "transformers": "4.57.6", "peft": "0.18.0",
                   "accelerate": "1.14.0", "nltk": "3.9.2", "numpy": "2.2.6",
                   "safetensors": "0.8.0", "simalign": "0.4"}
STAGE_PACKAGES = {
    "policy": tuple(POLICY_PACKAGES),
    "tts": ("torch", "transformers", "qwen-tts", "soundfile", "librosa", "numpy"),
    "seamless": ("torch", "torchaudio", "seamless-communication", "fairseq2", "fairseq2n", "simuleval", "sacrebleu"),
}


def environment_report(environment: str, require_recorded_versions=False) -> dict:
    packages, errors, warnings = {}, [], []
    for package in STAGE_PACKAGES[environment]:
        try:
            actual = importlib.metadata.version(package)
            packages[package] = {"version": actual, "recorded_current_server_version": POLICY_PACKAGES.get(package) if environment == "policy" else None}
            if environment == "policy" and actual != POLICY_PACKAGES[package]:
                message = f"{package}: installed {actual}, recorded server {POLICY_PACKAGES[package]}"
                (errors if require_recorded_versions else warnings).append(message)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = {"version": None}
            errors.append(f"Missing package: {package}")
    cuda = {}
    try:
        import torch
        cuda = {"available": torch.cuda.is_available(), "torch_cuda_build": torch.version.cuda,
                "device_count": torch.cuda.device_count(), "devices": []}
        for index in range(torch.cuda.device_count()):
            cuda["devices"].append({"index": index, "name": torch.cuda.get_device_name(index),
                                    "capability": list(torch.cuda.get_device_capability(index))})
        if not cuda["available"]:
            errors.append("CUDA is unavailable; checkpoint inference cannot run")
    except Exception as error:
        errors.append(f"Cannot inspect PyTorch/CUDA: {type(error).__name__}: {error}")
    return {"python": sys.executable, "python_version": sys.version, "environment": environment,
            "packages": packages, "cuda": cuda, "errors": errors, "warnings": warnings,
            "recorded_versions_are_current_server_not_a_verified_clean_install": True}


def check_inputs(entries: list[dict], data_root: Path, verify_hashes=False) -> dict:
    manifest = json.loads((ROOT / "provenance/artifact_manifest.json").read_text(encoding="utf-8"))
    lookup = {item["relative_to_remote_root"]: item for item in manifest["entries"]}
    errors, results = [], []
    for relative in sorted({value for row in entries for value in row["inputs"].values()}):
        path = data_path(data_root, relative)
        finding = {"relative": relative, "exists": path.is_file(), "sha256_matches": None}
        registered = lookup.get(relative)
        if not registered or not registered.get("sha256"):
            errors.append(f"Input lacks registered SHA-256: {relative}")
        if not path.is_file():
            errors.append(f"Missing input: {path}")
        elif registered and verify_hashes:
            finding["sha256_matches"] = sha256(path) == registered.get("sha256")
            if not finding["sha256_matches"]:
                errors.append(f"Input SHA-256 mismatch: {path}")
        results.append(finding)
    checked_adapters = set()
    for entry in entries:
        if entry["adapter_relative"] in checked_adapters:
            continue
        checked_adapters.add(entry["adapter_relative"])
        for finding in adapter_hashes(entry, data_root, manifest):
            results.append(finding)
            if not finding["exists"] or finding.get("error") or finding["matches_sha256"] is not True:
                errors.append(f"Selected adapter missing/unverified: {finding['relative']}")
    return {"data_root_semantics": "SN_Research-relative bundle root", "inputs": results, "errors": errors,
            "data_hashes_checked": verify_hashes, "selected_adapter_hashes_always_checked": True}


def nltk_report(resource_root: Path | None) -> dict:
    errors, resources = [], {}
    try:
        import nltk
        if resource_root:
            nltk.data.path.insert(0, str(resource_root.resolve()))
        for name in ("corpora/wordnet", "corpora/omw-1.4"):
            found = None
            for candidate in (name, name + ".zip"):
                try:
                    found = str(nltk.data.find(candidate)); break
                except LookupError:
                    pass
            resources[name] = found
            if found is None:
                errors.append(f"Missing NLTK resource {name}; provision separately, do not alter tokenization")
    except Exception as error:
        errors.append(f"Cannot inspect NLTK resources: {type(error).__name__}: {error}")
    return {"resources": resources, "errors": errors}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    p.add_argument("--family", choices=sorted(EXPECTED_COUNTS), action="append", default=[])
    p.add_argument("--id", action="append", default=[])
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--models-root", type=Path, required=True)
    p.add_argument("--environment", choices=sorted(STAGE_PACKAGES), default="policy")
    p.add_argument("--require-recorded-versions", action="store_true")
    p.add_argument("--verify-data-hashes", action="store_true")
    p.add_argument("--nltk-data", type=Path)
    p.add_argument("--skip-nltk", action="store_true", help="For TTS/Seamless-only environment inspection, not policy evaluation")
    args = p.parse_args()
    rows = select_entries(load_matrix(args.matrix), args.family, args.id)
    environment = environment_report(args.environment, args.require_recorded_versions)
    inputs = check_inputs(rows, args.data_root, args.verify_data_hashes)
    snapshot = model_snapshot(args.models_root)
    model_errors = check_snapshot(snapshot)
    nltk = {"errors": [], "status": "skipped_by_request"} if args.skip_nltk else nltk_report(args.nltk_data)
    errors = environment["errors"] + inputs["errors"] + model_errors + nltk["errors"]
    report = {"status": "pass" if not errors else "fail", "selected_rollout_count": len(rows),
              "families": dict(Counter(row["family"] for row in rows)), "environment": environment,
              "model": {"snapshot": str(snapshot), "errors": model_errors}, "inputs": inputs,
              "nltk": nltk, "errors": errors,
              "scope_warning": "Preflight is not successful inference, end-to-end retraining, or bitwise result reproduction."}
    print(json.dumps(report, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
