#!/usr/bin/env python3
"""Load the bounded final-paper checkpoint matrix without importing ML packages.

The matrix enumerates historical invocations, not new experimental settings.
SN_Research-relative data paths resolve under --data-root. Model snapshots are
resolved separately so an authorized data bundle need not duplicate base models.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = ROOT / "configs/reproduction/experiments.json"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
EXPECTED_COUNTS = {"locked500": 35, "clerc401": 35, "diagnostic250": 8, "speech500": 4}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_./-]+", value):
        raise ValueError(f"Unsafe matrix relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or value == ".":
        raise ValueError(f"Non-normalized matrix path: {value!r}")
    return path


def data_path(data_root: Path, value: str) -> Path:
    root = data_root.resolve()
    path = root.joinpath(*relative_path(value).parts).resolve()
    if root not in path.parents:
        raise ValueError(f"Data symlink/path escapes the installed bundle: {value}")
    return path


def load_matrix(path: Path = DEFAULT_MATRIX) -> dict:
    matrix = json.loads(path.read_text(encoding="utf-8"))
    entries = matrix["experiments"]
    if len(entries) != 82 or len({row["id"] for row in entries}) != 82:
        raise ValueError("Expected exactly 82 unique final-paper checkpoint evaluations")
    if dict(Counter(row["family"] for row in entries)) != EXPECTED_COUNTS:
        raise ValueError("Final-paper evaluation family counts do not match 35/35/8/4")
    for row in entries:
        if not re.fullmatch(r"[a-z0-9_]+", row["id"]):
            raise ValueError("Unsafe experiment ID")
        if row["model_revision"] != MODEL_REVISION or row["model_id"] != MODEL_ID:
            raise ValueError("The paper matrix must use its fixed Qwen snapshot")
        if row["precision"] not in ("bfloat16", "float16"):
            raise ValueError("Unknown recorded inference precision")
        if row["generation_batch_size"] not in (1, 16, 32):
            raise ValueError("Missing/unknown recorded generation batch size")
        if row.get("inference_world_size") != 1:
            raise ValueError("Recorded checkpoint inference is single-process, single-device per invocation")
        for key in ("adapter_relative", "historical_output_relative", "verification_local_relative"):
            relative_path(row[key])
        relative_path(row["config"])
        for value in row["inputs"].values():
            relative_path(value)
        if set(row["input_sha256"]) != set(row["inputs"]):
            raise ValueError("Input fingerprints must cover every selected input")
        for value in [row["config_sha256"], *row["input_sha256"].values(), *row["adapter_sha256"].values()]:
            if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
                raise ValueError("Every selected config, input and adapter needs a valid recorded SHA-256")
        if sha256(ROOT / row["config"]) != row["config_sha256"]:
            raise ValueError(f"Repository config differs from final-paper matrix: {row['config']}")
        if not row.get("invocation_evidence") or not row.get("source_order"):
            raise ValueError("Every run needs invocation and source-order evidence")
        if row["checkpoint"] not in ("adapter_final", "adapter_best"):
            raise ValueError("Unselected checkpoint in final-paper matrix")
    return matrix


def select_entries(matrix: dict, families=(), ids=()) -> list[dict]:
    families, ids = set(families), set(ids)
    known_ids = {row["id"] for row in matrix["experiments"]}
    if ids - known_ids or families - set(EXPECTED_COUNTS):
        raise ValueError(f"Unknown selection: {sorted(ids-known_ids)} / {sorted(families-set(EXPECTED_COUNTS))}")
    selected = [row for row in matrix["experiments"]
                if (not families or row["family"] in families) and (not ids or row["id"] in ids)]
    if not selected:
        raise ValueError("The selected family/ID intersection is empty")
    return selected


def model_snapshot(models_root: Path) -> Path:
    """Accept a snapshot itself, HF_HOME, HF_HOME/hub, or a model cache root."""
    root = models_root.resolve()
    candidates = [root, root / "hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots" / MODEL_REVISION,
                  root / "models--Qwen--Qwen2.5-7B-Instruct/snapshots" / MODEL_REVISION]
    for path in candidates:
        if path.name == MODEL_REVISION and (path / "config.json").is_file():
            return path
    # Deterministic missing path is useful in dry-run reports; preflight refuses it.
    return candidates[1]


def check_snapshot(path: Path) -> list[str]:
    errors = []
    if path.name != MODEL_REVISION:
        errors.append("Base snapshot directory must carry the recorded revision")
    required = ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors.index.json")
    for name in required:
        if not (path / name).is_file():
            errors.append(f"Missing base-model file: {path/name}")
    index = path / "model.safetensors.index.json"
    if index.is_file():
        try:
            shards = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
            if len(shards) != 4:
                errors.append(f"Expected 4 recorded base-model shards, found {len(shards)}")
            for shard in shards:
                target = path.joinpath(*relative_path(shard).parts)
                if not target.is_file() or target.stat().st_size == 0:
                    errors.append(f"Missing/empty base-model shard: {target}")
        except (ValueError, KeyError, TypeError) as error:
            errors.append(f"Invalid model weight index: {error}")
    return errors


def protected_output(output_root: Path, data_root: Path, models_root: Path) -> Path:
    """Refuse writes to source, installed data, models, or their broad ancestors."""
    output = output_root.resolve()
    if output == Path(output.anchor) or len(output.parts) < 3:
        raise ValueError("Choose an explicit, sufficiently narrow new output directory")
    for protected in (ROOT.resolve(), (data_root / "data").resolve(), models_root.resolve()):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError(f"Output overlaps a protected source/data/model location: {protected}")
    return output


def adapter_hashes(entry: dict, data_root: Path, artifact_manifest: dict) -> list[dict]:
    """Inspect exactly the selected adapter config and weights against the asset index."""
    findings = []
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        relative = entry["adapter_relative"] + "/" + filename
        matches = [item for item in artifact_manifest["entries"]
                   if item["relative_to_remote_root"] == relative]
        target = data_path(data_root, relative)
        finding = {"relative": relative, "exists": target.is_file(), "matches_sha256": None}
        if len(matches) != 1 or not matches[0].get("sha256"):
            finding["error"] = "Selected adapter lacks a unique registered SHA-256"
        elif target.is_file():
            finding["matches_sha256"] = sha256(target) == matches[0]["sha256"]
        findings.append(finding)
    return findings
