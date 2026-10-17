#!/usr/bin/env python3
"""Copy exactly the paper verifier's 83 inputs from an installed private bundle.

--data-root is the installed SN_Research logical root, containing data/how2sign_...
The output must be a new PRIVATE directory outside this repository and outside
the installed source tree. This performs offline registry checks, not fresh
server verification. It never creates verification_input_manifest.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/verification"))
from fetch_verification_inputs import REMOTE_ROOT, target_paths, validate_relative_path

DATA_PREFIX = "data/how2sign_english_asl_gloss_v1"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def contained(root: Path, relative: str) -> Path:
    path = root.joinpath(*validate_relative_path(relative).parts)
    resolved = path.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("Materialization path escapes its private root through a link")
    return path


def plan(data_root: Path, manifest_path: Path):
    registry = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    entries = {}
    for item in registry["entries"]:
        entries.setdefault(item["remote_path"], []).append(item)
    result = []
    for source_rel, alias in target_paths():
        validate_relative_path(source_rel)
        validate_relative_path(alias)
        remote_path = REMOTE_ROOT + "/" + source_rel
        matches = entries.get(remote_path, [])
        if len(matches) != 1:
            raise ValueError("Each allowed verification input needs exactly one artifact registry entry: " + source_rel)
        entry = matches[0]
        expected_relative = DATA_PREFIX + "/" + source_rel
        if (entry.get("type") != "file"
                or entry.get("relative_to_remote_root") != expected_relative
                or not isinstance(entry.get("sha256"), str)
                or not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"])
                or type(entry.get("size_bytes")) is not int
                or entry["size_bytes"] < 0):
            raise ValueError("Invalid artifact registry identity for: " + source_rel)
        source = contained(data_root, expected_relative)
        if not source.is_file():
            raise FileNotFoundError("Missing installed paper verification input: " + str(source))
        if source.stat().st_size != entry["size_bytes"] or digest(source) != entry["sha256"]:
            raise ValueError("Installed paper verification input hash/size mismatch: " + source_rel)
        result.append({"source_relative": expected_relative, "relative_path": alias,
                       "sha256": entry["sha256"], "bytes": entry["size_bytes"]})
    if len(result) != 83 or len({row["relative_path"] for row in result}) != 83:
        raise ValueError("The final-paper verification contract requires exactly 83 unique inputs")
    return result


def materialize(data_root: Path, output_dir: Path, manifest_path: Path, mode="copy"):
    data_root = data_root.resolve()
    output_dir = output_dir.resolve()
    if mode != "copy":
        raise ValueError("Only independent copies are supported; links would alter the protected bundle's link counts")
    if output_dir == ROOT or output_dir.is_relative_to(ROOT):
        raise ValueError("Raw verification inputs must remain outside the source repository")
    if output_dir == data_root or output_dir.is_relative_to(data_root) or data_root.is_relative_to(output_dir):
        raise ValueError("Source bundle and output directory must not overlap")
    if output_dir.exists():
        raise FileExistsError("Choose a new output directory; existing files are never overwritten")
    # Validate every input before creating any output file or directory.
    rows = plan(data_root, manifest_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    for row in rows:
        source = contained(data_root, row["source_relative"])
        dest = contained(output_dir, row["relative_path"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest = contained(output_dir, row["relative_path"])
        with source.open("rb") as incoming, dest.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, 8 * 1024 * 1024)
        if dest.stat().st_size != row["bytes"] or digest(dest) != row["sha256"]:
            raise ValueError("Copied input failed integrity check; preserve output for inspection")
    receipt = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(output_dir), "data_root": str(data_root), "mode": mode,
        "input_files": len(rows), "all_inputs_match_registered_sha256_and_size": True,
        "artifact_manifest_sha256": digest(manifest_path),
        "verification_scope": "Offline installed-bundle bytes matched the committed registry; no new server hash observation.",
        "remote_verification_performed": False,
        "files": rows,
    }
    receipt_path = contained(output_dir, "materialization_receipt.json")
    with receipt_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("copy",), default="copy")
    args = parser.parse_args()
    try:
        receipt = materialize(args.data_root, args.output_dir, ROOT / "provenance/artifact_manifest.json", args.mode)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({"input_files": receipt["input_files"], "input_root": receipt["input_root"],
                      "verifier_argument": "--input-root " + json.dumps(receipt["input_root"]),
                      "remote_verification_performed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
