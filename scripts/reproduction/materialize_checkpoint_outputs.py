#!/usr/bin/env python3
"""Copy a complete fresh checkpoint suite into the unchanged 83-input verifier layout.

Reads the public runner's checkpoint_output_index_*.json and receipt.json files;
no scheduler-specific metadata is required. Fresh predictions are checked against
their own completed receipts, NEVER against frozen historical prediction hashes.
The one CLERC reference manifest is checked against the installed asset registry.
The original numerical verifier remains unchanged and may report real differences.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ROLLOUT_COUNT = 82
EXPECTED_REFERENCE_COUNT = 401
DATA_PREFIX = "data/how2sign_english_asl_gloss_v1"
RECEIPT_NAME = "fresh_checkpoint_materialization_receipt.json"


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


matrix_api = module_at("fresh_materialize_matrix", ROOT / "scripts/reproduction/experiment_matrix.py")
targets_api = module_at("fresh_materialize_targets", ROOT / "scripts/verification/fetch_verification_inputs.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def no_links(path):
    for candidate in (path, *path.parents):
        require(not candidate.is_symlink(), "Symlink paths are not accepted: " + str(candidate))
        junction = getattr(candidate, "is_junction", None)
        require(not junction or not junction(), "Junction paths are not accepted: " + str(candidate))


def absolute(value):
    path = Path(value)
    require(path.is_absolute(), "Recorded paths must be absolute: " + str(value))
    no_links(path)
    return path.resolve()


def relative(value):
    require(isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9_./-]+", value)), "Unsafe relative path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and path.as_posix() == value and value != ".",
            "Relative path must be normalized and traversal-free")
    return path


def overlap(left, right):
    return left == right or left in right.parents or right in left.parents


def load_contract():
    entries = matrix_api.load_matrix()["experiments"]
    require(len(entries) == EXPECTED_ROLLOUT_COUNT and len({r["id"] for r in entries}) == len(entries),
            "Expected exactly the complete final-paper rollout matrix")
    aliases = {r["verification_local_relative"] for r in entries}
    require(len(aliases) == len(entries), "Matrix aliases must be unique")
    targets = targets_api.target_paths()
    require(len(targets) == EXPECTED_ROLLOUT_COUNT + 1 and len({alias for _, alias in targets}) == len(targets),
            "Verifier target layout must contain exactly 82 predictions plus one reference")
    refs = [(source, alias) for source, alias in targets if alias not in aliases]
    require(len(refs) == 1 and refs[0][0].endswith("/epee_sources_401.jsonl"), "Missing unique original CLERC reference target")
    require(aliases | {refs[0][1]} == {alias for _, alias in targets}, "Matrix aliases and verifier targets differ")
    for row in entries:
        relative(row["verification_local_relative"])
        require(type(row["expected_count"]) is int and row["expected_count"] > 0, "Invalid expected rollout count")
    return entries, DATA_PREFIX + "/" + refs[0][0], refs[0][1]


def count_records(path):
    with path.open(encoding="utf-8-sig") as handle:
        return sum(bool(line.strip()) for line in handle)


def collect(checkpoint_roots, entries):
    expected = {r["id"]: r for r in entries}
    indexed, members, evidence = {}, {}, []
    for root in checkpoint_roots:
        paths = list(root.rglob("*"))
        for path in paths:
            no_links(path)
        indices = sorted(p for p in paths if p.is_file() and p.match("checkpoint_output_index_*.json"))
        require(bool(indices), "No checkpoint output index found under: " + str(root))
        for index_path in indices:
            index = read_json(index_path)
            rows = index["outputs"]
            require(index.get("results_are_new_inference") is True
                    and type(index.get("rollout_count")) is int and index["rollout_count"] == len(rows),
                    "Index is not a valid fresh-inference output index: " + str(index_path))
            for row in rows:
                identifier = row["id"]
                require(identifier in expected, "Unexpected indexed experiment ID: " + identifier)
                require(identifier not in indexed, "Duplicate indexed experiment ID: " + identifier)
                entry = expected[identifier]
                require(row["verification_local_relative"] == entry["verification_local_relative"], "Index alias differs from matrix: " + identifier)
                require(row["family"] == entry["family"] and row["historical_output_relative"] == entry["historical_output_relative"],
                        "Index metadata differs from matrix: " + identifier)
                result = absolute(row["result"])
                require(root in result.parents and result.is_file(), "Indexed result is outside its declared checkpoint root or missing: " + str(result))
                indexed[identifier] = {"source": result, "index": index_path, "root": root}
            evidence.append({"path": str(index_path), "sha256": digest(index_path), "kind": "fresh_output_index"})
        receipts = sorted(p for p in paths if p.is_file() and p.name == "receipt.json")
        require(bool(receipts), "No completed runner receipts under: " + str(root))
        for receipt_path in receipts:
            receipt = read_json(receipt_path)
            require(receipt.get("id") == receipt_path.parent.name and receipt.get("status") == "completed" and type(receipt.get("returncode")) is int
                    and receipt["returncode"] == 0 and receipt.get("result_is_fresh_inference_not_archived_copy") is True,
                    "Failed/incomplete/non-fresh checkpoint receipt: " + str(receipt_path))
            require(bool(receipt.get("members")), "Empty checkpoint receipt: " + str(receipt_path))
            for row in receipt["members"]:
                identifier = row["id"]
                require(identifier in expected and identifier not in members, "Unexpected or duplicate receipted ID: " + identifier)
                result = absolute(row["result"])
                require(root in result.parents and receipt_path.parent in result.parents and result.is_file(),
                        "Receipted result escapes its runner job/root or is missing: " + str(result))
                require(row.get("status") == "completed" and type(row.get("record_count")) is int
                        and row["record_count"] == expected[identifier]["expected_count"]
                        and ("expected_count" not in row or (type(row["expected_count"]) is int
                             and row["expected_count"] == expected[identifier]["expected_count"])),
                        "Incomplete or wrong-size member receipt: " + identifier)
                require(isinstance(row.get("sha256"), str) and bool(re.fullmatch(r"[a-f0-9]{64}", row["sha256"])),
                        "Invalid fresh-output SHA-256: " + identifier)
                require(count_records(result) == row["record_count"] and digest(result) == row["sha256"],
                        "Fresh output count/SHA differs from its own receipt: " + identifier)
                members[identifier] = {"source": result, "sha256": row["sha256"], "bytes": result.stat().st_size,
                                       "record_count": row["record_count"], "receipt": str(receipt_path)}
            evidence.append({"path": str(receipt_path), "sha256": digest(receipt_path), "kind": "completed_fresh_runner_receipt"})
    require(set(indexed) == set(expected) and set(members) == set(expected),
            "Incomplete checkpoint suite; indexed missing=" + str(sorted(set(expected) - set(indexed)))
            + "; receipted missing=" + str(sorted(set(expected) - set(members))))
    rows = []
    for entry in entries:
        identifier = entry["id"]
        require(indexed[identifier]["source"] == members[identifier]["source"], "Index and receipt result paths differ: " + identifier)
        rows.append({"id": identifier, "relative_path": entry["verification_local_relative"],
                     "origin": "fresh_checkpoint_inference", **members[identifier]})
    require(len({row["source"] for row in rows}) == len(rows), "Different experiment IDs share one result file")
    return rows, evidence


def reference_row(data_root, registry_path, source_relative, alias):
    registry = read_json(registry_path)
    matches = [r for r in registry["entries"] if r.get("relative_to_remote_root") == source_relative]
    require(len(matches) == 1, "Reference manifest lacks a unique artifact-registry entry")
    entry = matches[0]
    require(entry.get("type") == "file" and type(entry.get("size_bytes")) is int
            and isinstance(entry.get("sha256"), str) and bool(re.fullmatch(r"[a-f0-9]{64}", entry["sha256"])),
            "Invalid registered CLERC reference identity")
    source = absolute(data_root.joinpath(*relative(source_relative).parts))
    require(data_root in source.parents and source.is_file(), "Reference path escapes bundle or is missing")
    require(source.stat().st_size == entry["size_bytes"] and digest(source) == entry["sha256"],
            "Installed CLERC reference differs from registered SHA/size")
    require(count_records(source) == EXPECTED_REFERENCE_COUNT, "Original CLERC reference count is not 401")
    relative(alias)
    return {"id": "clerc_reference_manifest", "relative_path": alias, "origin": "registered_original_clerc_reference",
            "source": source, "sha256": entry["sha256"], "bytes": entry["size_bytes"], "record_count": EXPECTED_REFERENCE_COUNT}


def materialize(checkpoint_roots, data_root, output_dir, *, models_root=None, registry_path=None):
    roots = [absolute(path) for path in checkpoint_roots]
    require(bool(roots) and all(r.is_dir() for r in roots), "Declare one or more existing checkpoint root directories")
    require(all(not overlap(a, b) for i, a in enumerate(roots) for b in roots[i+1:]), "Checkpoint roots must be distinct and non-overlapping")
    data_root = absolute(data_root)
    output_dir = absolute(output_dir)
    require(data_root.is_dir(), "Installed bundle root is missing")
    require(output_dir.parent != output_dir and len(output_dir.parts) >= 3, "Choose a narrow new output directory")
    protected = [ROOT.resolve(), data_root, *roots]
    if models_root is not None:
        protected.append(Path(models_root).resolve())
    if os.environ.get("HF_HOME"):
        protected.append(Path(os.environ["HF_HOME"]).resolve())
    if os.name == "posix":
        protected.append(Path("/123090274/SN_Research"))
    for path in protected:
        require(not overlap(output_dir, path), "Output overlaps protected source/data/models/checkpoint location: " + str(path))
    require(not output_dir.exists(), "Choose a NEW output directory; never overwrite existing results")
    entries, ref_source, ref_alias = load_contract()
    rows, evidence = collect(roots, entries)
    registry_path = registry_path or ROOT / "provenance/artifact_manifest.json"
    rows.append(reference_row(data_root, registry_path, ref_source, ref_alias))
    require(len(rows) == EXPECTED_ROLLOUT_COUNT + 1 and len({r["relative_path"] for r in rows}) == len(rows),
            "Fresh verifier layout must contain exactly 83 unique inputs")
    # Everything is validated before this first write. Preserve partial output on
    # interruption for inspection; a subsequent invocation must choose a new root.
    output_dir.mkdir(parents=True, exist_ok=False)
    for row in rows:
        destination = output_dir.joinpath(*relative(row["relative_path"]).parts)
        no_links(row["source"])
        no_links(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with row["source"].open("rb") as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, 8 * 1024 * 1024)
        require(destination.stat().st_size == row["bytes"] and digest(destination) == row["sha256"],
                "Copied bytes changed; preserve new output directory for inspection")
    receipt = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
               "input_root": str(output_dir), "checkpoint_roots": [str(r) for r in roots], "data_root": str(data_root),
               "input_files": len(rows), "fresh_rollout_files": len(entries), "original_reference_files": 1,
               "mode": "independent_byte_copies_no_links", "fresh_outputs_match_completed_runner_receipts": True,
               "reference_matches_artifact_registry": True, "historical_prediction_hashes_compared": False,
               "remote_verification_performed": False, "numeric_verification_performed": False,
               "scope": "Fresh output identity/count checks only. The unchanged original paper verifier must separately evaluate metrics and may fail on numerical differences; no expected values were weakened.",
               "artifact_manifest_sha256": digest(registry_path), "evidence": evidence,
               "files": [{k: str(v) if isinstance(v, Path) else v for k, v in r.items()} for r in rows]}
    with (output_dir / RECEIPT_NAME).open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2); handle.write("\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, action="append", required=True,
                        help="Repeat for disjoint runner output roots; indices/receipts are scanned recursively")
    parser.add_argument("--data-root", type=Path, required=True, help="Installed SN_Research logical bundle root")
    parser.add_argument("--output-dir", type=Path, required=True, help="NEW private verifier-input directory")
    parser.add_argument("--models-root", type=Path, help="Optional separate model/cache root to explicitly protect")
    args = parser.parse_args(argv)
    try:
        receipt = materialize(args.checkpoint_root, args.data_root, args.output_dir, models_root=args.models_root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"passed": False, "error": type(error).__name__ + ": " + str(error)}), file=sys.stderr)
        return 1
    print(json.dumps({"passed": True, "input_files": receipt["input_files"], "fresh_rollout_files": receipt["fresh_rollout_files"],
                      "input_root": receipt["input_root"], "materialization_receipt": str(Path(receipt["input_root"]) / RECEIPT_NAME),
                      "historical_prediction_hashes_compared": False, "numeric_verification_performed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
