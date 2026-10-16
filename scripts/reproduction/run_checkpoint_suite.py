#!/usr/bin/env python3
"""Rerun selected final-paper checkpoints into NEW outputs only.

Examples:
  python scripts/reproduction/run_checkpoint_suite.py --list
  python scripts/reproduction/run_checkpoint_suite.py --family locked500 --dry-run \
    --data-root /private/bundle --models-root /private/hf_cache --output-root /new/rerun

This does not retrain checkpoints or replace archived predictions. Speech entries
rerun the policy on the recorded shared frontend, not TTS/Seamless inference.
Timing measured on different hardware is not expected to be identical.
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

from experiment_matrix import (DEFAULT_MATRIX, EXPECTED_COUNTS, MODEL_REVISION, ROOT,
                               adapter_hashes, check_snapshot, data_path, load_matrix,
                               model_snapshot, protected_output, select_entries, sha256)


def resolved_adapter(entry, data_root, training_root=None):
    if training_root is None:
        return data_path(data_root,entry["adapter_relative"]), {"mode":"archived_registered_adapter"}
    root = training_root.resolve()
    receipt_path = root / entry["training_run_id"] / "training_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("id") != entry["training_run_id"] or receipt.get("status") != "completed" or receipt.get("fresh_training_executed") is not True:
        raise ValueError(f"Incomplete/untrusted new-training receipt: {receipt_path}")
    adapter = Path(receipt["selected_adapter"]).resolve()
    if root not in adapter.parents:
        raise ValueError("A new-training receipt cannot point outside its independent training output root")
    fingerprints = {}
    for name in ("adapter_config.json","adapter_model.safetensors"):
        target = adapter/name
        expected = receipt.get("selected_adapter_sha256",{}).get(name)
        if not target.is_file() or expected != sha256(target):
            raise ValueError(f"New-training selected checkpoint is missing or has changed: {target}")
        fingerprints[name] = expected
    return adapter, {"mode":"new_training_receipt", "receipt":str(receipt_path), "receipt_sha256":sha256(receipt_path),
                     "adapter":str(adapter),"adapter_sha256":fingerprints,
                     "historical_adapter_relative_for_comparison_only":entry["adapter_relative"]}


def make_plan(entry: dict, data_root: Path, models_root: Path, output_root: Path,
              python: str = sys.executable, device: str = "cuda:0", training_root: Path | None = None) -> dict:
    job = output_root / entry["id"]
    cfg = ROOT / entry["config"]
    source_config = json.loads(cfg.read_text(encoding="utf-8"))
    if source_config["model_revision"] != MODEL_REVISION or source_config["seed"] != entry["seed"]:
        raise ValueError(f"Config revision/seed mismatch for {entry['id']}")
    local_config = dict(source_config)
    local_config["model_id"] = str(model_snapshot(models_root))
    runtime_config = job / "runtime_config.json"
    result_dir = job / "outputs"
    adapter, adapter_origin = resolved_adapter(entry,data_root,training_root)
    common = ["--config", str(runtime_config), "--adapter-dir", str(adapter)]
    inputs = {name: data_path(data_root, value) for name, value in entry["inputs"].items()}
    kind = entry["kind"]
    if kind in ("action", "ablation"):
        args = common + ["--parent-splits", str(inputs["parent_splits"]), "--causal-pairs", str(inputs["causal_pairs"]),
                         "--output-dir", str(result_dir), "--stage", entry["stage"], "--split", entry["split"],
                         "--device", device, "--generation-batch-size", str(entry["generation_batch_size"])]
        result = result_dir / "sentence_rollouts.jsonl"
    elif kind == "learned":
        args = common + ["--pairs", str(inputs["pairs"]), "--split", "test", "--allow-locked-test",
                         "--output-dir", str(result_dir), "--stage", entry["stage"], "--k-values", str(entry["k"]),
                         "--generation-batch-size", str(entry["generation_batch_size"]), "--action-max-new-tokens", "16"]
        result = result_dir / f"k{entry['k']}" / "sentence_rollouts.jsonl"
    elif kind.startswith("epee_"):
        args = common + ["--manifest", str(inputs["manifest"]), "--output-dir", str(result_dir),
                         "--stage", entry["stage"], "--expected-count", "401",
                         "--generation-batch-size", str(entry["generation_batch_size"])]
        if kind == "epee_action":
            args += ["--system-family", "causalgloss"]
        if kind == "epee_learned":
            args += ["--k-values", str(entry["k"]), "--action-max-new-tokens", "16"]
            result = result_dir / f"k{entry['k']}" / "sentence_rollouts.jsonl"
        else:
            result = result_dir / "sentence_rollouts.jsonl"
    elif kind == "speech":
        args = common + ["--frontend-jsonl", str(inputs["frontend"]),
                         "--base-model-path", str(model_snapshot(models_root)),
                         "--output-jsonl", str(result_dir / "sentence_rollouts.jsonl"),
                         "--output-report", str(result_dir / "speech_report.json"),
                         "--device", device, "--max-post-source-actions", "64"]
        result = result_dir / "sentence_rollouts.jsonl"
    else:
        raise ValueError(f"Unknown experiment kind: {kind}")
    return {"id": entry["id"], "family": entry["family"], "job_dir": str(job),
            "command": [python, str(ROOT / "scripts/paper.py"), "run", entry["entrypoint"], *args],
            "runtime_config": str(runtime_config), "runtime_config_contents": local_config,
            "original_config_sha256": sha256(cfg), "rewritten_config_fields": ["model_id"],
            "inputs": {name: str(value) for name, value in inputs.items()},
            "result": str(result), "historical_reference": str(data_path(data_root, entry["historical_output_relative"])),
            "precision": entry["precision"], "generation_batch_size": entry["generation_batch_size"],
            "expected_count": entry["expected_count"], "entry": entry, "adapter_origin":adapter_origin}


def coalesce_plans(plans: list[dict]) -> list[dict]:
    """Reuse one loaded learned checkpoint for its selected k values.

    The original evaluators themselves reset states per k and emit k-specific
    subdirectories. This is their historical sweep mode, not result synthesis.
    """
    grouped, output = {}, []
    for plan in plans:
        entry = plan["entry"]
        if entry["kind"] not in ("learned", "epee_learned"):
            plan["members"] = [dict(plan)]
            output.append(plan)
            continue
        key = (entry["family"], entry["kind"], entry["seed"], entry["config"], entry["adapter_relative"])
        grouped.setdefault(key, []).append(plan)
    for members in grouped.values():
        members.sort(key=lambda item: item["entry"]["k"])
        plan = dict(members[0])
        first = plan["entry"]
        prefix = "locked" if first["family"] == "locked500" else "clerc"
        plan["id"] = f"{prefix}_transllama_{first['seed']}_sweep"
        job = Path(plan["job_dir"]).parent / plan["id"]
        old_config, old_outputs = plan["runtime_config"], str(Path(plan["job_dir"]) / "outputs")
        plan["job_dir"], plan["runtime_config"] = str(job), str(job / "runtime_config.json")
        command = [str(job / "runtime_config.json") if part == old_config else
                   str(job / "outputs") if part == old_outputs else part for part in plan["command"]]
        command[command.index("--k-values") + 1] = ",".join(str(item["entry"]["k"]) for item in members)
        plan["command"] = command
        for member in members:
            member["result"] = str(job / "outputs" / f"k{member['entry']['k']}" / "sentence_rollouts.jsonl")
        plan["result"] = members[0]["result"]
        plan["members"] = members
        output.append(plan)
    return output


def validate_plan(plan: dict, data_root: Path, models_root: Path, verify_hashes=True) -> list[str]:
    errors = check_snapshot(model_snapshot(models_root))
    job = Path(plan["job_dir"])
    if job.exists():
        errors.append(f"Refusing to overwrite an existing experiment directory: {job}")
    manifest = json.loads((ROOT / "provenance/artifact_manifest.json").read_text(encoding="utf-8"))
    for relative in plan["entry"]["inputs"].values():
        target = data_path(data_root, relative)
        if not target.is_file():
            errors.append(f"Missing input: {target}")
        elif verify_hashes:
            matches = [row for row in manifest["entries"] if row["relative_to_remote_root"] == relative]
            if len(matches) != 1 or not matches[0].get("sha256"):
                errors.append(f"Input has no unique registered SHA-256: {relative}")
            elif sha256(target) != matches[0]["sha256"]:
                errors.append(f"Input SHA-256 mismatch: {target}")
    if plan["adapter_origin"]["mode"] == "archived_registered_adapter":
        for item in adapter_hashes(plan["entry"], data_root, manifest):
            if not item["exists"] or item.get("error") or item["matches_sha256"] is not True:
                errors.append(f"Selected adapter unavailable or not hash-verified: {item['relative']}")
    else:
        origin = plan["adapter_origin"]
        if sha256(Path(origin["receipt"])) != origin["receipt_sha256"]:
            errors.append("New-training receipt changed after planning")
        for name,expected in origin["adapter_sha256"].items():
            target = Path(origin["adapter"])/name
            if not target.is_file() or sha256(target) != expected:
                errors.append(f"New-training checkpoint changed after planning: {target}")
    return errors


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    p.add_argument("--list", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--family", action="append", choices=sorted(EXPECTED_COUNTS), default=[])
    p.add_argument("--id", action="append", default=[])
    p.add_argument("--seed", type=int, action="append", default=[], help="Select evaluation-config RNG seed(s) within family/ID selection")
    p.add_argument("--separate-k", action="store_true", help="Load learned checkpoint separately per k; default reuses historical k-sweep")
    p.add_argument("--data-root", type=Path, help="Installed bundle root corresponding to original SN_Research")
    p.add_argument("--models-root", type=Path, help="HF_HOME (with hub/) or exact recorded snapshot")
    p.add_argument("--output-root", type=Path, help="A NEW output location outside source/data/model trees")
    p.add_argument("--python", default=sys.executable, help="Policy environment Python; no automatic installation")
    p.add_argument("--training-root",type=Path,help="Optional independently completed new-training root; validate receipts/hashes instead of using archived adapters")
    p.add_argument("--device", default="cuda:0", help="Use CUDA_VISIBLE_DEVICES to choose physical GPU")
    p.add_argument("--nltk-data", type=Path, help="Optional installed NLTK resource root, preserved in child process")
    return p


def main():
    p = parser()
    args = p.parse_args()
    rows = select_entries(load_matrix(args.matrix), args.family, args.id)
    if args.seed:
        rows = [row for row in rows if row["seed"] in args.seed]
        if not rows:
            p.error("The family/ID/seed selection is empty")
    if args.list:
        print(json.dumps(rows, indent=2)); return 0
    if any(value is None for value in (args.data_root, args.models_root, args.output_root)):
        p.error("--data-root, --models-root and --output-root are required outside --list")
    try:
        output = protected_output(args.output_root, args.data_root, args.models_root)
        if args.training_root:
            trained = args.training_root.resolve()
            if output == trained or trained in output.parents or output in trained.parents:
                p.error("Inference outputs must not overlap an independent new-training output root")
        plans = [make_plan(row, args.data_root, args.models_root, output, args.python, args.device,args.training_root) for row in rows]
        if not args.separate_k:
            plans = coalesce_plans(plans)
        else:
            for plan in plans:
                plan["members"] = [dict(plan)]
    except ValueError as error:
        p.error(str(error))
    if args.dry_run:
        print(json.dumps({"mode": "dry_run_no_files_written", "checkpoint_invocations": len(plans),
                          "rollout_count": len(rows), "plans": plans}, indent=2)); return 0
    # Validate ALL selected jobs before any model invocation. A typo cannot start
    # a partial run or let an unregistered checkpoint silently substitute.
    failures = {plan["id"]: errors for plan in plans
                if (errors := validate_plan(plan, args.data_root, args.models_root))}
    if failures:
        print(json.dumps({"status": "preflight_failed", "failures": failures}, indent=2)); return 2
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "TOKENIZERS_PARALLELISM": "false"})
    if args.nltk_data:
        env["NLTK_DATA"] = str(args.nltk_data.resolve())
    output.mkdir(parents=True, exist_ok=True)
    selection_hash = hashlib.sha256("\n".join(sorted(row["id"] for row in rows)).encode()).hexdigest()[:12]
    index_path = output / f"checkpoint_output_index_{selection_hash}.json"
    index = [{"id": member["entry"]["id"], "family": member["family"], "result": member["result"],
              "verification_local_relative": member["entry"]["verification_local_relative"],
              "historical_output_relative": member["entry"]["historical_output_relative"]}
             for plan in plans for member in plan["members"]]
    with index_path.open("x", encoding="utf-8") as handle:
        json.dump({"results_are_new_inference": True, "rollout_count": len(index), "outputs": index}, handle, indent=2)
        handle.write("\n")
    for plan in plans:
        job = Path(plan["job_dir"])
        job.mkdir(parents=True, exist_ok=False)
        Path(plan["runtime_config"]).write_text(json.dumps(plan["runtime_config_contents"], indent=2) + "\n", encoding="utf-8")
        (job / "invocation.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        started = datetime.now(timezone.utc).isoformat()
        with (job / "execution.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(plan["command"], env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        member_receipts = []
        for member in plan["members"]:
            target = Path(member["result"])
            record_count = sum(1 for line in target.open(encoding="utf-8") if line.strip()) if target.is_file() else 0
            member_receipts.append({"id": member["entry"]["id"], "result": str(target), "record_count": record_count,
                                    "expected_count": member["expected_count"], "sha256": sha256(target) if target.is_file() else None,
                                    "status": "completed" if completed.returncode == 0 and record_count == member["expected_count"] else "failed"})
        receipt = {"id": plan["id"], "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                   "returncode": completed.returncode, "members": member_receipts,
                   "status": "completed" if all(item["status"] == "completed" for item in member_receipts) else "failed",
                   "result_is_fresh_inference_not_archived_copy": True}
        (job / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt), flush=True)
        if receipt["status"] != "completed":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
