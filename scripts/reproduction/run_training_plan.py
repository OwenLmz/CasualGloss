#!/usr/bin/env python3
"""Inspect or execute the thirty recorded final-paper training invocations.

DEFAULT: dry-run, no files written and no model loaded. Training is expensive:
it requires --execute. New runs cannot overwrite source, data, cached models or
existing job directories. This runner never edits historical results.

--initialization-source new follows newly trained upstream adapter dependencies;
archived retrains an individual stage from the recorded upstream adapter only.
Neither mode guarantees bit-identical new weights or paper results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from experiment_matrix import ROOT, MODEL_REVISION, check_snapshot, data_path, model_snapshot, protected_output, relative_path, sha256

DEFAULT_PLAN = ROOT / "configs/reproduction/training.json"


def load_training(path=DEFAULT_PLAN):
    plan = json.loads(path.read_text(encoding="utf-8"))
    rows = plan["runs"]
    if len(rows) != 30 or len({row["id"] for row in rows}) != 30:
        raise ValueError("The final-paper plan must contain exactly thirty unique training invocations")
    known = {row["id"] for row in rows}
    for row in rows:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", row["id"]):
            raise ValueError("Unsafe training ID")
        if row["world_size"] not in (1, 2, 4) or row["launcher"] not in ("python", "torchrun"):
            raise ValueError("Missing recorded training world size/launcher")
        if set(row["dependencies"]) - known:
            raise ValueError("Unresolved dependency in training plan")
        for value in [row["config"], row["script"], *row["inputs"].values()]:
            relative_path(value)
        if sha256(ROOT / row["config"]) != row["config_sha256"]:
            raise ValueError(f"Training config differs from recorded plan: {row['config']}")
        config = json.loads((ROOT / row["config"]).read_text(encoding="utf-8"))
        if config["model_revision"] != MODEL_REVISION or config["seed"] != row["seed"]:
            raise ValueError("Training model revision/seed mismatch")
        micro = config.get("micro_batch_size_per_gpu", config.get("micro_batch_size"))
        effective = row["world_size"] * micro * config["gradient_accumulation_steps"]
        if effective != row["training_budget"]["effective_batch_size"]:
            raise ValueError("Training batch composition does not match recorded invocation")
    return plan


def select_training(plan, ids=(), families=(), include_dependencies=False):
    rows = plan["runs"]
    mapping = {row["id"]: row for row in rows}
    if set(ids) - set(mapping):
        raise ValueError(f"Unknown training ID: {sorted(set(ids)-set(mapping))}")
    known_families = {row["family"] for row in rows}
    if set(families) - known_families:
        raise ValueError("Unknown training family")
    selected = {row["id"] for row in rows if (not ids or row["id"] in ids) and (not families or row["family"] in families)}
    if not selected:
        raise ValueError("Training selection is empty")
    if include_dependencies:
        pending = list(selected)
        while pending:
            for dependency in mapping[pending.pop()]["dependencies"]:
                if dependency not in selected:
                    selected.add(dependency); pending.append(dependency)
    return [row for row in rows if row["id"] in selected]


def source_adapter(row, data_root, output_root, initialization_source):
    initial = row["initialization"]
    if initial["base_only"]:
        return None
    if initialization_source == "archived":
        return data_path(data_root, initial["archived_adapter_relative"])
    return output_root / initial["new_training_dependency"] / "outputs" / initial["new_dependency_checkpoint"]


def training_invocation(row, data_root, models_root, output_root, initialization_source="new", python=sys.executable):
    job = output_root / row["id"]
    config = json.loads((ROOT / row["config"]).read_text(encoding="utf-8"))
    config["model_id"] = str(model_snapshot(models_root))
    config_path = job / "runtime_config.json"
    arguments = ["--config", str(config_path)]
    for name, relative in row["inputs"].items():
        arguments += ["--" + name.replace("_", "-"), str(data_path(data_root, relative))]
    adapter = source_adapter(row, data_root, output_root, initialization_source)
    if adapter:
        arguments += ["--source-adapter-dir", str(adapter)]
    arguments += ["--output-dir", str(job / "outputs")]
    command = [python]
    if row["launcher"] == "torchrun":
        command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={row['world_size']}"]
    command += [str(ROOT / row["script"]), *arguments]
    post = []
    aux = {}
    if row["selection"]["kind"] == "development_best_vs_final":
        select_inputs = row["selection"]["inputs"]
        guard_config = json.loads((ROOT / "configs/method/qwen2_5_7b_lora_state_recovery_v2_3_full.json").read_text(encoding="utf-8"))
        guard_config["model_id"] = str(model_snapshot(models_root))
        guard_path = job / "guard_runtime_config.json"
        aux[str(guard_path)] = guard_config
        for candidate in ("guard", "best", "final"):
            post.append([python, str(ROOT / "method/causalgloss/evaluate_free_running_rollout.py"),
                         "--config", str(guard_path if candidate == "guard" else config_path),
                         "--parent-splits", str(data_path(data_root, select_inputs["parent_splits"])),
                         "--causal-pairs", str(data_path(data_root, select_inputs["causal_pairs"])),
                         "--adapter-dir", str(adapter if candidate == "guard" else job / "outputs" / f"adapter_{candidate}"),
                         "--output-dir", str(job / "selection_outputs" / candidate),
                         "--stage", f"{row['id']}_{candidate}_development", "--split", "v2_3_development",
                         "--device", "cuda:0", "--generation-batch-size", "32"])
        post.append([python, str(ROOT / "method/causalgloss/assess_screen_gate.py"),
                     "--baseline", str(job / "selection_outputs/guard/rollout_metrics.json"),
                     "--candidate", "best=" + str(job / "selection_outputs/best/rollout_metrics.json"),
                     "--candidate", "final=" + str(job / "selection_outputs/final/rollout_metrics.json"),
                     "--output", str(job / "development_selection.json")])
    elif row["selection"]["kind"] == "learned_lr_screen":
        name = Path(row["config"]).stem
        post.append([python, str(ROOT / "baselines/transllama/evaluate_transllama.py"),
                     "--config", str(config_path), "--pairs", str(data_path(data_root, row["selection"]["validation_pairs"])),
                     "--split", "validation", "--adapter-dir", str(job / "outputs/adapter_best"),
                     "--output-dir", str(job / "screen_validation"), "--stage", name + "-validation",
                     "--k-values", "1,2,3,4,5,6,7,8", "--generation-batch-size", "32", "--action-max-new-tokens", "16"])
    return {"id": row["id"], "job_dir": str(job), "runtime_config": str(config_path), "runtime_config_contents": config,
            "auxiliary_configs": aux, "command": command, "post_training_selection_commands": post,
            "initialization_source": initialization_source, "initialization_adapter": str(adapter) if adapter else None,
            "recorded_budget": row["training_budget"], "observed_training_not_a_new_target": row["observed_training"],
            "selection": row["selection"], "row": row}


def environment_for(script, nltk_data=None):
    environment = os.environ.copy()
    environment.update({"PYTHONPATH": str(Path(script).parent), "PYTHONDONTWRITEBYTECODE": "1",
                        "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    if nltk_data:
        environment["NLTK_DATA"] = str(nltk_data.resolve())
    return environment


def verify_data(rows, data_root):
    registry = {item["relative_to_remote_root"]: item for item in json.loads((ROOT/"provenance/artifact_manifest.json").read_text(encoding="utf-8"))["entries"]}
    requirements = {}
    for row in rows:
        requirements.update({relative: row["input_sha256"][name] for name, relative in row["inputs"].items()})
        for value in row["selection"].get("inputs", {}).values():
            requirements[value] = registry[value]["sha256"]
        if row["selection"].get("validation_pairs"):
            value = row["selection"]["validation_pairs"]
            requirements[value] = registry[value]["sha256"]
    errors = []
    for relative, expected in requirements.items():
        target = data_path(data_root, relative)
        if not target.is_file() or sha256(target) != expected:
            errors.append(f"Missing/unverified registered training input: {target}")
    return errors


def verify_archived_initializations(rows, data_root):
    registry = {item["relative_to_remote_root"]: item for item in json.loads((ROOT/"provenance/artifact_manifest.json").read_text(encoding="utf-8"))["entries"]}
    errors = []
    for relative in sorted({row["initialization"]["archived_adapter_relative"] for row in rows if not row["initialization"]["base_only"]}):
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            path = relative + "/" + filename
            target = data_path(data_root,path)
            registered = registry.get(path)
            if not registered or not registered.get("sha256") or not target.is_file() or sha256(target) != registered["sha256"]:
                errors.append(f"Archived initialization missing/unverified: {target}")
    return errors


def prepare_lr_selection(output_root, python, nltk_data):
    """Feed NEW summaries to the unchanged frozen LR selector; never retune silently."""
    selection = output_root / "learning_rate_selection.json"
    if selection.exists():
        return json.loads(selection.read_text(encoding="utf-8"))
    source = output_root / "learning_rate_selection_inputs"
    source.mkdir(exist_ok=False)
    for lr in ("1e-5", "3e-5", "6e-5"):
        name = f"screen_lr_{lr}"
        job = output_root / f"transllama_{name}"
        for original, copied in ((job/"outputs/training_summary.json", source/"lr_screen"/name/"training_summary.json"),
                                 (job/"screen_validation/sweep_summary.json", source/"lr_screen_dev"/name/"sweep_summary.json")):
            if not original.is_file():
                raise FileNotFoundError(f"New validation-only LR screen is incomplete: {original}")
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, copied)
    script = ROOT / "baselines/transllama/select_learning_rate.py"
    subprocess.run([python, str(script), "--runs-root", str(source), "--configs-root", str(ROOT/"configs/transllama"),
                    "--output", str(selection)], env=environment_for(script, nltk_data), check=True)
    return json.loads(selection.read_text(encoding="utf-8"))


def execute_training(plan, output_root, python, nltk_data):
    row = plan["row"]
    if row["selection"]["kind"] == "learned_final_after_screen" and plan["initialization_source"] == "new":
        selected = prepare_lr_selection(output_root, python, nltk_data)["selected"]
        selected_config = json.loads((ROOT/"configs/transllama"/(selected+".json")).read_text(encoding="utf-8"))
        if selected_config["learning_rate"] != plan["runtime_config_contents"]["learning_rate"]:
            raise RuntimeError("New LR screen selected a different rate than the published 6e-5; stop and report, do not rewrite paper settings")
    adapter = Path(plan["initialization_adapter"]) if plan["initialization_adapter"] else None
    if adapter and (not (adapter/"adapter_model.safetensors").is_file() or not (adapter/"adapter_config.json").is_file()):
        raise FileNotFoundError(f"Initialization adapter missing; run required upstream dependency first: {adapter}")
    if adapter and plan["initialization_source"] == "new":
        dependency = row["initialization"]["new_training_dependency"]
        receipt_path = output_root/dependency/"training_receipt.json"
        dependency_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (dependency_receipt.get("id") != dependency or dependency_receipt.get("status") != "completed"
                or dependency_receipt.get("fresh_training_executed") is not True
                or Path(dependency_receipt["selected_adapter"]).resolve() != adapter.resolve()):
            raise ValueError("New-training initialization must be the validated selected checkpoint of its declared dependency")
        for name in ("adapter_config.json","adapter_model.safetensors"):
            if sha256(adapter/name) != dependency_receipt.get("selected_adapter_sha256",{}).get(name):
                raise ValueError(f"New-training initialization changed after its receipt: {adapter/name}")
    job = Path(plan["job_dir"])
    job.mkdir(parents=True, exist_ok=False)
    for path, content in {plan["runtime_config"]: plan["runtime_config_contents"], **plan["auxiliary_configs"]}.items():
        Path(path).write_text(json.dumps(content, indent=2)+"\n", encoding="utf-8")
    (job/"invocation.json").write_text(json.dumps(plan, indent=2)+"\n",encoding="utf-8")
    started = datetime.now(timezone.utc).isoformat()
    commands = [plan["command"], *plan["post_training_selection_commands"]]
    returncode = 0
    with (job/"execution.log").open("w",encoding="utf-8") as log:
        for command in commands:
            script = command[command.index(f"--nproc_per_node={row['world_size']}") + 1] if "torch.distributed.run" in command else command[1]
            completed = subprocess.run(command, env=environment_for(script, nltk_data), stdout=log, stderr=subprocess.STDOUT, check=False)
            returncode = completed.returncode
            if returncode:
                break
    checkpoint = "adapter_best"
    if not returncode and row["selection"]["kind"] == "development_best_vs_final":
        choice = json.loads((job/"development_selection.json").read_text(encoding="utf-8"))["selected_candidate"]
        if choice not in ("best", "final"):
            raise ValueError("Unknown development checkpoint selection")
        checkpoint = "adapter_" + choice
    selected_adapter = job / "outputs" / checkpoint
    successful = not returncode and (selected_adapter/"adapter_model.safetensors").is_file()
    receipt = {"id": row["id"], "status": "completed" if successful else "failed", "returncode": returncode,
               "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
               "selected_adapter": str(selected_adapter), "selection_rule": row["selection"]["rule"],
               "historical_selected_checkpoint_for_comparison_only": row["selection"]["historical_selected_checkpoint"],
               "fresh_training_executed": True, "bitwise_equality_not_assumed": True}
    if successful:
        receipt["selected_adapter_sha256"] = {name:sha256(selected_adapter/name) for name in ("adapter_config.json","adapter_model.safetensors")}
    (job/"training_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(receipt),flush=True)
    return successful


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    p.add_argument("--list", action="store_true")
    action = p.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true", help="Explicitly authorize expensive model training in this invocation")
    action.add_argument("--dry-run", action="store_true", help="Default behavior; no files written")
    p.add_argument("--id", action="append", default=[])
    p.add_argument("--family", action="append", default=[])
    p.add_argument("--with-dependencies", action="store_true")
    p.add_argument("--initialization-source", choices=("new", "archived"), default="new")
    p.add_argument("--data-root", type=Path)
    p.add_argument("--models-root", type=Path)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--nltk-data", type=Path)
    args = p.parse_args()
    rows = select_training(load_training(args.plan), args.id, args.family, args.with_dependencies)
    if args.list:
        print(json.dumps(rows,indent=2)); return 0
    if any(value is None for value in (args.data_root,args.models_root,args.output_root)):
        p.error("--data-root, --models-root, --output-root are required outside --list")
    output = protected_output(args.output_root,args.data_root,args.models_root)
    plans = [training_invocation(row,args.data_root,args.models_root,output,args.initialization_source,args.python) for row in rows]
    if not args.execute:
        print(json.dumps({"mode":"dry_run_no_files_written_no_training", "training_run_count":len(plans),"plans":plans},indent=2)); return 0
    unresolved = [row["id"] for row in rows if row["status"] != "resolved" or row["unresolved"]]
    if unresolved:
        p.error(f"Refusing unresolved training invocations: {unresolved}")
    errors = check_snapshot(model_snapshot(args.models_root)) + verify_data(rows,args.data_root)
    if args.initialization_source == "archived":
        errors += verify_archived_initializations(rows,args.data_root)
    errors += ["Refusing existing job: "+plan["job_dir"] for plan in plans if Path(plan["job_dir"]).exists()]
    if errors:
        print(json.dumps({"status":"preflight_failed","errors":errors},indent=2)); return 2
    output.mkdir(parents=True,exist_ok=True)
    for plan in plans:
        if not execute_training(plan,output,args.python,args.nltk_data):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
