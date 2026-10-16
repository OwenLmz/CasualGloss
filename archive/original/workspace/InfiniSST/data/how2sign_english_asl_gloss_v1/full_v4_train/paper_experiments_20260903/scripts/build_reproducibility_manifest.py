#!/usr/bin/env python3
"""Create a reproducibility manifest for the frozen paper experiment package."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()

    relative_files = [
        "data/how2sign_english_asl_gloss_v1/full_v4_train/state_recovery_v2_3_full/configs/qwen2_5_7b_lora_state_recovery_v2_3_full.json",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/state_recovery_v2_3_full/runs/qwen2_5_7b_lora_state_recovery_v2_3_full/run_metadata.json",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/state_recovery_v2_3_full/runs/qwen2_5_7b_lora_state_recovery_v2_3_full/training_summary.json",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/state_recovery_v2_3_full/runs/qwen2_5_7b_lora_state_recovery_v2_3_full/adapter_final/adapter_model.safetensors",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/state_recovery_v2_3_full/reports/frozen_method_protocol.json",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/state_recovery_v2_3_full/reports/full_development_selection.json",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/state_recovery_v2_3_full/final_evaluation_v2_3/reports/locked500_v2_3_comparison.json",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/state_recovery_v2_3_full/final_evaluation_v2_3/reports/clerc_epee_v2_3_comparison.json",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/transllama_style_wait_k_inference_v1/reports/summary.tsv",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/transllama_style_wait_k_inference_v1/reports/comparison.json",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/paper_experiments_20260903/reports/v2_3_vs_waitk_v1_k1.json",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/paper_experiments_20260903/scripts/compare_v2_3_vs_waitk.py",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/paper_experiments_20260903/scripts/build_paper_artifacts.py",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/paper_experiments_20260903/scripts/select_v2_3_cases.py",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/paper_experiments_20260903/scripts/build_reproducibility_manifest.py",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/paper_experiments_20260903/EXPERIMENT_COMPLETION_REPORT.md",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/paper_experiments_20260903/cases/case_selection_summary.json",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/paper_experiments_20260903/cases/selected_v2_3_vs_v1_cases.jsonl",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/paper_experiments_20260903/artifacts/paper_results_summary.json",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/paper_experiments_20260903/artifacts/tables/table_main_locked500.tsv",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/paper_experiments_20260903/artifacts/tables/table_modified_waitk_sweep.tsv",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/paper_experiments_20260903/artifacts/tables/table_ablation_confirmation250.tsv",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/paper_experiments_20260903/artifacts/tables/table_epee_external401.tsv",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/paper_experiments_20260903/artifacts/tables/table_synthetic_mandarin_e2e100.tsv",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/paper_experiments_20260903/artifacts/tables/table_key_paired_effects.tsv",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/paper_experiments_20260903/artifacts/figures/fig_quality_latency_frontier.png",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/paper_experiments_20260903/artifacts/figures/fig_quality_latency_frontier.pdf",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/paper_experiments_20260903/artifacts/figures/fig_quality_latency_frontier.svg",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/paper_experiments_20260903/artifacts/figures/fig_paired_effects.png",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/paper_experiments_20260903/artifacts/figures/fig_paired_effects.pdf",
        "data/how2sign_english_asl_gloss_v1/full_v4_train/paper_experiments_20260903/artifacts/figures/fig_paired_effects.svg",
        "data/how2sign_english_asl_gloss_v1/chinese_speech_frontend_v0/scripts/run_speech_to_gloss_e2e.py",
        "data/how2sign_english_asl_gloss_v1/chinese_speech_frontend_v0/scripts/compare_causal_e2e_versions.py",
        "data/how2sign_english_asl_gloss_v1/chinese_speech_frontend_v0/outputs/task5_pilot_100.seamlessstreaming.jsonl",
        "data/how2sign_english_asl_gloss_v1/chinese_speech_frontend_v0/paper_e2e_20260903/v1.jsonl",
        "data/how2sign_english_asl_gloss_v1/chinese_speech_frontend_v0/paper_e2e_20260903/v2_3.jsonl",
        "data/how2sign_english_asl_gloss_v1/chinese_speech_frontend_v0/paper_e2e_20260903/reports/v2_3_vs_v1.json",
    ]
    missing = [relative for relative in relative_files if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required reproducibility assets: {missing}")

    run_metadata_path = root / relative_files[1]
    training_metadata = read_json(run_metadata_path)
    waitk_pair = read_json(
        root
        / "data/how2sign_english_asl_gloss_v1/full_v4_train/paper_experiments_20260903/reports/v2_3_vs_waitk_v1_k1.json"
    )
    e2e_pair = read_json(
        root
        / "data/how2sign_english_asl_gloss_v1/chinese_speech_frontend_v0/paper_e2e_20260903/reports/v2_3_vs_v1.json"
    )
    current_gpus = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    git_commit = command_output(["git", "-C", str(root), "rev-parse", "HEAD"])
    current_versions: dict[str, str | None] = {}
    for package in ("torch", "transformers", "peft", "matplotlib"):
        try:
            module = __import__(package)
            current_versions[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            current_versions[package] = None

    file_records = {}
    for relative in relative_files:
        path = root / relative
        record: dict[str, Any] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
        if path.suffix == ".jsonl":
            record["nonempty_line_count"] = line_count(path)
        file_records[relative] = record

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "frozen_paper_experiment_reproducibility_manifest",
        "project_root": str(root),
        "git_commit": git_commit,
        "method": {
            "model_id": training_metadata["config"]["model_id"],
            "model_revision": training_metadata["config"]["model_revision"],
            "dataset_version": training_metadata["config"]["dataset_version"],
            "seed": training_metadata["config"]["seed"],
            "training_gpu_names": training_metadata["gpu_names"],
            "world_size": training_metadata["world_size"],
            "effective_batch_size": training_metadata["effective_batch_size"],
            "precision": training_metadata["config"]["precision"],
            "learning_rate": training_metadata["config"]["learning_rate"],
            "epochs": training_metadata["config"]["epochs"],
            "software_versions": training_metadata["versions"],
            "data_split_sha256": training_metadata["sha256"],
        },
        "current_aggregation_environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "gpus": current_gpus.splitlines() if current_gpus else None,
            "software_versions": current_versions,
        },
        "statistics": {
            "locked_waitk_pair": waitk_pair["bootstrap"],
            "speech_e2e_pair": {
                "method": "paired_parent_level_percentile_bootstrap",
                "repetitions": e2e_pair["bootstrap_samples"],
                "seed": e2e_pair["seed"],
            },
            "quality_noninferiority_margin": waitk_pair[
                "quality_noninferiority_margin"
            ],
        },
        "data_governance": {
            "locked_test_used_for_selection": False,
            "epee_used_for_selection": False,
            "qualitative_cases_selected_after_aggregate_metrics_frozen": True,
        },
        "files": file_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "file_count": len(file_records),
                "training_gpus": manifest["method"]["training_gpu_names"],
                "current_gpus": manifest["current_aggregation_environment"]["gpus"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
