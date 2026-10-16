#!/usr/bin/env python3
"""Write hashes and runtime metadata for the final v2.2 experiment artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
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


def package_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = args.project_root / "data/how2sign_english_asl_gloss_v1"
    train = data / "full_v4_train"
    semantic = train / "semantic_causal_alignment_v2_2_full"
    fine_tuning = train / "fine_tuning_v2_semantic_v2_2_full"
    audit = train / "causal_alignment_latency_audit_v2_2"
    local_agreement = train / "local_agreement_causal_v2_3"
    frontend = data / "chinese_speech_frontend_v0"
    primary = fine_tuning / "runs/qwen2_5_7b_lora_semantic_causal_v2_2_1_treebank_full"
    recovery = fine_tuning / "runs/qwen2_5_7b_lora_semantic_causal_v2_2_2_recovery_full"

    paths = [
        semantic / "configs/semantic_causal_alignment_v2_2_full.json",
        semantic / "scripts/build_confident_burst_alignment_v2_2.py",
        semantic / "scripts/split_and_validate_v2_2.py",
        semantic / "scripts/audit_split_leakage.py",
        semantic / "scripts/repair_action_prompt_detokenization.py",
        semantic / "reports/semantic_alignment_v2_2_full_summary.json",
        semantic / "reports/split_leakage_audit.json",
        fine_tuning / "configs/qwen2_5_7b_lora_semantic_causal_v2_2_1_full.json",
        fine_tuning / "configs/qwen2_5_7b_lora_semantic_causal_v2_2_2_recovery_full.json",
        fine_tuning / "scripts/train_qwen_lora_causal_ddp.py",
        fine_tuning / "scripts/evaluate_causal_action_model.py",
        fine_tuning / "scripts/evaluate_free_running_rollout.py",
        fine_tuning / "scripts/build_delay_recovery_actions.py",
        fine_tuning / "scripts/build_recovery_calibration_manifest.py",
        fine_tuning / "scripts/test_latency_metrics.py",
        audit / "scripts/audit_latency.py",
        audit / "AUTONOMOUS_RUN_LEDGER.md",
        audit / "scripts/compare_locked_rollouts.py",
        audit / "scripts/select_validation_rollout_checkpoint.py",
        audit / "scripts/select_latency_quality_model.py",
        audit / "scripts/plot_causal_alignment_results.py",
        audit / "scripts/analyze_policy_imitation_gap.py",
        audit / "reports/validation_v2_2_model_decision.json",
        audit / "reports/locked500_final_model_comparison.json",
        audit / "reports/locked500_predeclared_model_decision.json",
        audit / "reports/FINAL_V2_2_RESEARCH_SUMMARY.json",
        audit / "reports/FINAL_V2_2_RESULTS.png",
        audit / "reports/FINAL_V2_2_RESULTS.pdf",
        audit / "reports/final_v2_2_policy_imitation_gap.json",
        frontend / "scripts/run_speech_to_gloss_e2e.py",
        frontend / "scripts/compare_causal_e2e_versions.py",
        frontend / "scripts/test_e2e_latency_and_actions.py",
        frontend / "reports/task5_pilot_100.causal_e2e_selection.json",
        local_agreement / "README.md",
        local_agreement / "scripts/generate_finetune1_prefix_predictions.py",
        local_agreement / "scripts/evaluate_local_agreement_policy.py",
        local_agreement / "scripts/select_validation_local_agreement_policy.py",
        local_agreement / "scripts/test_local_agreement_policy.py",
        local_agreement / "scripts/run_local_agreement_v2_3_after_v2_2.sh",
        primary / "run_metadata.json",
        primary / "training_summary.json",
        primary / "validation_checkpoint_selection.json",
        primary / "adapter_best/adapter_model.safetensors",
        primary / "adapter_final/adapter_model.safetensors",
        recovery / "run_metadata.json",
        recovery / "training_summary.json",
        recovery / "validation_checkpoint_selection.json",
        recovery / "adapter_best/adapter_model.safetensors",
        recovery / "adapter_final/adapter_model.safetensors",
    ]
    files: dict[str, Any] = {}
    for path in paths:
        relative = str(path.relative_to(args.project_root))
        files[relative] = (
            {"exists": True, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            if path.is_file()
            else {"exists": False}
        )

    torch_version = package_version("torch")
    cuda_version = None
    gpu_names: list[str] = []
    if torch_version is not None:
        import torch

        cuda_version = torch.version.cuda
        if torch.cuda.is_available():
            gpu_names = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]

    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "v2_2_reproducibility_manifest",
        "runtime": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch_version,
            "transformers": package_version("transformers"),
            "peft": package_version("peft"),
            "cuda_runtime": cuda_version,
            "gpu_names": gpu_names,
        },
        "file_count": len(files),
        "files": files,
        "note": (
            "Large action manifests are represented by hashes in run_metadata.json; "
            "this manifest hashes the core code, configs, adapters and final decisions."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "file_count": len(files)}, indent=2))


if __name__ == "__main__":
    main()
