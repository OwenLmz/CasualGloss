#!/usr/bin/env python3
"""Verify that the full Seed 2 replication changes only seed and run identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as handle:
        return sum(1 for line in handle if line.strip())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed1-config", type=Path, required=True)
    parser.add_argument("--seed2-config", type=Path, required=True)
    parser.add_argument("--causal-root", type=Path, required=True)
    parser.add_argument("--seed1-root", type=Path, required=True)
    parser.add_argument("--source-adapter", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    seed1 = read_json(args.seed1_config)
    seed2 = read_json(args.seed2_config)
    all_keys = sorted(set(seed1) | set(seed2))
    differences = {
        key: {"seed1": seed1.get(key), "seed2": seed2.get(key)}
        for key in all_keys
        if seed1.get(key) != seed2.get(key)
    }
    allowed_differences = {"experiment_name", "seed"}

    validation = read_json(args.causal_root / "reports" / "validation_report.json")
    manifests = {
        "train_actions": args.causal_root / "manifests" / "train_actions.jsonl",
        "validation_actions": args.causal_root / "manifests" / "validation_actions.jsonl",
        "test_actions": args.causal_root / "manifests" / "test_actions.jsonl",
        "parent_splits": args.causal_root / "manifests" / "parent_splits_full.jsonl",
        "causal_pairs": args.causal_root / "outputs" / "causal_aligned_pairs_full.jsonl",
    }
    counts = {name: count_jsonl(path) for name, path in manifests.items()}
    seed1_run = (
        args.seed1_root
        / "runs"
        / "qwen2_5_7b_lora_causal_action_full_corrected_v1"
    )
    checks = {
        "configs_have_identical_keys": set(seed1) == set(seed2),
        "only_seed_and_experiment_name_differ": set(differences) == allowed_differences,
        "seed1_is_expected": seed1.get("seed") == 20260720,
        "seed2_is_expected": seed2.get("seed") == 20260721,
        "same_model_revision": seed1.get("model_revision") == seed2.get("model_revision"),
        "same_dataset_version": seed1.get("dataset_version") == seed2.get("dataset_version"),
        "causal_dataset_validation_ok": bool(validation.get("ok")),
        "action_counts_exact": counts["train_actions"] == 397700
        and counts["validation_actions"] == 9209
        and counts["test_actions"] == 8365,
        "parent_count_exact": counts["parent_splits"] == 19947,
        "causal_pair_count_exact": counts["causal_pairs"] == 19947,
        "source_adapter_exists": (
            args.source_adapter / "adapter_model.safetensors"
        ).is_file(),
        "seed1_action_metrics_exist": (
            seed1_run / "evaluation_full" / "metrics_summary.json"
        ).is_file(),
        "seed1_rollouts_exist": (
            seed1_run / "rollout_full" / "sentence_rollouts.jsonl"
        ).is_file(),
    }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "full_seed_replication_preflight_pseudo_labels_not_gold_asl",
        "seed_pair": [seed1.get("seed"), seed2.get("seed")],
        "config_differences": differences,
        "manifest_counts": counts,
        "locked_file_sha256": {
            name: sha256_file(path) for name, path in manifests.items()
        },
        "checks": checks,
        "ok": all(checks.values()),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
