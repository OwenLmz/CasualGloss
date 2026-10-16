#!/usr/bin/env python3
"""Run a small saved-checkpoint check using the archived free-running evaluator.

Inputs/outputs contain research text and belong outside the Git release. This is
not a full-benchmark replication or a from-scratch training reproduction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from paper import ROOT, command, environment


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-rollouts", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 1 <= args.count <= 32:
        parser.error("count must be between 1 and 32; this is a small smoke test")
    destination = args.output_dir.resolve()
    if destination == ROOT or ROOT in destination.parents:
        parser.error("Use a PRIVATE output directory outside the source repository")
    # Never overwrite historical outputs or an earlier check.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    with args.reference_rollouts.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) == args.count:
                break
    if len(rows) != args.count or len({r["id"] for r in rows}) != args.count:
        raise ValueError("Reference must provide the requested number of unique parents")
    parents = args.output_dir / "parents.jsonl"
    pairs = args.output_dir / "pairs.jsonl"
    parents.write_text("".join(json.dumps({"id": r["id"], "split": "test"}) + "\n" for r in rows), encoding="utf-8")
    pairs.write_text("".join(json.dumps({k: r[k] for k in ("id", "source_text_en", "source_tokens", "target_pseudo_asl_gloss_tokens")}, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    original = ROOT / "archive/original/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train/fine_tuning_v2_semantic_v2_2_full/scripts/evaluate_free_running_rollout.py"
    cmd = command("rollout", ["--config", str(args.config), "--parent-splits", str(parents),
                             "--causal-pairs", str(pairs), "--adapter-dir", str(args.adapter_dir),
                             "--output-dir", str(args.output_dir / "rollout"), "--stage", "release_smoke_not_full_benchmark",
                             "--split", "test", "--device", args.device, "--generation-batch-size", str(args.count)])
    subprocess.run(cmd, env=environment(), check=True)
    new = [json.loads(line) for line in (args.output_dir / "rollout/sentence_rollouts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {row["id"]: row for row in new}
    comparisons = []
    for old in rows:
        fresh = by_id[old["id"]]
        comparisons.append({
            "id_sha256": hashlib.sha256(old["id"].encode()).hexdigest(),
            "gloss_exact": fresh["predicted_pseudo_asl_gloss_tokens"] == old["predicted_pseudo_asl_gloss_tokens"],
            "write_positions_exact": fresh["write_source_positions"] == old["write_source_positions"],
            "termination_exact": fresh["termination_reason"] == old["termination_reason"],
        })
    import torch
    import transformers
    import peft
    report = {
        "level": "small checkpoint inference smoke; not full benchmark or retraining",
        "count": args.count, "reference_sha256": sha(args.reference_rollouts),
        "config_sha256": sha(args.config), "evaluator_sha256": sha(original),
        "adapter_sha256": sha(args.adapter_dir / "adapter_model.safetensors"),
        "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
        "device": torch.cuda.get_device_name(torch.device(args.device)),
        "batch_size": args.count,
        "caveat": "Changed hardware or batch grouping may change greedy floating-point outcomes. No seed was selected using this check.",
        "comparisons": comparisons,
        "all_exact": all(all(row[k] for k in ("gloss_exact", "write_positions_exact", "termination_exact")) for row in comparisons),
    }
    (args.output_dir / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["all_exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
