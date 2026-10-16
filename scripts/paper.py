#!/usr/bin/env python3
"""Navigate and dispatch to the byte-identical paper implementations.

This wrapper supplies import paths, not a replacement algorithm. No config,
checkpoint selection, model prompt, decoding rule or data content is rewritten.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "archive/original/SN_Research"
FULL = ORIGINAL / "data/how2sign_english_asl_gloss_v1/full_v4_train"
FRONT = ORIGINAL / "data/how2sign_english_asl_gloss_v1/chinese_speech_frontend_v0"
ENTRIES = {
    "rollout": FULL / "fine_tuning_v2_semantic_v2_2_full/scripts/evaluate_free_running_rollout.py",
    "train-policy": FULL / "fine_tuning_v2_semantic_v2_2_full/scripts/train_qwen_lora_causal_ddp.py",
    "teacher-eval": FULL / "fine_tuning_v2_semantic_v2_2_full/scripts/evaluate_causal_action_model.py",
    "build-burst": FULL / "semantic_causal_alignment_v2_2_full/scripts/build_confident_burst_alignment_v2_2.py",
    "build-recovery": FULL / "state_recovery_v2_3_full/scripts/prepare_v2_3_manifests.py",
    "train-learned": FULL / "transllama_learned_adaptive_20260918/scripts/train_transllama.py",
    "eval-learned": FULL / "transllama_learned_adaptive_20260918/scripts/evaluate_transllama.py",
    "eval-epee-action": FULL / "clerc_epee_adaptive_comparison_20260920/scripts/evaluate_action_epee_fp16.py",
    "eval-epee-learned": FULL / "clerc_epee_adaptive_comparison_20260920/scripts/evaluate_transllama_epee_fp16.py",
    "speech-cascade": FRONT / "scripts/run_speech_to_gloss_e2e.py",
}


def command(entry: str, forwarded: list[str], nproc: int = 1) -> list[str]:
    path = ENTRIES[entry]
    if not path.is_file():
        raise FileNotFoundError(f"Missing original entry: {path}")
    if nproc < 1:
        raise ValueError("nproc must be positive")
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if nproc != 1 and entry != "train-policy":
        raise ValueError("Only the original DDP train-policy entry supports multi-process dispatch")
    prefix = [sys.executable]
    if nproc > 1:
        prefix += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}"]
    return prefix + [str(path)] + forwarded


def environment() -> dict[str, str]:
    env = os.environ.copy()
    # Preserve hierarchy-dependent first-party imports without modifying originals.
    dependencies = [
        FULL / "fine_tuning_v2_semantic_v2_2_full/scripts",
        FULL / "transllama_learned_adaptive_20260918/scripts",
        FULL / "clerc_epee_adaptive_comparison_20260920/scripts",
        FULL / "clerc_epee_external_eval_v1/scripts",
        FULL / "fine_tuning_v1/scripts",
        FRONT / "scripts",
    ]
    old = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join([*(str(p) for p in dependencies), *([old] if old else [])])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("list", help="List exact archived entry paths; no ML dependencies needed")
    run = sub.add_parser("run", help="Forward CLI arguments unchanged to an original script")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--nproc", type=int, default=1)
    run.add_argument("entry", choices=sorted(ENTRIES))
    run.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.mode == "list":
        for key, path in ENTRIES.items():
            print(f"{key:20} {path.relative_to(ROOT).as_posix()}" + (" [MISSING]" if not path.is_file() else ""))
        return int(any(not p.is_file() for p in ENTRIES.values()))
    cmd = command(args.entry, args.args, args.nproc)
    if args.dry_run:
        import json
        print(json.dumps(cmd, indent=2, ensure_ascii=False))
        return 0
    return subprocess.run(cmd, env=environment(), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
