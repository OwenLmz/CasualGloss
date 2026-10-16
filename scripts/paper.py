#!/usr/bin/env python3
"""Dispatch paper-only workflows with isolated first-party import paths.

Arguments after the entry name are forwarded unchanged. This dispatcher does
not select data/checkpoints, rewrite configs, or change a model algorithm.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Entry:
    path: str
    dependencies: tuple[str, ...] = ()
    distributed: bool = False


ENTRIES = {
    "train-policy": Entry("method/causalgloss/train_qwen_lora_causal_ddp.py", distributed=True),
    "rollout": Entry("method/causalgloss/evaluate_free_running_rollout.py"),
    "train-upstream": Entry("training/upstream_gloss/train_qwen_lora_full_ddp.py", distributed=True),
    "prepare-upstream": Entry("training/upstream_gloss/prepare_full_finetuning_splits.py"),
    "score-prefix": Entry("data_preparation/semantic_alignment/score_prefix_semantics.py"),
    "prepare-semantic": Entry("data_preparation/semantic_alignment/prepare_full_inputs.py"),
    "finalize-semantic": Entry("data_preparation/semantic_alignment/finalize_semantic_alignment_v2_1.py"),
    "build-burst": Entry("data_preparation/semantic_alignment/build_confident_burst_alignment_v2_2.py"),
    "split-burst": Entry("data_preparation/semantic_alignment/split_and_validate_v2_2.py"),
    "repair-prompts": Entry("data_preparation/semantic_alignment/repair_action_prompt_detokenization.py"),
    "build-recovery": Entry("data_preparation/recovery/prepare_v2_3_manifests.py"),
    "build-waitk": Entry("baselines/fixed_waitk/build_wait_k_actions.py"),
    "validate-waitk": Entry("baselines/fixed_waitk/validate_wait_k_actions.py"),
    "build-learned-alignments": Entry("baselines/transllama/build_alignments.py"),
    "train-learned": Entry("baselines/transllama/train_transllama.py"),
    "select-learned-lr": Entry("baselines/transllama/select_learning_rate.py"),
    "eval-learned": Entry("baselines/transllama/evaluate_transllama.py"),
    "train-ablation": Entry("ablations/train_qwen_lora_causal_ddp.py", distributed=True),
    "eval-ablation": Entry("ablations/evaluate_free_running_rollout.py"),
    "prepare-ablations": Entry("ablations/prepare_experiment_manifests.py"),
    "prepare-epee": Entry("evaluation/clerc/prepare_epee_external_eval.py"),
    "eval-epee-action": Entry("evaluation/clerc/evaluate_action_epee_fp16.py", ("method/causalgloss",)),
    "eval-epee-fixed": Entry("evaluation/clerc/evaluate_fixed_waitk_epee_fp16.py", ("method/causalgloss",)),
    "eval-epee-learned": Entry("evaluation/clerc/evaluate_transllama_epee_fp16.py", ("baselines/transllama",)),
    "compare-epee": Entry("evaluation/clerc/compare_epee_adaptive.py"),
    "compare-epee-fixed": Entry("evaluation/clerc/compare_fixed_waitk_epee.py"),
    "speech-cascade": Entry("speech/frontend/run_speech_to_gloss_e2e.py"),
    "speech-compare": Entry("speech/frontend/compare_causal_e2e_versions.py"),
    "corpus-prepare": Entry("speech/corpus/prepare_full_corpus_manifest.py"),
    "corpus-translate": Entry("speech/corpus/translate_full_corpus_en_zh.py"),
    "corpus-tts": Entry("speech/corpus/generate_full_corpus_tts.py"),
    "corpus-finalize": Entry("speech/corpus/finalize_cleaned_corpus.py"),
}


def command(entry: str, forwarded: list[str], nproc: int = 1) -> list[str]:
    spec = ENTRIES[entry]
    path = ROOT / spec.path
    if not path.is_file():
        raise FileNotFoundError(f"Missing paper entry: {spec.path}")
    if nproc < 1:
        raise ValueError("nproc must be positive")
    if nproc != 1 and not spec.distributed:
        raise ValueError(f"{entry} does not implement distributed training; use --nproc 1")
    forwarded = list(forwarded)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    prefix = [sys.executable]
    # These historical trainers initialize a process group unconditionally,
    # including one-GPU runs. torchrun supplies their rank environment.
    if spec.distributed:
        prefix += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}"]
    return prefix + [str(path)] + forwarded


def environment(entry: str) -> dict[str, str]:
    spec = ENTRIES[entry]
    env = os.environ.copy()
    # Never put every training implementation on a shared PYTHONPATH: the
    # ablation trainer has its own same-named causal_action_common module.
    locations = [(ROOT / spec.path).parent, *(ROOT / item for item in spec.dependencies)]
    unique = list(dict.fromkeys(str(path) for path in locations))
    # Intentionally do not inherit PYTHONPATH. Site-packages remains available;
    # unmanaged first-party modules must not silently override paper modules.
    env["PYTHONPATH"] = os.pathsep.join(unique)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("list", help="List retained entrypoints without importing ML packages")
    run = sub.add_parser("run", help="Forward arguments to one retained implementation")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--nproc", type=int, default=1)
    run.add_argument("entry", choices=sorted(ENTRIES))
    run.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.mode == "list":
        missing = []
        for name, spec in ENTRIES.items():
            exists = (ROOT / spec.path).is_file()
            print(f"{name:26} {spec.path}" + (" [MISSING]" if not exists else ""))
            if not exists:
                missing.append(name)
        return int(bool(missing))
    cmd = command(args.entry, args.args, args.nproc)
    env = environment(args.entry)
    if args.dry_run:
        print(json.dumps({"command": cmd, "PYTHONPATH": env["PYTHONPATH"]}, indent=2))
        return 0
    # Preserve the caller's working directory: supplied relative data paths
    # must have the same meaning as when running the underlying script.
    return subprocess.run(cmd, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
