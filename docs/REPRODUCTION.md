# Reproduction: increasing levels of evidence

Use a fresh output directory at every step. Never point new training/evaluation output at a selected historical run. Historical originals remain unchanged in `archive/original`; `scripts/paper.py` only dispatches to those implementations with explicit arguments and required first-party import paths.

## 1. Source / numerical checks first

```bash
python scripts/check_release.py
python -m unittest discover -s tests -p "test_*.py"
python -m unittest discover -s tests/verification -v
```

Follow [VERIFICATION.md](VERIFICATION.md) to obtain the 91 explicitly registered verification inputs privately and independently rescore saved outputs. CPU + NumPy suffices. This reconstructs paper numbers without retraining, but is not proof that a new model run matches.

## 2. Neural environment

Historical selected text-policy metadata records torch `2.11.0+cu128`, transformers `4.57.6`, peft `0.18.0`, NLTK tokenization and SDPA attention. See `requirements/`. The original experiments used different GPUs; the audit server currently has four 24-GB RTX 4090 D devices. Do not label audit hardware as historical training hardware. CLERC shared comparison was run in FP16 on V100S; the generic text rollout entry uses BF16.

Use the pinned backbone revision and the correct selected adapter/config. Check device capability, driver and memory before installation. The speech frontend, Qwen3-TTS and SeamlessStreaming use separate environments and external code. We have not certified a clean installation of every speech dependency.

Inspect available commands without running an experiment:

```bash
python scripts/paper.py list
python scripts/paper.py run --dry-run rollout -- --help
```

With core dependencies installed, use `run rollout -- --help` for the archived evaluator's actual options. All arguments after `--` are passed unchanged. This dispatch does **not** remove hardcoded paths from arbitrary shell scripts or turn historical configs into portable universal defaults.

## 3. Small checkpoint check

```bash
python scripts/checkpoint_smoke.py \
  --reference-rollouts /private/path/to/cg_s1.jsonl \
  --config /path/to/original/seed20260829_config.json \
  --adapter-dir /private/path/to/selected/adapter_final \
  --output-dir /private/path/to/a_new_smoke_check \
  --count 8 --device cuda:0
```

This uses the first eight saved parents in their original order, builds source/reference fixtures privately, calls the unmodified free-running evaluator, and compares generated tokens, WRITE positions and termination. Reference glosses are used by the original evaluation budget/metrics, **not** supplied as the model's committed history. Report hardware/batch differences even if all samples match. The small sample is fixed before evaluation and is not selected for favorable outputs.

## 4. Full locked500 inference (explicit paths)

Below `F` is your private materialized full_v4_train artifact tree; `CODE` is the archived source tree. This is a Bash example, not a command to paste with unresolved paths.

```bash
F=/absolute/path/to/private_assets/data/how2sign_english_asl_gloss_v1/full_v4_train
CODE=archive/original/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train
python scripts/paper.py run rollout -- \
  --config "$CODE/state_recovery_v2_3_full/configs/qwen2_5_7b_lora_state_recovery_v2_3_full.json" \
  --parent-splits "$F/causal_alignment_v1_corrected_full/manifests/parent_splits_full.jsonl" \
  --causal-pairs "$F/causal_alignment_v1_corrected_full/outputs/causal_aligned_pairs_full.jsonl" \
  --adapter-dir "$F/state_recovery_v2_3_full/runs/qwen2_5_7b_lora_state_recovery_v2_3_full/adapter_final" \
  --output-dir /private/path/to/fresh_locked500_s1 \
  --stage release_replication_s1 --split test --device cuda:0 \
  --generation-batch-size 32
```

Use the other seeds' own configs/adapters. Keep the 500 test IDs and input hashes fixed, record library versions and compare complete trajectories, not merely rounded F1. A full text benchmark is not a speech-cascade replication.

## 5. Training continuation and from-raw reconstruction

Training chain: weak-target generation → Fine-tune-1 → lexical/prefix-semantic alignment → confident/burst Treebank actions → stage-A policy (534,877 actions) → WAIT-balanced + recovery stage B (439,103 actions). The last stage starts from the selected stage-A adapter, **not** directly from the base Qwen checkpoint. WAIT-balanced supervision belongs to the recovery-stage construction, not the selected stage-A training input.

```bash
python scripts/paper.py run --dry-run --nproc 2 train-policy -- \
  --config /path/to/seed_config.json \
  --train-manifest /private/train_actions_full.jsonl \
  --validation-manifest /private/development_actions.jsonl \
  --test-manifest /private/confirmation_actions.jsonl \
  --source-adapter-dir /private/selected_stage_a/adapter_best \
  --output-dir /private/a_new_training_run
```

The `--test-manifest` name in the historical trainer here refers to reserved confirmation actions, not locked500 model selection. Preserve the original development-only selection and evaluate locked500 only after that selection. Before removing `--dry-run`, inspect the original full/screen gate scripts, training budget, GPU count and effective batch. Do not claim an arbitrary short smoke training reproduces the paper.

Fixed baselines require schedule-specific training manifests; learned baseline requires independent SimAlign alignments/LR screening and three final seeds. CLERC normalization, best-reference scoring and FP16 shared comparison must remain frozen. Speech replication needs translated/TTS manifests, correct reused waveforms, exact frontend commits, and serial computation timing. The artifact registry and [paper map](PAPER_TO_CODE.md) identify each branch.

## Known portability boundaries

- Original Bash launchers have absolute server paths/GPU selections. Kept for evidence, not advertised as one-click portable commands.
- Original figure scripts may import an external Nature visual-QA helper. Numerical verification is independent of it; the helper is not vendored.
- Historical SimulStreaming wrappers reference upstream third-party code. They are not the paper's SeamlessStreaming frontend by substitution.
- Stochastic data generation or changed hardware may prevent byte-identical regeneration even with the same nominal seed. Frozen artifacts and honest comparison are required.
