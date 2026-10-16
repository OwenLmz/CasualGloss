# Fine-tune 2: Full-data Seed 2 Replication

This directory contains the second random-seed replication of the full-data
causal-action Fine-tune 2 experiment.

## Controlled design

- Model: `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`
- Initialization: the same frozen Fine-tune 1 best LoRA adapter used by Seed 1
- Data: `causal_alignment_v1_corrected_full_v1`
- Training: one epoch, 397,700 pseudo-action examples, effective batch size 32
- Seed 1: `20260720`
- Seed 2: `20260721`
- Locked evaluation: the same 500 parent sentences and 8,365 action states

Only `experiment_name` and `seed` may differ from the Seed 1 full config.

## Comparisons

1. Seed 2 full versus Seed 2 5K tests whether the data-scale effect replicates.
2. Seed 2 full versus Seed 1 full measures random-seed sensitivity.

All gloss targets and emission times are pseudo labels, not gold ASL
annotations. The experiment therefore evaluates reproducibility of the learned
pseudo causal policy, not definitive ASL linguistic correctness.

## Run

```bash
bash scripts/run_finetune2_full_seed2.sh
```

The runner performs two preflights, a GPU smoke test, four-GPU training,
gold-state action evaluation, free-running sentence rollout, paired bootstrap
comparisons, structural validation, and SHA-256 hashing.
