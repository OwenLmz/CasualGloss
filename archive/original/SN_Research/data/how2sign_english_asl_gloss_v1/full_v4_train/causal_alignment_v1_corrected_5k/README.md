# Corrected causal alignment, 5,000 parents

This directory scales the corrected 500-parent causal-action experiment to a
controlled 5,000-parent dataset. It is a data-construction stage only; it does
not train a new model.

## Scientific status

- Source: official How2Sign manually realigned English transcription from the
  Fine-tune 1 train split.
- Target: fixed-checkpoint Qwen2.5-7B-Instruct v4 pseudo ASL gloss.
- Timing: lexical anchors plus SimAlign IterMax pseudo alignment.
- Gold status: neither the gloss nor token timing is gold ASL annotation.
- Human status: the 50-row review sheet still requires a qualified ASL reviewer.

## What changed from the original 500-parent construction

1. Rule-based fallback rows and gloss tokens with more than two hyphen
   components are removed before GPU alignment.
2. A deterministic 6,200-parent pool is aligned in four GPU shards, leaving a
   margin for post-alignment quality filtering.
3. The timeline directly emits `<END>` when source and target are complete.
   It never gives `<WAIT>` and `<END>` to an identical complete state.
4. Final train/validation/test splits contain exactly 4,000/500/500 parents.
5. How2Sign video IDs are treated as connected groups, so the same source
   video cannot appear in more than one split.

## Reproduce on the research server

From `/123090274/SN_Research`:

```bash
source /123090274/SN_Research/venvs/wlasl-cu128-env/bin/activate
nohup bash data/how2sign_english_asl_gloss_v1/full_v4_train/causal_alignment_v1_corrected_5k/scripts/run_corrected_causal_alignment_5k.sh \
  > data/how2sign_english_asl_gloss_v1/full_v4_train/causal_alignment_v1_corrected_5k/reports/run_5k.log 2>&1 &
```

Monitor without printing the full log:

```bash
tail -n 30 data/how2sign_english_asl_gloss_v1/full_v4_train/causal_alignment_v1_corrected_5k/reports/run_5k.log
```

## Core outputs

- `outputs/causal_aligned_pairs_5000.jsonl`: final parent-level dataset.
- `outputs/prefix_action_examples_corrected_5000.jsonl`: all corrected actions.
- `manifests/train_actions.jsonl`: Fine-tune 2 training actions.
- `manifests/validation_actions.jsonl`: model-selection actions.
- `manifests/test_actions.jsonl`: locked held-out actions.
- `reviews/causal_alignment_review_50.tsv`: risk-oriented review sample.
- `reports/finalization_report.json`: selection, filtering, and split statistics.
- `reports/validation_report.json`: end-to-end invariant checks.
- `reports/core_file_sha256.json`: reproducibility hashes.

Do not describe this dataset as gold ASL gloss or gold sign timing in a paper.
