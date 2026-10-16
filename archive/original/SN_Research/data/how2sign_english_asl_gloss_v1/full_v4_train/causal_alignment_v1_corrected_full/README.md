# Full corrected causal-alignment dataset

This directory builds the full English-text to pseudo-ASL-gloss causal-action
dataset after the controlled 5K scale experiment passed all predeclared gates.

## Locked evaluation protocol

- Candidate source: `fine_tuning_v1/manifests/train.jsonl`.
- The 500 validation and 500 test parents from
  `causal_alignment_v1_corrected_5k` are preserved exactly.
- Any additional parent sharing a How2Sign video ID with the locked validation
  or test sets is quarantined and cannot enter training.
- Structural filters and the corrected timeline policy are unchanged from the
  validated 5K experiment.
- Targets remain LLM-generated pseudo glosses and timing remains heuristic
  lexical/SimAlign pseudo alignment. Neither is gold ASL annotation.

## Main outputs

- `outputs/causal_aligned_pairs_full.jsonl`: selected train/validation/test parents.
- `outputs/prefix_action_examples_corrected_full.jsonl`: all selected actions.
- `manifests/{train,validation,test}_{pairs,actions}.jsonl`: split manifests.
- `manifests/parent_splits_full.jsonl`: compact parent split manifest.
- `outputs/heldout_video_quarantine_pairs.jsonl`: non-locked rows excluded to
  protect held-out video groups.
- `reports/finalization_report.json`: counts, quality metrics, and leakage audit.
- `reports/validation_report.json`: end-to-end structural validation.
- `reports/core_file_sha256.json`: reproducibility hashes.

## Run on the research server

```bash
bash data/how2sign_english_asl_gloss_v1/full_v4_train/causal_alignment_v1_corrected_full/scripts/run_corrected_causal_alignment_full.sh
```

This stage constructs and validates data only. Full-scale model fine-tuning is
intentionally a separate experiment.
