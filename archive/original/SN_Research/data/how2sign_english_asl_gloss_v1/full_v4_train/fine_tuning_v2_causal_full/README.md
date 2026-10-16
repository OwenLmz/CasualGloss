# Full-scale Fine-tune 2 causal-action experiment

This experiment tests whether scaling causal supervision from 71,882 to
397,700 training actions improves autonomous incremental pseudo-gloss output.

## Controlled comparison

- Both the 5K and full models start from the same Fine-tune 1 LoRA adapter.
- Both use the same model revision, seed, learning rate, effective batch size,
  one-epoch budget, system prompt, and corrected timeline policy.
- Both are evaluated on the exact same locked 500-parent test set.
- Free-running rollout never inserts gold committed gloss into model history.
- Paired bootstrap confidence intervals are computed over the 500 test parents.

## Scientific boundary

The experiment measures learning of a pseudo causal translation policy. The
gloss targets are LLM-generated and timing is lexical/SimAlign-derived; neither
constitutes gold ASL supervision.

## Server entry point

```bash
bash data/how2sign_english_asl_gloss_v1/full_v4_train/fine_tuning_v2_causal_full/scripts/run_finetune2_full.sh
```
