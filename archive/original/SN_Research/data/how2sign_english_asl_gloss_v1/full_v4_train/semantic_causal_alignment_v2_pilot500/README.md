# Semantic Causal Alignment v2 - 500-parent pilot

This experiment replaces purely lexical source-to-gloss timing with a model-derived
semantic sufficiency signal while preserving high-confidence lexical anchors.

## Frozen inputs

- Parent data: the training split of `causal_alignment_v1_corrected_full`.
- Target: Qwen-generated weak pseudo ASL gloss, explicitly non-gold.
- Scorer: the frozen Fine-tune 1 `Qwen2.5-7B-Instruct` LoRA checkpoint.
- Pilot: 500 proportionally stratified training parents selected with seed `20260803`.

## Method

For each English source prefix, the scorer teacher-forces the full target pseudo gloss.
Because the model is autoregressive, the probability of gloss token `j` is conditioned
only on the observed English prefix and committed gloss tokens before `j`; it cannot see
future gloss tokens. A prefix becomes eligible when its token probability is sufficiently
close to the full-source probability and remains stable for a fixed number of later
prefixes. Monotonic dynamic programming combines this semantic evidence with reliable v1
lexical anchors. A scheduling pass preserves the existing one-action-per-source-step
training format.

Three policies are frozen before execution: `low_latency`, `balanced`, and
`high_confidence`. The primary policy is `balanced`.

## Scientific scope

This is a pseudo-alignment experiment, not gold ASL timing. The internal probability
diagnostic is produced by Fine-tune 1 and therefore is not an independent linguistic
evaluation. Qualified ASL review remains necessary.

## Core outputs

- `outputs/semantic_causal_aligned_pairs_v2_pilot500.jsonl`: parent-level v2 alignments.
- `manifests/semantic_actions_v2_pilot500.jsonl`: expanded Fine-tune 2 actions.
- `reports/semantic_alignment_v2_pilot500_summary.json`: v1/v2 and policy comparison.
- `previews/v1_vs_v2_alignment_preview_50.tsv`: readable high-priority examples.
- `reviews/semantic_alignment_manual_review_50.tsv`: pending review worksheet.
- `reports/validation_report.json`: structural validation result.
- `RESULTS.md`: concise scientific interpretation.

## Run

```bash
cd /123090274/SN_Research
bash data/how2sign_english_asl_gloss_v1/full_v4_train/semantic_causal_alignment_v2_pilot500/scripts/run_semantic_alignment_v2_pilot500.sh
```
