# Full corrected causal-alignment results

## Status

- Engineering construction status: `passed`
- Scientific status: `llm_pseudo_gloss_and_heuristic_pseudo_timing_not_gold_asl`
- Timeline policy: `corrected_direct_end_at_unchanged_complete_state_v1`
- Qualified ASL linguistic review: pending

## Scale

- Fine-tune 1 source parents: 27079
- Candidates after fixed prefilter: 22859
- Structurally eligible after alignment: 22569
- Final selected parents: 19947
- Train/validation/test parents: 18947/500/500
- Train/validation/test actions: 397700/9209/8365
- Held-out-video quarantine parents: 2622
- Structurally excluded parents: 290

## Locked evaluation integrity

- Locked validation parent set exact match: True
- Locked test parent set exact match: True
- Train-validation source-video overlap: 0
- Train-test source-video overlap: 0
- Validation-test source-video overlap: 0

## Alignment profile

- Mean lexical alignment ratio: 0.965035
- Mean SimAlign fallback ratio: 0.024819
- Mean unaligned gloss-token ratio: 0.010145
- Mean WAIT-action ratio: 0.52924

## Interpretation boundary

This dataset is suitable for the next controlled full-scale causal-policy
fine-tuning experiment. It does not provide gold ASL gloss or gold sign timing.
The fixed validation and test parents must remain evaluation-only.
