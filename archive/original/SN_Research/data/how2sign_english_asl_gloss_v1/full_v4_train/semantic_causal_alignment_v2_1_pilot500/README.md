# Semantic Causal Alignment v2.1 Pilot 500

This directory is the fixed 500-parent gate for Semantic Causal Alignment v2.1.

## What changed from v2

1. Occurrence-aware monotonic lexical anchors distinguish repeated concepts.
2. Unmatched tokens receive a transparent monotonic-neighbor interpolation anchor.
3. Bounded-delay rescue candidates prevent one weak token from moving every later gloss to sentence end.
4. Negation is constrained to a short delay after its source evidence; irreconcilable order is excluded.
5. Corrupt English and strongly noncausal pseudo-gloss order are explicitly excluded from Fine-tune 2.

## Frozen semantic scorer

- Base: `Qwen/Qwen2.5-7B-Instruct`
- Revision: `a09a35458c702b33eeacc393d103063234e8bc28`
- Initialization: the frozen Fine-tune 1 LoRA adapter
- Signal: teacher-forced pseudo-gloss token log probability at every English prefix

No model parameters are updated while building the alignment.

## Gate

The predeclared gate combines all 500 parents, the fixed 50-row technical stress set,
and a deterministic random 50-row check. See `reports/quality_gate_v2_1.json`.

These checks validate pseudo-label structure, timing consistency, and reproducibility.
They do not validate ASL linguistic correctness. The target gloss and timing remain non-gold.
