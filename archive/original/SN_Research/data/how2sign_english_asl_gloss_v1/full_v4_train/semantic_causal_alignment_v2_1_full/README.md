# Semantic Causal Alignment v2.1 Full

This directory scales the pilot-gated v2.1 alignment to the locked How2Sign splits:

- 18,947 train parents
- 500 validation parents
- 500 test parents

## Pipeline

1. `prepare_full_inputs.py` checks source integrity and streaming-causal compatibility.
2. Two balanced A100 shards score every pseudo-gloss token under every English prefix.
3. `reconcile_prefilter_after_rule_fix.py` records the contraction-rule delta without hiding it.
4. `finalize_semantic_alignment_v2_1.py` fuses frozen semantic scores and lexical constraints.
5. `quality_gate_v2_1.py` applies the same thresholds declared before scale-up.
6. `split_full_outputs.py` exports locked train, validation, and test action manifests.
7. `run_after_scoring.sh` starts Full Fine-tune 2 only after every preceding gate passes.

## Scientific status

The English transcriptions are official manually realigned How2Sign text. The target glosses
are Qwen-generated weak pseudo ASL glosses, and the emission positions are model-derived
pseudo timings. Neither is a gold ASL annotation. Qualified ASL evaluation remains required.
