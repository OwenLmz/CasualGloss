# Epee pseudo-gloss generator validation

This launcher evaluates the frozen Qwen2.5 v4 pseudo-gloss generator directly
against the multi-reference human gloss annotations in CLERC Epee v0.3.

The experiment is independent of the existing CausalGloss external-transfer
evaluation. It does not train or tune a model on Epee.

Frozen inputs:

- Epee commit: `93bc8aa0a6526e86af80db5f59af8d133292c9b5`
- Generator: `Qwen/Qwen2.5-7B-Instruct`
- Model revision: `a09a35458c702b33eeacc393d103063234e8bc28`
- Prompt: `asl_gloss_prompt_v4_deterministic_text_only.txt`
- Greedy decoding, seed `20260720`, batch size `4`

The V100 replication uses FP16 because V100 does not provide native BF16
execution. A frozen How2Sign smoke set is compared with the original BF16
outputs before the full Epee evaluation is accepted.
