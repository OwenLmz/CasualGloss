# Full Fine-tune 2 on Semantic Causal Alignment v2.1

The experiment continues the fixed Fine-tune 1 LoRA adapter on one-action-at-a-time causal
supervision. Each state contains the observed English prefix, already committed pseudo gloss,
and source status. The target is exactly one gloss token, `<WAIT>`, or `<END>`.

## Controlled setup

- Base model: Qwen2.5-7B-Instruct at the fixed project revision
- Initialization: the same Fine-tune 1 adapter used in prior scale experiments
- Training: one predeclared epoch, LoRA, BF16, learning rate `3e-5`
- Hardware: two GPUs, micro-batch 4 per GPU, gradient accumulation 4
- Effective batch: 32
- Evaluation: locked How2Sign video-group validation and test splits

Training and test metrics measure agreement with pseudo causal actions. They are not verified
ASL translation accuracy.
