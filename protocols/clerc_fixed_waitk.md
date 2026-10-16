# Protocol Amendment: CLERC Epee Fixed Wait-k Transfer

Status: frozen before any Fixed Wait-k CLERC Epee rollout is generated.

Date frozen: 2026-09-20 (Asia/Shanghai)

This amendment extends the already frozen CLERC Epee adaptive-policy protocol.
The CausalGloss, TransLLaMA-style, data, inference, scoring,
and reporting settings in the parent protocol are unchanged. Existing CLERC
results were known before this amendment, but no Fixed Wait-k CLERC output was
generated or inspected before the choices below were fixed.

## Scientific question

Does the quality-latency behavior of the submitted CausalGloss system relative
to schedule-specific Fixed Wait-k models transfer zero-shot from the locked
How2Sign-derived benchmark to external human-annotated CLERC Epee references?

## Frozen Fixed Wait-k systems

Evaluate the complete submitted-paper fixed frontier, k in {1, 3, 5, 7}.
Every model is a schedule-specific trained policy with the same frozen
Qwen2.5-7B-Instruct backbone revision, Fine-tune-1 initialization, training
data, one-epoch budget, and effective batch size used in the paper.

- k=1: original seed 20260821, selected adapter_best checkpoint.
- k=3: recovered original seed 20260821, selected adapter_best checkpoint.
- k=5: original seed 20260821 plus seeds 20260904 and 20260905, each using
  its independently selected adapter_best checkpoint.
- k=7: original seed 20260821 plus seeds 20260904 and 20260905, each using
  its independently selected adapter_best checkpoint.

No checkpoint is retrained, reselected, or tuned on CLERC Epee. The k=3
recovery checkpoint is the same finite, completed checkpoint used in the
submitted-paper locked-500 fixed frontier. The failed non-finite k=3 run is not
used.

## Shared evaluation conditions

- Frozen manifest: exactly 401 independent source phrases and all 1,200 nested
  signer-specific references from the parent protocol.
- FP16 inference on Tesla V100S 32 GB for every newly generated Fixed Wait-k
  rollout.
- Greedy decoding, identical tokenizer revision, source tokenization, source
  order, action-state transitions, finite action budget, and latency reference
  length used for CausalGloss and TransLLaMA-style.
- Human references are never exposed to prompts, action histories, decoding,
  checkpoint selection, or k selection.
- The same content normalization and automatic human-reference scoring are
  used for every system.
- New outputs are written only under fixed_waitk directories. Existing paper
  and adaptive-comparison artifacts remain read-only.

## Predeclared analysis

The complete k={1,3,5,7} frontier is reported. CLERC outcomes are not used to
select a favorable k.

The primary Fixed Wait-k comparison is CausalGloss versus k=7 because k=7 was
the highest-quality fixed point on the independent locked-500 benchmark and
has three training seeds. This choice is frozen before Fixed Wait-k CLERC
evaluation.

- For k=5 and k=7, uncertainty uses 20,000 bootstrap draws with independent
  resampling of training seeds for CausalGloss and Fixed Wait-k, followed by
  paired resampling of the common source phrases. Seed IDs are not described
  as matched because the first fixed-policy seed differs from the first
  CausalGloss seed.
- For k=1 and k=3, only one submitted-paper Fixed Wait-k checkpoint exists.
  Intervals therefore resample paired source phrases after averaging the three
  CausalGloss seeds and do not estimate Fixed Wait-k training-seed variance.
- Primary quality is content-normalized best-human-reference Token F1.
- Primary latency is source-token LAAL with the frozen per-phrase median
  content-reference length.
- Completion, failure-penalized latency, no-output rate, invalid actions, and
  output length accompany every comparison.

The result is automatic zero-shot evaluation against human annotations, not an
expert preference study and not evidence of full ASL or non-manual correctness.

