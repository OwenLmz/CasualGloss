# Paper-scope extract: CLERC Epee frozen comparison protocol

This extract keeps the final paper's comparison methods. The unreported CLERC
lexical comparator is omitted; the frozen inference/scoring/statistical rules
for the reported methods below are unchanged. This is not a new protocol freeze.

Status: frozen before implementation smoke tests and before any new CLERC
rollout is generated.

Date frozen: 2026-09-20 (Asia/Shanghai)

## Scientific question

Does the submitted CausalGloss policy retain quality while reducing source-token
latency relative to the independently trained TransLLaMA-style learned adaptive
baseline when both are evaluated zero-shot against external human-annotated
CLERC Epee gloss references?

This is an automatic evaluation against human annotations. It is not an expert
human preference study and does not establish full ASL grammatical or
non-manual correctness.

## Frozen data and independent unit

- Dataset: CLERC Epee v0.3, commit
  `93bc8aa0a6526e86af80db5f59af8d133292c9b5`.
- Frozen manifest:
  `clerc_epee_external_eval_v1/manifests/epee_sources_401.jsonl`.
- Exactly one inference example per `phrase_id`: 401 independent source
  phrases.
- All signer-specific glosses are retained as nested references. The 1,200
  clip-level references are not treated as 1,200 independent observations.
- No Epee example or reference may be used for training, checkpoint selection,
  hyperparameter selection, prompt selection, or selection of inference `k`.

## Frozen systems

### CausalGloss

Use the same three submitted-paper training seeds and the same selected
`adapter_final` checkpoints used in the locked-500 paper comparison:

- `20260829`: `state_recovery_v2_3_full/.../adapter_final`
- `20260904`: `state_recovery_v2_3_multiseed_20260904/.../adapter_final`
- `20260905`: `state_recovery_v2_3_multiseed_20260905/.../adapter_final`

The native CausalGloss policy starts from one English source token. There is no
post-hoc latency parameter sweep for CausalGloss.

### TransLLaMA-style learned adaptive baseline

Use the same three submitted-paper training seeds and selected `adapter_best`
checkpoints:

- `20260829`
- `20260904`
- `20260905`

Run the complete predeclared minimum-initial-read sweep `k = 1,...,8`. The
confirmatory comparison is fixed at `k = 5`, because it was the nearest-quality
point on the independent locked-500 benchmark before this CLERC experiment.
CLERC outcomes must not be used to select another `k`. The complete sweep is
reported descriptively.

## Shared inference conditions

- Frozen backbone: `Qwen/Qwen2.5-7B-Instruct`, revision
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- Hardware for this replication: Tesla V100S 32 GB.
- Precision: FP16 for every newly generated rollout from every system.
- Greedy decoding; no sampling.
- Same tokenizer revision and each checkpoint's frozen system prompt.
- Same Epee source tokenization and source order.
- Same finite action budget for all systems:
  `source_length + max(30, 4 * max_human_reference_length + 10)`.
- Same latency reference length for all systems: the frozen per-phrase median
  content-normalized human-reference length.
- No human reference is exposed to the model prompt, state, or decoding loop.
- Invalid actions are recorded. Before source completion they advance exactly
  one source token as a finite-progress safeguard, matching the submitted
  evaluators.
- New outputs are written only to this experiment directory. Existing
  checkpoints, manifests, rollouts, reports, and submitted-paper artifacts are
  read-only and must never be overwritten.

## Precision compatibility gate

Before the full run, execute a deterministic, hash-selected smoke manifest.
The smoke set is selected without inspecting model outputs.

The gate requires:

1. every checkpoint loads and produces structurally valid rollout records;
2. source IDs and source tokens match the frozen manifest;
3. no human reference enters rollout history;
4. action transitions pass invariant checks;
5. all compared methods use FP16 under the same software stack;
6. on models with existing BF16 rollouts, FP16/BF16 differences are quantified
   on fixed examples before full evaluation. Material metric drift or a failure
   of action invariants stops the experiment for diagnosis; the gate is not
   weakened after seeing results.

Exact token equality is recorded but is not required if aggregate smoke-set
quality and latency remain within the predeclared numerical tolerance of 0.01
absolute Token F1 and 0.10 source-token LAAL. Any exception must be reported.

## Outcomes

### Primary quality endpoint

Per-phrase content-normalized best-human-reference Token F1. Content
normalization removes `QUESTION`, maps trailing `_N` signer variants to their
base gloss, and collapses only exact whole-sequence repetition, matching the
existing frozen Epee evaluation.

### Primary latency endpoint

Length-adaptive average lagging (LAAL) in English source tokens, using the same
frozen per-phrase median content-reference length for every system.

### Secondary endpoints

- strict best-reference Token F1;
- mean-over-reference Token F1;
- best and mean sequence similarity;
- minimum and mean gloss error rate;
- precision, recall, predicted length, relative length error, and exact match;
- completion rate, termination reasons, invalid-action rate, action count, and
  over-generation diagnostics;
- BLEU only as exploratory supporting information.

## Confirmatory comparison and decision rule

The single confirmatory comparison is CausalGloss versus TransLLaMA-style
`k = 5`.

- Quality retention is supported only if the 95% hierarchical bootstrap CI for
  `CausalGloss - TransLLaMA-style` best-reference content Token F1 has lower
  bound greater than the predeclared non-inferiority margin `-0.03`.
- Latency improvement is supported only if the entire 95% hierarchical
  bootstrap CI for `CausalGloss - TransLLaMA-style` LAAL is below zero.
- Strong support requires both conditions and acceptable completion behavior.
- A point estimate alone is not evidence of equivalence or superiority.

All other `k` comparisons are descriptive. No multiplicity-adjusted
confirmatory claim is made over the `k = 1,...,8` sweep.

## Statistical analysis

- CausalGloss and TransLLaMA-style use matched training seed IDs
  `20260829`, `20260904`, and `20260905`.
- The independent evaluation unit is source phrase.
- Primary uncertainty uses 20,000 hierarchical bootstrap draws: resample the
  three matched seed pairs, then resample paired source phrases within each
  selected seed pair.
- Quality for empty output is zero. Completion and no-output rates are reported
  separately.
- Primary LAAL uses phrases with finite paired LAAL and reports the paired
  count. A failure-sensitive analysis and completion rates accompany it so
  exclusion of no-write cases cannot be hidden.

## Reporting boundary

Permitted wording if supported:

`zero-shot external evaluation against human-annotated gloss references`.

Not permitted:

- `expert human evaluation`;
- `human preference study`;
- `gold proof of ASL correctness`;
- claims about non-manual markers or full natural-signing quality.

All protocol deviations, failed runs, precision differences, incomplete
outputs, and unfavorable results must be retained and reported to the authors.
