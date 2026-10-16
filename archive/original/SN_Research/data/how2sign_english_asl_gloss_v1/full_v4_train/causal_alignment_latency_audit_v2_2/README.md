# Semantic Causal Alignment v2.2

## Scientific status

This experiment studies causal timing over Qwen-generated pseudo ASL glosses.
Neither the gloss tokens nor their emission times are gold ASL annotations.
The locked-test comparisons therefore measure pseudo-label imitation quality and
latency, not verified ASL translation accuracy.

## Motivation

The v2.1 experiment unexpectedly increased latency. A metric audit showed that
this was not an artifact of the legacy normalized AL alone:

| Locked 500 metric | v1 | v2.1 |
|---|---:|---:|
| Legacy normalized AL | 0.1575 | 0.2421 |
| Reference-length AL (source tokens) | 1.0661 | 3.0735 |
| LAAL (source tokens) | 2.5896 | 3.9548 |
| AP | 0.5597 | 0.6303 |
| DAL (source tokens) | 4.0642 | 6.2530 |
| Pseudo-gloss sequence similarity | 0.6894 | 0.6742 |
| Pseudo-gloss token F1 | 0.7016 | 0.6841 |

Two implementation causes were identified:

1. v2.1 forced strictly increasing emission positions before source completion,
   allowing at most one WRITE per observed source token.
2. Its rollout advanced the source after every action, including WRITE. Under
   standard READ/WRITE semantics, WRITE commits target output at the current
   prefix; only WAIT/READ consumes another source token.

## v2.2 policy

v2.2 uses a confident-prefix burst policy:

- exact and stem matches supply lexical anchors;
- Fine-tune 1 prefix likelihoods estimate when each gloss token becomes
  semantically supportable;
- a short stability test rejects transient confidence spikes;
- bounded rescue prevents weak tokens from delaying all later tokens;
- multiple monotonic gloss tokens may share one source position;
- negation is delayed until negative source evidence is observed when possible;
- difficult source/gloss orders are retained with explicit flags rather than
  silently deleted.

The policy is related to confidence-based prefix commitment in
[Adaptive Simultaneous Sign Language Translation with Confident Translation
Length Estimation](https://aclanthology.org/2024.lrec-main.34/). The ASL-specific
lexical anchoring, semantic rescue and negation handling are project-side
extensions. The multi-path alternative for future work is motivated by
[Hidden Markov Transformer](https://openreview.net/forum?id=9y0HFvaAYD6).

## Development-only selection

The parameters were selected on the prior 500-parent pilot development split,
not on the locked test set. The grid contains 864 configurations. A candidate
must satisfy:

- semantic sufficiency ratio at least 0.995;
- mean semantic gap no more than 0.10 nats;
- at most one late-negation diagnostic;
- normalized AL within 0.00002 of the lowest eligible latency.

Within the near-fastest set, semantic coverage is preferred. The selected
configuration is:

```json
{
  "lexical_mode": "gap_guarded",
  "semantic_margin": 0.75,
  "lexical_gap_margin": 0.25,
  "minimum_probability": 0.01,
  "stability_window": 1,
  "maximum_rescue_delay": 3,
  "rescue_gap_margin": 1.0
}
```

Its development metrics are normalized AL 0.14933, LAAL 2.1098 source tokens,
mean semantic gap 0.07496 nats and semantic sufficiency 0.99528.

## Full alignment data

| Item | Count |
|---|---:|
| Parents | 19,619 |
| Actions | 558,464 |
| Gloss tokens | 169,057 |
| Train parents / actions | 18,632 / 534,877 |
| Validation parents / actions | 491 / 12,256 |
| Test parents / actions | 496 / 11,331 |

Full-label normalized AL is 0.14892 and LAAL is 2.05873 source tokens. The
transition validator confirmed that WRITE retains the current source prefix and
WAIT advances exactly one token.

A separate leakage audit found 19,619 unique parents and 1,949 unique How2Sign
video groups, with zero parent-ID, video-group or exact-source overlap across
train, validation and test. No record was missing its source video ID.

## Prompt consistency correction (v2.2.1)

An audit found that the first v2.2 action builder manually detokenized English
prefixes (for example, `we 're`), whereas the locked evaluator and earlier data
used NLTK `TreebankWordDetokenizer` (`we're`). The initial run was stopped at
step 500 and retained under an explicitly named `aborted_...` directory. A
deterministic repair created separate v2.2.1 manifests without modifying the
original files. It changed 200,640 of 534,877 train prompts while preserving
every ID, target action, split and source-prefix length. The permanent builder,
delay-recovery builder and evaluator now share the same Treebank detokenizer.

## Optional delay-recovery calibration (v2.2.2)

Correct READ/WRITE transitions add a WAIT after the burst at each observed
source prefix. Consequently, WAIT comprises 66.2% of the primary v2.2.1 train
actions, compared with about 53% in v1/v2.1. This may create an overly
conservative class prior even when the timing labels themselves are earlier.

The optional v2.2.2 experiment is kept separate from the primary run and is
run as a prespecified recovery ablation. It:

- retains all regular WRITE and END states;
- selects WAIT states evenly across every source sentence;
- adds 46,614 off-policy WRITE states in which the source has advanced one to
  three tokens beyond a scheduled emission while committed gloss is behind;
- continues from the best v2.2.1 adapter at a lower learning rate.

The resulting 413,881-action calibration set is 45.1% WAIT, 50.4% WRITE and
4.5% END. This design is inspired by the delay-recovery SFT and WAIT-balance
motivation in [Hikari](https://arxiv.org/abs/2603.11578), but is an adaptation
to English-to-pseudo-ASL-gloss rather than a reproduction of its speech model.
It is never selected merely because it was run. It replaces v2.2.1 only when
the validation-only comparison shows lower LAAL while preserving valid
completion, pseudo-gloss quality and output length.

The predeclared decision rule is:

1. select between v2.2.1 and v2.2.2 using only the 491-parent validation
   split, requiring lower LAAL with non-inferior pseudo-gloss quality,
   valid completion, output length and invalid-output rate;
2. evaluate that validation-selected v2.2 variant once against v1 on the
   unchanged locked-500 test;
3. retain v1 if the selected variant does not improve the confirmatory
   latency-quality trade-off.

This rule prevents both test-set model selection and selecting a model merely
because it emits fewer tokens or because one aggregate metric happens to
improve.

## Locked evaluation protocol

The unchanged 500-parent v1 test split is used for free-running comparison.
Each model uses its own committed history. The report includes:

- pseudo-gloss exact match, sequence similarity and token F1;
- valid completion after the full source, premature END, invalid output and
  repetition rates;
- legacy prediction-length normalized AL for backwards compatibility;
- reference-length AL and its normalized form;
- LAAL and its normalized form;
- AP and DAL;
- paired 5,000-resample bootstrap intervals against v1.

LAAL is emphasized because length-adaptive AL is less vulnerable to apparently
low latency caused by under-generation. See the
[LAAL paper](https://arxiv.org/abs/2206.05807) and
[SimulEval evaluator documentation](https://simuleval.readthedocs.io/en/v1.1.0/user_guide/evaluator.html).

For strict model selection, an `<END>` action counts as a valid completion only
when the complete source has been observed. Premature `<END>` remains in the
files as a separate failure flag; it cannot inflate the completion criterion.

## Core reproducibility files

- `scripts/audit_latency.py`: v1/v2.1 diagnosis.
- `scripts/tune_confident_burst_policy.py`: deterministic development grid.
- `scripts/compare_locked_rollouts.py`: paired multi-version comparison.
- `../semantic_causal_alignment_v2_2_full/scripts/build_confident_burst_alignment_v2_2.py`:
  full label builder.
- `../semantic_causal_alignment_v2_2_full/scripts/split_and_validate_v2_2.py`:
  split and transition validation.
- `../semantic_causal_alignment_v2_2_full/scripts/audit_split_leakage.py`:
  strict parent/video-group leakage audit.
- `../fine_tuning_v2_semantic_v2_2_full/scripts/train_qwen_lora_causal_ddp.py`:
  LoRA training.
- `../fine_tuning_v2_semantic_v2_2_full/scripts/evaluate_free_running_rollout.py`:
  corrected burst-capable rollout and standard latency metrics.
- `../fine_tuning_v2_semantic_v2_2_full/scripts/test_latency_metrics.py`:
  deterministic metric checks.
- `../fine_tuning_v2_semantic_v2_2_full/scripts/build_wait_balanced_manifest.py`:
  optional deterministic WAIT-state selection with failure checks.
- `../fine_tuning_v2_semantic_v2_2_full/scripts/build_delay_recovery_actions.py`:
  optional off-policy overdue-WRITE construction.
- `../fine_tuning_v2_semantic_v2_2_full/scripts/build_recovery_calibration_manifest.py`:
  audited merge of balanced and delay-recovery states.
