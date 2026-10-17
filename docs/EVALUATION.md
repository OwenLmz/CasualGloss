# Final-paper evaluation

This document describes the paper's scoring inputs, metrics and statistical
protocols. Commands for data preparation, training and inference are in
[REPRODUCTION.md](REPRODUCTION.md).

## Experiment inputs

| Experiment | Inputs and unit of evaluation |
|---|---|
| Locked500 / Table 1 / Figure 1 | 500 parent examples per rollout; 35 outputs covering three CausalGloss seeds, eight fixed Wait-k runs and three learned-baseline seeds at k=1,...,8. |
| CLERC Epee | 401 prompts with 1,200 human references nested within prompts; 35 policy outputs. References are not independent experimental units. |
| Completed-only analysis | For each learned k=5 seed, select prompts where that baseline completes; compare both systems on the same selected prompts. This is a post-hoc sensitivity analysis, not the all-prompt main analysis. |
| Table 2 and controls | 250 diagnostic parents per output; full model, five trained component ablations, native no-recovery and guard-only controls. |
| Speech cascade | 500 examples per output; the lexical internal control and three CausalGloss seeds consume the shared recorded frontend events. |

The saved-prediction scoring utility reads an explicit 83-file input list:
82 prediction files containing 35,535 rows and one CLERC reference manifest.
The allowlist is in `scripts/verification/fetch_verification_inputs.py`.

## Metrics and aggregation

- **Token F1:** multiset token overlap for the weak-reference experiments.
  CLERC uses normalized content tokens and the best reference for each prompt.
- **Sequence similarity:** order-sensitive similarity, separate from token F1.
- **BLEU-4:** unsmoothed corpus BLEU-4 for Table 1; do not substitute an average
  of sentence BLEU scores or frontend BLEU with different tokenization.
- **LAAL:** length-adaptive average lagging from the recorded source positions.
  Speech reports source-time and computation-aware variants separately.
- **Completion:** the recorded END/full-source completion condition. Completion
  and acceptable output length are different criteria.
- **Seed statistics:** arithmetic means and sample SD where multiple independent
  training seeds are available. Fixed k=1/3 have one seed; fixed k=5/7 have three.

CLERC normalization removes QUESTION / trailing question-mark markers, strips
numbered variants and collapses exact periodic repetitions. Human references
remain nested under their original prompt throughout scoring.

The implementation of these definitions is
`scripts/verification/verify_paper_results.py`; the CLERC preprocessing and
rollout interfaces are also available in `evaluation/clerc/`.

## Uncertainty and paired comparisons

The learned-baseline and CLERC comparisons use the paper's hierarchical
seed/parent bootstrap with 20,000 draws. Fixed Locked500, Table 2 and speech
use their specified 10,000-draw procedures. The speech paper-interval route
preserves the original 14-metric order and shared Python RNG stream; the
evaluator also exposes the NumPy paired-parent interval calculation.

Paired LAAL differences use only parents with finite LAAL for both systems;
the corresponding sample counts must accompany the comparison. Do not subtract
two separately filtered means and call the result a paired difference.

The native no-recovery control has 249/250 completed outputs; the full model,
five trained component ablations and guard-only control have 100% completion.
Keep these conditions separate when interpreting completion or paired metrics.

## Generation settings

The paper runners preserve the selected adapters, batch sizes, arithmetic
precision, input order and stopping budgets in the experiment configurations.
Main CausalGloss seeds use `adapter_final`. The final independent learned
baseline is the selected SimAlign-trained version, decoded at k=1,...,8.

The original generation action cap depends on reference length: CLERC uses the
longest available reference length, while Locked500 and speech use their target
length. Human reference content is not inserted into the CLERC model history,
but the cap is reference-dependent. A deployment test without references needs
an independently specified budget rather than this evaluation cap.

The retained SeamlessStreaming launcher reads audio in 320-ms increments.
Audio-read positions, frontend commit times, policy compute time and real
end-to-end response time are different quantities; retain their units and
definitions when processing speech logs.

## Result files

`results/paper_results.json` contains aggregates of the frozen predictions used
for the submitted paper. It supplies the numerical inputs for portable Figure 1
rendering. `results/expected_paper_results.json` records rounded paper display
values, selected intervals and input counts for the scoring utility.

New predictions and generated scoring reports belong in a new external output
directory, not in place of these original-paper inputs. The code does not
require private texts, waveforms or model weights to be committed to Git.

Weak pseudo-gloss labels, human-reference overlap metrics and completion do
not by themselves measure expert ASL comprehensibility. The speech experiments
use synthetic Mandarin audio and retain output over-generation; pose/video
generation is outside this repository's task.
