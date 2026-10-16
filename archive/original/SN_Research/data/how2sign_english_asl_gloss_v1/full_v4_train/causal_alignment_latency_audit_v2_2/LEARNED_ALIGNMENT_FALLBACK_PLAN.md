# Learned Alignment Fallback Plan

This plan is activated only if neither v2.2.1 nor v2.2.2 passes the
predeclared locked-test latency-quality criterion.  It avoids further manual
threshold tuning on the test set.

## v2.3: local agreement over confident translation prefixes (implemented)

For each English prefix, the frozen Fine-tune 1 text-to-pseudo-gloss model is
run with deterministic decoding. The online policy compares a fixed window of
consecutive prefix translations and commits only their longest common gloss
prefix. Commits are irreversible. Any later prediction that conflicts with
already committed output is retained as an explicit conflict rather than
silently repaired. At source completion, the uncommitted suffix of the final
Fine-tune 1 translation is emitted before `<END>`.

The implementation is in `local_agreement_causal_v2_3`. It evaluates windows
2, 3 and 4 with minimum prefix lengths 1, 2 and 3. The configuration is selected
on all 500 validation parents by maximizing pseudo-gloss sequence quality under
two constraints: LAAL must not exceed the separately measured v1 validation
LAAL, and the irreversible-conflict parent ratio must not exceed 0.20. If no
candidate passes, the locked test is not evaluated. If one passes, exactly the
selected configuration is evaluated on all 500 locked test parents.

This follows the training-data principle in [Adaptive Simultaneous Sign
Language Translation with Confident Translation Length
Estimation](https://aclanthology.org/2024.lrec-main.34/), adapted from
sign-video-to-text to English-text-to-pseudo-ASL-gloss, and the local-agreement
principle used in retranslation. It directly estimates how much output is
stable rather than assigning each token one hand-selected emission position.
This is an adaptation, not a claimed reproduction of the cited system.

## v2.4: multiple latent emission candidates

For each target gloss token, retain several monotonic source-prefix candidates
instead of one hard pseudo position.  Optimize the target likelihood
marginalized over those positions, or distill the resulting posterior into a
READ/WRITE policy.  This is motivated by [Hidden Markov Transformer for
Simultaneous Machine Translation](https://openreview.net/forum?id=9y0HFvaAYD6),
which treats emission moments as latent events.

The candidate window should be built only from lexical and semantic evidence;
negation and entity constraints remain hard safety bounds.  Validation chooses
the latency weight and candidate-window width.  The locked test remains
untouched until one configuration is selected.

## v2.5: explicit information sufficiency

Train a lightweight head to estimate whether the current source prefix contains
enough information for the next gloss.  Positive and negative states can be
derived from prefix/full-source teacher predictions.  This is related to the
Information Quantifier in [Quantifying Information of Tokens for Simple and
Flexible Simultaneous Machine
Translation](https://aclanthology.org/2023.conll-1.14/).

## Additional policy-learning references

If a dedicated policy head is required, three complementary directions are
retained for an ablation rather than mixed into one untestable method:

- [DrFrattn](https://aclanthology.org/2025.emnlp-main.1767/) learns an adaptive
  policy directly from a trained model's attention. For this project, the
  corresponding experiment would predict WAIT/WRITE from Fine-tune 1 attention
  statistics while keeping the gloss generator frozen.
- [Self-Modifying State Modeling](https://aclanthology.org/2024.acl-long.528/)
  motivates sampling alternative prefix/commit histories so the policy sees
  more than the single oracle trajectory. The present v2.2.2 delay-recovery
  states are a narrow controlled version of this idea, not a reproduction.
- [PsFuture](https://aclanthology.org/2024.emnlp-main.111/) provides a zero-shot
  pseudo-future policy and Prefix-to-Full adaptation. It is suitable as a
  no-extra-policy-head baseline if the trained action policy remains unstable.

These methods are activated only after validation evidence; the locked test is
not used to select among them.

## Evaluation contract

- select checkpoints and method variants using validation only;
- test once on the unchanged 500-parent split;
- report LAAL, reference-length AL, AP and DAL together;
- require completion and output-length non-inferiority;
- require pseudo-gloss sequence-similarity and token-F1 non-inferiority;
- keep the label status explicit: pseudo gloss and pseudo timing, not gold ASL.
