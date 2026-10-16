# Fixed Wait-k baselines v1

This directory contains the fixed-policy baselines for the ICASSP 2027
simultaneous English-to-pseudo-ASL-gloss experiments.

Scientific controls:

- The immutable parent splits are reused from `causal_alignment_v1_corrected_full`.
- Train, validation, and locked test parent IDs remain disjoint.
- The policy is the standard prefix-to-prefix schedule
  `g(t) = min(k + t - 1, |x|)`.
- `WRITE` keeps the current source prefix and `<WAIT>` consumes exactly one
  additional source token, matching the corrected v2.2 state semantics.
- Targets remain Qwen-generated pseudo glosses, not gold ASL annotations.
- Pilot and 5K experiments use development data only. The locked 500-parent
  test set is evaluated only after the full baseline configuration is frozen.

Candidate schedules are prepared for `k in {1, 3, 5, 7}`. To conserve the
pre-submission compute budget, the current primary learned baseline is
`Wait-k=3`, using the same Qwen2.5-7B LoRA initialization, optimization budget,
decoding, and metrics as the causal models. Additional full-data k values are
optional latency-curve experiments, not automatic pipeline stages.

The final paper comparison is generated only after the configuration is frozen.
It evaluates v1, v2.1, v2.2, and Wait-k=3 on the same locked 500-parent test
split with paired bootstrap confidence intervals. No Wait-k value is selected
or tuned using the locked test set.
