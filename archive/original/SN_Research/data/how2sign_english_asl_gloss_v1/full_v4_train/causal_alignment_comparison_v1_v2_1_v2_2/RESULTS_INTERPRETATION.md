# Unified Locked-500 Interpretation

## Scope

All three systems are evaluated on the same 500 parent IDs, identical English source text, identical source tokens, and identical fixed pseudo-gloss references.
The reference is not expert-verified gold ASL, so these results measure pseudo-label imitation and streaming behavior.

## Mean results

| Metric | v1 | v2.1 | v2.2 |
|---|---:|---:|---:|
| Sequence Similarity | 0.6894 | 0.6742 | 0.6149 |
| Token F1 | 0.7016 | 0.6841 | 0.6220 |
| Reference-length AL | 1.0661 | 3.0735 | 0.8449 |
| LAAL | 2.5896 | 3.9548 | 1.8430 |
| Valid Completion Rate | 1.0000 | 0.9980 | 0.9940 |
| Invalid Output Ratio | 0.0001 | 0.0000 | 0.0000 |

## Paired v2.2 versus v1

| Metric | Delta | 95% CI | Conclusion |
|---|---:|---:|---|
| Sequence Similarity | -0.0745 | [-0.0952, -0.0545] | statistically_clear_regression |
| Token F1 | -0.0796 | [-0.1001, -0.0591] | statistically_clear_regression |
| Reference-length AL | -0.0672 | [-0.3822, 0.2305] | no_statistically_clear_difference |
| LAAL | -0.6027 | [-0.8550, -0.3654] | statistically_clear_improvement |
| Valid Completion Rate | -0.0060 | [-0.0140, 0.0000] | no_statistically_clear_difference |
| Invalid Output Ratio | -0.0001 | [-0.0002, 0.0000] | no_statistically_clear_difference |

## Main conclusion

v2.2 produces statistically shorter LAAL than v1; its mean reference-length AL is lower, but that AL difference is not statistically clear.
Sequence Similarity and Token F1 are both statistically lower for v2.2.
The result is therefore a latency-quality trade-off rather than an unqualified system improvement.
Valid completion and invalid-output behavior should be interpreted from their paired confidence intervals rather than raw means alone.
