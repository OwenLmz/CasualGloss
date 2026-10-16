# CLERC Epee Adaptive-Policy Comparison: Audited Results

Status: complete and reproducible. No manuscript text has been changed.

## Scope and integrity

- Zero-shot external evaluation against human-annotated CLERC Epee gloss
  references.
- 401 independent source phrases; 1,200 signer references remain nested within
  phrase and are not treated as independent samples.
- CausalGloss and TransLLaMA-style results use the same three matched training
  seeds: 20260829, 20260904, and 20260905.
- All new rollouts use the same Qwen2.5-7B revision, V100 FP16 environment,
  greedy decoding, source order, shared action budget, and latency reference
  lengths.
- The confirmatory learned-baseline point was fixed at k=5 before CLERC
  inference. The complete k=1,...,8 sweep is descriptive.
- All 28 full rollout files passed ID, source-token, transition, action-budget,
  dtype, and reference-exposure audits: four action-policy files plus 24
  learned-baseline seed/k files.
- The fixed FP16/BF16 smoke gate passed. Aggregate content Token F1 drift was
  0.0000 for both comparable checkpoints; LAAL drift was 0.0347 for lexical v1
  and 0.0000 for CausalGloss seed 20260829, below the frozen 0.10 threshold.
- Repeating the complete 20,000-draw analysis produced byte-identical reports.

## Confirmatory result: CausalGloss versus TransLLaMA-style k=5

| Outcome | CausalGloss, 3-seed mean | Learned k=5, 3-seed mean | Causal - learned (95% hierarchical bootstrap CI) |
|---|---:|---:|---:|
| Content best-reference Token F1 | 0.64898 | 0.54885 | +0.10012 [+0.08168, +0.11844] |
| Source-token LAAL | 1.53880 | 3.68486 | -2.15309 [-2.35247, -1.99013] |
| Valid completion rate | 1.00000 | 0.86949 | +0.13051 [+0.10973, +0.15295] |

The LAAL reduction relative to learned k=5 is 58.24%. The quality CI is above
both zero and the frozen -0.03 non-inferiority margin. The latency CI is wholly
below zero. The failure-penalized LAAL analysis agrees: -2.14570, 95% CI
[-2.34514, -1.98314]. Therefore the frozen strong-support rule passes.

## Complete learned operating curve

Across k=1,...,8, learned-baseline mean Token F1 ranges from 0.54151 to
0.55646 and LAAL ranges from 2.78361 to 5.29558. CausalGloss has both higher
Token F1 and lower LAAL at every learned operating point. Every descriptive
hierarchical CI excludes zero in the direction favoring CausalGloss.

The learned baseline also exposes a meaningful external-transfer failure mode:
depending on k, only about 85.8%-88.3% of examples terminate normally, and
roughly 11.2%-14.0% reach the shared maximum action count. At k=5, the
three-seed mean invalid-action ratio is 0.04817 and the maximum-action rate is
0.12718. These failures are retained, not removed.

## Lexical-policy control

| Outcome | CausalGloss, 3-seed mean | Lexical v1, single checkpoint | Causal - lexical (95% paired-source CI) |
|---|---:|---:|---:|
| Content best-reference Token F1 | 0.64898 | 0.64082 | +0.00816 [-0.00432, +0.02130] |
| Source-token LAAL | 1.53880 | 1.42200 | +0.11914 [+0.05897, +0.17948] |
| Valid completion rate | 1.00000 | 0.99751 | +0.00249 [0.00000, +0.00748] |

This control is important. The external Epee result strongly supports
CausalGloss over the learned TransLLaMA-style baseline, but it does not show
that CausalGloss is superior to the lexical policy. The F1 difference from the
single lexical checkpoint is inconclusive and the lexical checkpoint is faster
on Epee. This must not be hidden if the new experiment is reported.

## Defensible interpretation

The experiment provides strong evidence that the submitted CausalGloss action
policy transfers more robustly than the independently trained TransLLaMA-style
learned adaptive baseline to external human-annotated gloss references, with
substantially better quality, lower source-token latency, and fewer completion
failures. It does not establish expert human preference, full ASL correctness,
or superiority over the lexical policy.

Recommended wording boundary:

> On zero-shot CLERC Epee evaluation against human-annotated gloss references,
> CausalGloss outperformed the TransLLaMA-style learned baseline at every
> predeclared k, including a +0.100 Token F1 and 58.2% lower LAAL at the fixed
> k=5 comparison. The lexical control remained competitive, indicating that
> Epee primarily supports robust human-gloss overlap rather than a universal
> advantage over all action policies.
