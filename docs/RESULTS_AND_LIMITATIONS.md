# Results and evidence boundaries

Reference: the submitted September 23 PDF identified in `provenance/paper_anchor.json`. Numeric release checks are reported separately in `docs/VERIFICATION.md` and `results/`; this page is a map of the paper's claims, not a new experiment.

| Locked500, three-seed mean ± sample SD | Fixed k=7 | Learned k=5 | CausalGloss |
|---|---:|---:|---:|
| Token F1 | .6457 ± .0010 | .7067 ± .0056 | .7047 ± .0055 |
| Sequence similarity | .6383 ± .0008 | .6955 ± .0051 | .6957 ± .0060 |
| BLEU-4 | 33.90 ± .19 | 28.35 ± .33 | 37.31 ± 2.88 |
| LAAL, source tokens | 4.3099 ± .0528 | 3.9061 ± .0458 | 1.7823 ± .2031 |

For CLERC Epee, CausalGloss mean F1 is .6490 and LAAL 1.5388 tokens, versus learned k=5 F1 .5489 and LAAL 3.6849. The post-hoc completed-baseline subset has F1 .6196 versus .5839, difference .0357, 95% CI [.021,.051], n=344–353 per seed. This subset analysis must remain explicitly post-hoc.

Speech mean F1 is .3944, raw LAAL 2.415 ± .042 seconds, completion 99.8%. The predeclared S1 comparison has a raw .152-second latency gain and a computation-aware .123-second gain. The output-length criterion fails for all three seeds: **over-generation remains a system-level failure**.

## What these results do not prove

- Weak pseudo-gloss overlap is not expert ASL correctness.
- Human-gloss best-reference overlap is not a comprehension study or non-manual-sign assessment.
- This is a task-adapted TransLLaMA-style comparison, not an exact reproduction of the original full system.
- The CLERC k=5 comparison is confirmatory under the frozen protocol; the broader k sweep is descriptive.
- Good latency does not excuse incomplete outputs; completion and length diagnostics must be preserved.
- No gloss-to-pose/video system is evaluated in this submitted paper.
- Reaggregating saved outputs is not retraining, and a small checkpoint smoke test is not a new full-benchmark replication.
