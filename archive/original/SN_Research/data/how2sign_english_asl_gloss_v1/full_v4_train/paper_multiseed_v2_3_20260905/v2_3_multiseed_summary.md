# v2.3 Three-Seed Locked-500 Summary

Seeds: 20260829, 20260904, 20260905. Values are mean +/- sample standard deviation across three training seeds.

## Text locked-500

| Metric | v2.3 mean +/- SD | Wait-k v1 k=1 |
|---|---:|---:|
| Token F1 | 0.7047 +/- 0.0055 | 0.7041 |
| Sequence similarity | 0.6957 +/- 0.0060 | 0.6927 |
| LAAL (source tokens) | 1.7823 +/- 0.2031 | 2.5690 |
| Completion | 1.0000 +/- 0.0000 | 1.0000 |

All three seeds satisfy the predeclared quality-noninferiority margin and have significantly lower LAAL than modified Wait-k k=1.

## Chinese speech end-to-end locked-500

| Metric | v2.3 mean +/- SD | v1 |
|---|---:|---:|
| Token F1 | 0.3944 +/- 0.0003 | 0.3941 |
| Sequence similarity | 0.3750 +/- 0.0010 | 0.3739 |
| LAAL (seconds) | 2.4149 +/- 0.0418 | 2.5429 |
| Compute-adjusted LAAL (seconds) | 2.8241 +/- 0.0126 | 2.9506 |
| Completion | 0.9980 +/- 0.0000 | 0.9980 |

All three seeds preserve quality within the predeclared margin and significantly reduce both LAAL and compute-adjusted LAAL relative to v1. The output-length-error tolerance is not met and must be reported as a limitation.
