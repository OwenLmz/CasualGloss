# Full Fine-tune 1 results

## Setup

- Train/validation/test: **27079 / 1516 / 1507**.
- Splits are request-disjoint and How2Sign source-video-disjoint.
- Model: `Qwen/Qwen2.5-7B-Instruct` with rank-16 LoRA on `q_proj` and `v_proj`.
- Best checkpoint: step **1693**, selected only by validation pseudo-label loss.

## Held-out test comparison

| Metric | Base model | Best LoRA | Delta |
|---|---:|---:|---:|
| Pseudo-label loss | 0.7778 | 0.2464 | -0.5314 |
| Teacher-forced token accuracy | 0.8249 | 0.9140 | +0.0892 |
| Greedy generation exact match | 0.0540 | 0.2540 | +0.2000 |
| Greedy generation sequence similarity | 0.5941 | 0.7210 | +0.1269 |

## Interpretation

These metrics test reproduction of deterministic pseudo glosses on unseen
How2Sign video groups. They are not gold ASL accuracy and do not evaluate sign
video generation or qualified linguistic correctness.
