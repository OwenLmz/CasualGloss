# Semantic Causal Alignment v2 pilot results

## Completion status

- 500 proportionally stratified training parents were scored.
- 4,306 pseudo-gloss tokens were aligned.
- 10,512 Fine-tune 2 action examples were generated.
- Structural validation passed with zero missing IDs, duplicate parents, malformed
  trajectories, or non-final `<END>` actions.

## Primary balanced-policy result

Relative to lexical/SimAlign v1, Semantic Causal Alignment v2 changed the scheduled
emission position of 52.44% of pseudo-gloss tokens. The mean absolute shift was 2.45
source tokens. Most changes were later emissions (1,902 tokens), while 356 tokens moved
earlier and 2,048 remained unchanged.

The internal semantic-evidence gap at emission decreased from 0.2600 to 0.0225 nats.
The paired 5,000-resample parent bootstrap estimate for v2 minus v1 was -0.2375 nats,
with a 95% confidence interval of [-0.2748, -0.2039]. The proportion of tokens meeting
the frozen balanced semantic-sufficiency criterion increased from 91.69% to 99.79%.

This consistency gain required more waiting. Mean normalized word-position Average
Lagging increased from 0.1558 to 0.1982. The paired estimate was +0.0424, with a 95%
confidence interval of [0.0305, 0.0540].

## Frozen policy sensitivity

| Policy | Mean normalized AL | Mean semantic gap (nats) | Mean full-source emissions per parent |
|---|---:|---:|---:|
| Low latency | 0.1635 | 0.0371 | 0.650 |
| Balanced (primary) | 0.1982 | 0.0225 | 1.144 |
| High confidence | 0.2732 | 0.0126 | 2.626 |

The low-latency policy is a useful downstream ablation: its latency is close to v1 while
its internal semantic gap is much smaller. It was not substituted for the predeclared
balanced primary result after observing the data.

## Interpretation limits

These results show that model-derived prefix timing is more consistent with the frozen
Fine-tune 1 model's full-source evidence than v1 lexical timing. They do not establish
gold ASL correctness. Both the target glosses and their timings remain pseudo labels,
and the scorer is not an independent evaluator. Qualified ASL review and downstream
Fine-tune 2 held-out experiments are needed before claiming translation improvement.
