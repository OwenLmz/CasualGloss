# CLERC Epee external ASL-gloss evaluation v1

This experiment evaluates the existing v1 and v2.2 causal-action adapters on
the same source-level CLERC Epee v0.3 prompts. It does not train or select a
model on Epee.

## Frozen protocol

- Dataset commit: `93bc8aa0a6526e86af80db5f59af8d133292c9b5`
- One inference example per `phrase_id` (401 sources), not one per clip.
- All signer-specific gloss sequences remain references.
- Primary quality: best-of-human-reference content-normalized Token F1.
- Strict quality keeps CLERC ID-gloss labels and `QUESTION` markers.
- Content normalization removes `QUESTION`, maps trailing `_N` sign variants
  to their base gloss, and collapses only exact whole-sequence repetition.
- v1 and v2.2 use their native learned WAIT/WRITE policy with an initial
  one-token English prefix.
- Paired source-level percentile bootstrap: 10,000 repetitions.
- Predeclared non-inferiority margin: -0.03 absolute Token F1.

CLERC Epee contains human ASL clip glosses, not a purpose-built text-to-gloss
translation benchmark. Repetition and annotation-convention sensitivity are
therefore reported explicitly rather than hidden.

