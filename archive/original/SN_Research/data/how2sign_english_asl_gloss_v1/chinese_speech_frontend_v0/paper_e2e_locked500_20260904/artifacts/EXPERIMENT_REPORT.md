# Locked-500 Chinese-Speech End-to-End Evaluation

## Scope

- Fixed IDs: the pre-existing locked 500 test parents; no model or policy selection used these results.
- Input: deterministic English-to-Mandarin translation followed by four-speaker synthetic TTS.
- Frontend: the already selected SeamlessStreaming 2.5B monotonic S2TT system.
- Comparison: v1 versus v2.3 on identical timestamped incremental English commits.
- References: weak pseudo-ASL gloss, not expert gold ASL.

## Main results

| Metric | v1 | v2.3 | Paired delta | 95% CI |
|---|---:|---:|---:|---:|
| Pseudo-gloss token F1 | 0.3941 | 0.3946 | 0.0004 | [-0.0084, 0.0095] |
| Sequence similarity | 0.3739 | 0.3740 | 0.0001 | [-0.0084, 0.0087] |
| Exact match | 0.0260 | 0.0200 | -0.0060 | [-0.0160, 0.0020] |
| Valid completion | 0.9980 | 0.9980 | 0.0000 | [0.0000, 0.0000] |
| Reference-length AL (s) | 2.0904 | 1.6175 | -0.4729 | [-0.5956, -0.3611] |
| LAAL (s) | 2.5429 | 2.3909 | -0.1520 | [-0.2200, -0.0921] |
| DAL (s) | 3.2704 | 3.0643 | -0.2061 | [-0.3026, -0.1211] |
| Compute-adjusted LAAL (s) | 2.9506 | 2.8277 | -0.1229 | [-0.1933, -0.0627] |
| Processing RTF | 0.3877 | 0.4127 | 0.0250 | [0.0232, 0.0270] |
| Relative length error | 0.4258 | 0.5421 | 0.1163 | [0.0867, 0.1469] |

## Predeclared interpretation checks

- laal_improved: **True**
- compute_adjusted_laal_improved: **True**
- sequence_similarity_noninferior: **True**
- token_f1_noninferior: **True**
- completion_noninferior: **True**
- length_error_within_tolerance: **False**
- all checks accepted: **False**

## Frontend audit

- English BLEU: 21.2326
- English chrF: 44.7714
- English WER: 0.6539
- Valid audio files: 500/500

## Required limitations

1. Mandarin audio is synthetic, not recorded from real speakers or noisy environments.
2. Pseudo-gloss content metrics do not establish expert ASL correctness.
3. Speech-time latency excludes model runtime; compute-adjusted LAAL is reported separately.
4. Synthetic Mandarin was generated from the English reference, so this is controlled system evaluation rather than a natural bilingual corpus.
