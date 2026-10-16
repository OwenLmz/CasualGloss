# ICASSP 2027 Experiment Completion Report

Date frozen: 2026-09-03

## Primary research question

Can a learned, ASL-oriented causal alignment policy retain pseudo-gloss quality while reducing latency relative to native earlier versions and a TransLLaMA-style modified Wait-k inference baseline?

## Completed evidence

1. Frozen locked-500 comparison of v1, v2.1, v2.2, and v2.3.
2. Complete modified Wait-k inference sweep for k=1 through 8 on v1 and v2.2.
3. Predeclared paired comparison of v2.3 against the highest-quality modified Wait-k point, v1 with k=1.
4. Frozen confirmation-set ablation of v2.2 native, inference guard only, and state-recovery-trained v2.3.
5. One-shot zero-shot evaluation against 401 unique English sources in the human-gloss CLERC Epee corpus.
6. End-to-end pilot on 100 synthetic Mandarin utterances using the same SeamlessStreaming outputs for v1 and v2.3.
7. Post-hoc qualitative case selection after aggregate metrics were frozen.
8. Reproducibility manifest containing hashes, model revision, data split hashes, seeds, software versions, and hardware records.

## Main locked-500 result

| System | Token F1 | Sequence similarity | LAAL | Valid completion |
|---|---:|---:|---:|---:|
| v1 native | 0.7016 | 0.6894 | 2.5896 | 100.0% |
| v2.1 native | 0.6841 | 0.6742 | 3.9548 | 99.8% |
| v2.2 native | 0.6220 | 0.6149 | 1.8430 | 99.4% |
| v2.3 learned causal policy | 0.7016 | 0.6925 | 1.6654 | 100.0% |
| v1 + modified Wait-k k=1 | 0.7041 | 0.6927 | 2.5690 | 100.0% |

Relative to v1 native, v2.3 changes Token F1 by -0.00002 with 95% CI [-0.0141, 0.0134], while reducing LAAL by 0.9242 source words with 95% CI [-1.2172, -0.6731].

Relative to the strongest modified Wait-k point, v1 with k=1, v2.3 changes Token F1 by -0.0025 with 95% CI [-0.0161, 0.0111]. It satisfies the fixed 0.03 quality non-inferiority margin. LAAL is lower by 0.9037 source words with 95% CI [-1.2013, -0.6397].

This supports the narrow primary claim: on the frozen pseudo-gloss test, v2.3 retains quality and significantly reduces latency relative to the best-quality modified Wait-k operating point.

## Ablation result

On the frozen 250-parent confirmation set:

- v2.3 versus v2.2 native: Token F1 +0.0752, 95% CI [0.0488, 0.1023]; LAAL -0.3065, 95% CI [-0.5591, -0.0458].
- v2.3 versus inference guard only: Token F1 +0.0382, 95% CI [0.0156, 0.0624]; LAAL -0.2001, 95% CI [-0.4336, 0.0425].

The first comparison supports the full state-recovery method. The second shows a clear quality gain over inference-only repair, while its latency CI crosses zero.

## External human-gloss result

On 401 unique CLERC Epee English sources, using best content Token F1 across available human gloss references:

- v1: 0.6418
- v2.2: 0.5932
- v2.3: 0.6450

v2.3 versus v1 is +0.0032 with 95% CI [-0.0112, 0.0179], satisfying the fixed -0.03 non-inferiority margin. v2.3 versus v2.2 is +0.0518 with 95% CI [0.0355, 0.0687].

Epee latency is not the primary endpoint. Its clip-level domain and sign-order differences make it unsuitable as a direct text-to-gloss latency benchmark. The all-401 LAAL for v2.3 is higher than v1 by 0.1708 source words, 95% CI [0.0830, 0.2597]. This limitation must be reported.

## Chinese-speech end-to-end pilot

The same 100 synthetic Mandarin utterances and the same committed SeamlessStreaming English prefixes were used for both systems.

| System | Token F1 | Sequence similarity | Speech-time LAAL | Compute-adjusted LAAL | Completion |
|---|---:|---:|---:|---:|---:|
| v1 | 0.3996 | 0.3839 | 2.6377 s | 3.1911 s | 100.0% |
| v2.3 | 0.4211 | 0.3983 | 2.4776 s | 3.1571 s | 100.0% |

Paired effects:

- Token F1: +0.0215, 95% CI [0.0031, 0.0405].
- Algorithmic speech-time LAAL: -0.1601 s, 95% CI [-0.2550, -0.0717].
- Compute-adjusted LAAL: -0.0341 s, 95% CI [-0.1310, 0.0569].

The learned policy reduces algorithmic waiting and improves pseudo-gloss overlap in this pilot. Its model processing real-time factor is higher, 0.5940 versus 0.4889, so the compute-adjusted latency gain is not statistically clear. Algorithmic and implementation latency must be reported separately.

## Qualitative error analysis

Among the 500 frozen examples, the post-hoc selector found:

- 91 cases with preserved quality and at least 0.5 source-word lower LAAL.
- 52 cases with at least +0.10 Token F1 and lower LAAL.
- 6 cases with earlier first emission and at least -0.10 Token F1.
- 46 successful cases containing same-prefix burst WRITE.

Two examples per category were frozen for qualitative analysis. These cases are descriptive and are not used for metric estimation.

## Objective paper-readiness assessment

The experiment package is sufficient to begin manuscript writing now. The strongest defensible story is not that the system solves ASL translation, but that state-recovery causal training repairs the quality collapse of an aggressive low-latency policy and moves the system to a better quality-latency operating point than modified Wait-k on the locked benchmark.

The paper must not claim expert-level ASL correctness. The main How2Sign-derived targets are pseudo glosses, the external Epee task is not a direct text-to-gloss benchmark, and the Mandarin speech pilot is synthetic. These are important limitations, not minor footnotes.

## Frozen artifacts

- `artifacts/tables/table_main_locked500.tsv`
- `artifacts/tables/table_modified_waitk_sweep.tsv`
- `artifacts/tables/table_ablation_confirmation250.tsv`
- `artifacts/tables/table_epee_external401.tsv`
- `artifacts/tables/table_synthetic_mandarin_e2e100.tsv`
- `artifacts/tables/table_key_paired_effects.tsv`
- `artifacts/figures/fig_quality_latency_frontier.{png,pdf,svg}`
- `artifacts/figures/fig_paired_effects.{png,pdf,svg}`
- `artifacts/reproducibility_manifest.json`

