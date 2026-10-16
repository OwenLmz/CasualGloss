# Release status — 2026-10-16

This document distinguishes completed preservation from remaining reproducibility and distribution work. Scope: the submitted CausalGloss gloss paper; no later pose/video extension. The submitted PDF's hash was checked again after the audit and remains unchanged.

## Completed

- Reviewed original-source preservation: **692** small original files collected; **690** source/config/protocol/metadata files are included in this Git source release (3,103,843 bytes). Two Markdown files containing real source/reference/prediction case studies are withheld from public distribution and preserved privately, unmodified. See `provenance/withheld_files.json` for identities only.
- All included original bytes match recorded SHA-256. Original model/data-generating algorithms were not edited; new wrappers and verification code are separate.
- Data/model register: **449** entries, including 28 selected LoRA groups, exact frozen train/development/test data, alignments, prefix-score outputs, corpus and speech references. Every selected LoRA binary was hashed in place. Large assets remain separately managed, not in Git.
- Historical output verification: **91** explicitly downloaded files match remote SHA-256; **83** raw rollouts / **35,936** records independently rescored. Table 1, Fig. 1, CLERC, completed-only, Table 2 point estimates and speech results match the submitted paper at its display precision. Statistical-protocol caveats are retained in [VERIFICATION.md](VERIFICATION.md).
- Small actual checkpoint execution and one **full S1 locked500 inference rerun** completed with the archived evaluator and selected adapter on the current RTX 4090 D server. This is **not** from-scratch training.
- Source syntax/integrity/hygiene checks and separate dispatch/metric unit tests are provided. Static closure does not certify every external speech environment or historical launcher.

## Full S1 re-execution: close, not identical

See [checkpoint_rerun_s1_500.json](../results/checkpoint_rerun_s1_500.json):

| | Frozen historical S1 | New S1 inference |
|---|---:|---:|
| Token F1 | .70158607 | .70405125 |
| LAAL, source tokens | 1.66535143 | 1.65958975 |
| Completion | 100% | 100% |

465/500 predicted gloss sequences, 459/500 WRITE-position sequences, and 453/500 joint gloss/timing/termination outcomes match exactly. No new checkpoint was selected. The hardware differs; the audit does not conclusively attribute differences to hardware alone. This is executable-checkpoint evidence, **not** byte-identical replication. The paper's three-seed historical records stay intact.

The earlier fixed first-eight smoke check is also retained, including its mismatches. The first attempted environment lacked NLTK and did not run inference; the successful reruns used the existing historical policy environment. No dependency installation or modification of that environment was needed.

## Still not claimed

- No full from-scratch re-training of all seeds, all baseline schedules, ablations, corpus TTS, or the speech frontend was performed.
- No new human ASL evaluation, video generation or pose experiment was added.
- Raw data, audio/video, backbone weights and LoRA binaries are **not uploaded in this Git source release**. They have exact access/provenance entries, not newly invented public download URLs. A complete independent backup of all large server artifacts is not yet made.
- Upstream data/model/output restrictions remain in force. An author-approved code license and approved public artifact hosting plan remain separate decisions.
- Some original figure-QA and speech dependencies are external; historical hardcoded launchers are preserved, not misleadingly called universally portable.

One paper wording caveat is explicitly retained: native no-recovery completion is 249/250, although the five component ablations/full/guard controls complete 250/250. Do not silently turn this into 100% for every listed control.

## Publication check

The target is `OwenLmz/CasualGloss`. Publishing must preserve its existing initial commit, exclude private inputs and withheld cases, and verify the pushed commit/remote tree. The audit's publication receipt is stored outside the source tree to avoid self-referential commit hashes. A local folder is not evidence of a successful push.
