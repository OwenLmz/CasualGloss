# Paper → experiment → code

Below, `F` denotes `archive/original/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train`; `S` denotes its sibling `chinese_speech_frontend_v0`. Exact individual file hashes are in `provenance/source_manifest.json`.

| Paper component | Original directory / entry | Required evidence |
|---|---|---|
| Pseudo-gloss construction | `F/scripts`, `fine_tuning_v1` | Source CSV identity, deduplication, weak-target outputs, split manifests, frozen upstream adapter |
| Lexical timing | `F/causal_alignment_v1_corrected_full/scripts` | Parent/source-video splits, alignment pairs; frozen held-out IDs |
| Prefix semantics | `F/semantic_causal_alignment_v2_pilot500`, `v2_1_pilot500`, `v2_1_full` | Prefix likelihood matrices and thresholds; these earlier helpers remain dependencies |
| Confidence / burst WRITE | `F/semantic_causal_alignment_v2_2_full/scripts` | Config, validated action transitions, Treebank detokenization |
| Stage-A policy | `F/fine_tuning_v2_semantic_v2_2_full/scripts` | 534,877 Treebank action examples, selected stage-A adapter; WAIT balancing is used in the later recovery-stage construction |
| Recovery / CausalGloss | `F/state_recovery_v2_3_full` and two `multiseed` directories | Three selection reports and **adapter_final**, seeds 20260829/20260904/20260905 |
| Fixed Wait-k frontier | `F/wait_k_baselines_v1`, `wait_k_baselines_multiseed_20260914` | Trained k=1,3,5,7; recovered finite k=3; k=5,7 have three seeds |
| Final learned baseline | `F/transllama_learned_adaptive_20260918` | Independent SimAlign trajectories, LR screening, three final seeds, k=1…8 rollouts |
| Table 1 / Fig. 1 | Final locked500 rollouts + archived workspace analysis code | Same frozen 500 parents; seed means/SD, BLEU and paired/hierarchical bootstrap |
| Table 2 | `F/professor_feedback_experiments_20260912`, recovery and guard controls | Confirmation/diagnostic 250, five two-stage retrained ablations plus recovery ablation/control |
| Human-gloss transfer | `F/clerc_epee_external_eval_v1`, `clerc_epee_adaptive_comparison_20260920` | Pinned Epee revision, 401 sources/1,200 references, final shared FP16 protocol |
| Completed-only F1 | Same directory, `posthoc_completed_only_epee_sensitivity_20260921.py` | Post-hoc subset based on baseline completion; not a replacement for full-401 analysis |
| Synthetic speech cascade | `S/scripts`, `S/paper_e2e_locked500_20260904` | Same 500 inputs, shared frontend commits, lexical control and three CausalGloss runs |
| 19,924-utterance corpus | `S/chinese_speech_to_pseudo_asl_corpus_v1_20260905` | **cleaned** corpus, 23 excluded training audio cases, 18,924/500/500 split |

## Version traps

- The final learned baseline is **not** the older `professor_feedback_experiments_20260912/learned_baseline`, nor the old inference-only wait-k adaptation.
- Main CausalGloss seed S1's `adapter_best` has global step 0. The selected `adapter_final` is a different file.
- Fixed k=3 uses the recovered finite run; the early nonfinite run is not a valid baseline.
- CLERC lexical control and speech lexical control use different recorded adapters. Do not substitute one for the other.
- Locked500 is not the same as filtered teacher-forced action rows, nor the 250-parent diagnostic set.
- The raw Chinese corpus has 19,947 entries; the paper's cleaned corpus has 19,924. Neither count is the v2.3 training-parent count of 18,632.

The three original model seeds, different fixed-baseline seeds, hardware, data hashes, and checkpoint identities must travel together. A convenient directory name is not evidence of selection.
