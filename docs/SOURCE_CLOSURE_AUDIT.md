# Original-source preservation and dependency closure audit

Audit date: 2026-10-16. This document describes preservation and static inspection of the submitted CausalGloss implementation. It does not claim that historical launchers are portable or that a training experiment was rerun.

## Archive and provenance

The public archive contains **690 original files / 3,103,843 bytes**. The complete preservation set is **692 originals**: the public 690 plus two case documents retained privately outside this repository.

- `archive/original/SN_Research/`: 657 remote original scripts, configs, prompts, protocols, READMEs, adapter configs, selection records, and run metadata (2,711,436 bytes).
- `archive/original/workspace/`: 33 local manuscript statistics, figure, analysis, and first-party helper originals (392,407 bytes).

The relative hierarchy below each original root is preserved. No archived original was edited. Different historical versions retain their own paths; files sharing a basename were not combined or silently replaced.

`provenance/source_manifest.json` is the combined public per-file manifest: original source path, archived path, SHA-256, byte count, category, and origin. The remote inventory additionally records filesystem mtime in UTC. Its reviewed input manifests are `provenance/source_allowlist.review.json` and `provenance/local_analysis_source_manifest.json`; the historical review inventory is filtered through `provenance/withheld_files.json` before any new public collection.

Two Markdown case-study files contain actual dataset sentences and gloss outputs. They were moved byte-for-byte to the private audit directory under `private_withheld_sources/<original archive path>`; they were not deleted or rewritten. The complete 692-entry manifest was saved privately as `private_full_source_manifest.json` before filtering the public manifest. Public `provenance/withheld_files.json` contains only their paths, SHA-256, bytes and withholding reason (13,509 bytes in total), with no sample text. Source collection and manifest assembly enforce their exclusion.

Remote collection used a reviewed explicit file allowlist, read-only standard-library Python over SSH, and base64 transport in bounded batches. The sender hashed the exact original bytes; the receiver decoded and independently checked size/SHA-256 before writing. Existing differing archive files would cause collection to stop. No remote files were written and no training or inference was started.

The archive excludes raw JSONL data, audio/video, model weights, environments/caches, vendor/third-party implementation trees, credentials, and the G2P-DDM/gloss-to-pose extension. Adapter configuration/selection/model-card text does not imply that the corresponding weights are included. Historical failed/screen runs can appear as contextual small metadata and are never evidence of selected paper checkpoints; the selected data/model list is separate.

## Core entrypoints and first-party closure

Below, `FULL` means `archive/original/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train`; `FRONT` is the sibling `chinese_speech_frontend_v0`.

| Paper workflow | Preserved original code and dependency |
|---|---|
| Pseudo-gloss construction | FULL/scripts: prepare, generate, repair, merge, expand, validate, summarize; generation config and original prompt text |
| Fine-tune-1 | FULL/fine_tuning_v1/scripts, including finetune_common.py, trainer, evaluator, splitter, launchers |
| Lexical timing | FULL/causal_alignment_v1_corrected_full and corrected_5k scripts/configs |
| Prefix semantic teacher | FULL/semantic_causal_alignment_v2_pilot500/scripts/score_prefix_semantics.py; reused v2.1 pilot finalize/quality helpers; v2.1 full preparation/reconciliation |
| Confidence, rescue, negation, occurrence, burst rules | FULL/semantic_causal_alignment_v2_2_full scripts/configs, including Treebank prompt repair and leakage/transition audits |
| Prior policy and shared model/data helpers | FULL/fine_tuning_v2_semantic_v2_2_full/scripts: causal_action_common.py, train_qwen_lora_causal_ddp.py, evaluate_free_running_rollout.py, action evaluator, wait balancing and recovery builders |
| Main recovery stage and three seeds | FULL/state_recovery_v2_3_full and both multiseed directories: manifest builder, screen/development selection, frozen confirmation comparison, exact launchers/configs |
| Fixed Wait-k | FULL/wait_k_baselines_v1/scripts/configs, k3 recovery launcher/config; additional k5/k7 seed configs; original SN_Research root seed/queue launchers |
| Five component ablations | FULL/professor_feedback_experiments_20260912/scripts and each ablation's stage_a/stage_b config; independent version of trainer/common/evaluator retained |
| Learned adaptive baseline | FULL/transllama_learned_adaptive_20260918/scripts/tests/configs: alignment, prefix sampling/masking, training, native-context decoding, learning-rate selection, final seed evaluation/aggregation/bootstrap |
| CLERC human-gloss evaluation | FULL/clerc_epee_external_eval_v1 preparation; FULL/clerc_epee_adaptive_comparison_20260920 scripts, epee_common.py, fixed extension protocol, completed-only script |
| Speech/corpus | FRONT/scripts/configs/patches and independent corpus builder scripts/configs/cards; translation, TTS, validation, frontend conversion, causal cascade and E2E report/figure builders |
| Controls and statistical helpers | FULL/local_agreement_causal_v2_3; causal_alignment_latency_audit_v2_2; original paper comparison/build scripts; local paper_revision/analysis and manuscript figure originals |

Important cross-directory first-party imports are preserved:

- CLERC action/fixed evaluators import `causal_action_common` and `evaluate_free_running_rollout` from the v2.2 training directory.
- CLERC learned evaluator imports `evaluate_transllama` and `transllama_training` from the final 20260918 baseline directory.
- Fixed comparator imports `compare_epee_adaptive`, which imports its sibling `epee_common`. All three are present.
- Several comparisons import `audit_latency.rollout_rows`; its original implementation is present in both its remote hierarchy and the local historical helper hierarchy.
- Full v2.1 scoring invokes the earlier pilot `score_prefix_semantics.py` and `finalize_semantic_alignment_v2_1.py`. These are preserved despite their older folder names.
- The v2.1 orchestration ends by invoking its own earlier fine-tuning launcher. That earlier trainer directory was added after the dependency scan revealed the edge.

The selected main checkpoints are all `adapter_final` for seeds 20260829/20260904/20260905. The 20260829 `adapter_best` is the unchanged v2.2.1 initialization at step 0. Final learned-baseline evidence comes from the 20260918 workflow, not the older professor-feedback learned-baseline branch. Fixed k3 uses the finite recovered run. These identities are recorded in the source configs, exact launchers and selection metadata, not inferred from directory labels.

## Static checks performed

`tools/audit_archived_sources.py` reads and parses source without importing or executing it; output is `provenance/source_closure_static.json`.

- Every one of the 690 public originals matched its manifest size and SHA-256; zero integrity failures. The two moved private originals were separately checked against their pre-move hashes and retain their exact original bytes.
- All 225 Python originals parsed successfully with Python's AST parser; zero syntax failures.
- Import inventory: 1,600 standard-library occurrences; 288 external-package occurrences; 65 same-directory first-party occurrences; 23 first-party occurrences whose modules are present in another archived directory.
- Six import occurrences need external components: four `audit_panel_alignment` and two `simulstreaming_whisper`.
- A credential-pattern scan of the archived text returned zero locations. Known token/key patterns, private-key headers and URL-embedded username/password forms were checked. Credential files were excluded by path. This is a bounded pattern check, not an assurance about every possible secret encoding.

Cross-directory availability is static evidence that the code is included; it is not proof that a historical absolute import path resolves after relocation. No original source was modified to hide this distinction.

The filename scanner's remaining unmatched references are reviewed rather than treated as absent first-party modules:

| Reference | Explanation |
|---|---|
| run_waitk_seed.sh; queue_waitk5_after_k7.sh | Short names in usage messages; actual dated launchers are archived at SN_Research root |
| un_finetune2_semantic_v2_2_1_full.sh | Regex fragment from the pgrep pattern '[r]un_finetune2…'; actual launcher is included |
| tokenizer.py | Third-party SimulStreaming patch target; patch is preserved, vendor tree is excluded |

## Remaining external and portability gaps

1. Four manuscript plot originals import `audit_panel_alignment.require_matplotlib_panel_alignment` from a personal Nature-skill installation:

   - `workspace/icassp2027_paper/scripts/build_core_results.py`: import line 29; use line 418.
   - `workspace/paper_revision_20260916/scripts/analyze_fixed_waitk_multiseed.py`: import line 31; use line 611.
   - `workspace/paper_revision_20260914/scripts/analyze_fixed_waitk.py`: import line 31.
   - `workspace/paper_revision_20260919_transllama/build_transllama_figure.py`: import line 36; use line 440.

   This is external figure-layout QA code. It was not copied from the skill library and no unverified project helper was substituted. Original numerical/plotting implementations remain archived. Portable figure reproduction must explicitly remove/configure the optional QA dependency in a separate reviewed entrypoint or provide a properly sourced alternative; the originals remain unchanged.

2. Two historical SimulStreaming wrappers import `simulstreaming_whisper` from the configured third-party checkout. That upstream source is excluded; the project-owned wrappers, pinned configuration, environment freezes, and Unicode-bounds patch are included. The submitted locked500 speech cascade additionally requires upstream SeamlessStreaming and its model, Qwen3-TTS and its model, and a compatible frontend environment. Source preservation alone does not install them.

3. Launchers retain original absolute `/123090274/SN_Research` paths, virtualenv executables, GPU assignments, and original file-location assumptions. Some local figures retain personal package directories. New portable wrappers must map roots explicitly and preserve the intended import version.

4. `workspace/analysis/compare_three_locked_rollouts_reference.py:19` has an existing `parents[2]` calculation that points outside the actual helper location. The helper is present; the historical path is not repaired inside the original.

5. `workspace/bootstrap_transllama_vs_causalgloss.py` retains absolute remote `CAUSAL_PATHS`. Its inputs must be mapped to the hash-verified corresponding three seed rollouts by the portable verification layer.

6. Small configs contain original data/audio/adapter paths. Source closure does not constitute dataset/model availability. The separate `provenance/artifact_manifest.json` records 449 entries (447 files and two audio directories), including 28 selected LoRA groups. Every selected weight has a live SHA-256; 36 large files explicitly retain `sha256: null`. It lists exact train/development/test, prefix-score, alignment, corpus, rollout, training metadata/log and audio locations. That metadata inventory does not download those objects. Release data/model instructions must state which objects are local, remote only, retrievable, or not publicly redistributed. An object listed remotely is not an uploaded GitHub artifact.

Core third-party packages observed include torch, transformers, PEFT, NumPy, NLTK, SimAlign, sacrebleu, matplotlib, pandas, soundfile/librosa and frontend-specific packages. The recorded training and frontend environment freezes are preserved; they should not be conflated into one environment. Main S1 metadata records RTX 5090/CUDA 12.8, while other seeds can record different hardware.

Original file text, embedded acknowledgements, model-card text, and any pre-existing license notices are preserved byte-for-byte. No new license was asserted for upstream code/model/data, and no vendor implementation was copied.

## Maintainer collection tools

`tools/collect_original_sources.py` supports separate inventory and retrieval phases. SSH target/identity/port are supplied through task-specific environment variables, not stored credentials. `tools/assemble_source_manifest.py` merges the remote and local manifests and verifies every original. Rerunning retrieval should be followed by assembly and the static audit.

These tools do not invoke any archived training or evaluation entrypoint. Their generated source manifest is a preservation record; scientific result recomputation is handled separately by the verification package.
