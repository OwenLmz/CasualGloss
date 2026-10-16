# Data, models and access

Source code is stored in Git; large research artifacts are managed separately. The public source folder is not a complete offline copy of the research server. No model or data download URL has been invented, and no rights to redistribute source data are assumed.

## Asset register

`provenance/artifact_manifest.json` records **449 entries**: 447 files and two audio-directory inventories. The 28 selected LoRA groups include upstream initialization, three main checkpoints, fixed baselines, the final learned baseline, lexical controls and both stages of the five trained ablations. All selected LoRA binaries were hashed in place. 408 files received fresh SHA-256 calculation; three entries reuse previously checked identities; 36 large files have explicit `sha256: null`. Audio-directory counts are inventories, not per-waveform verification.

For each entry consult `remote_path`, `relative_to_remote_root`, `role`, `category`, bytes, selection evidence, and `hash_verification`. `exists_checked_live: true` means it existed at audit time, not that its contents are published, licensed for every use, or backed up permanently. The server remains the authoritative storage for artifacts not downloaded.

Useful commands (metadata only):

```bash
python scripts/artifacts.py list --role paper_main_selected_final_seed20260829
python scripts/artifacts.py list --category corpus_manifests
python scripts/artifacts.py list --category prefix_scores
```

An authorized researcher can copy a bounded selection to a **private directory outside this source tree**:

```bash
python scripts/artifacts.py fetch \
  --role paper_main_selected_final_seed20260829 \
  --host USER@HOST --port PORT --ssh-key /absolute/path/to/your/key \
  --destination /absolute/path/to/private_assets \
  --accept-data-terms
```

Use `list` first. The fetcher refuses mismatching existing files and never overwrites them; it checks every downloaded file. When the audit intentionally left a large-file hash null, it first requests a new read-only remote SHA. The default selection limit is 100 MB; deliberately raise `--max-bytes` only after checking the plan. Audio directories are deliberately not recursively downloaded by this tool.

## Dataset identities (do not conflate them)

| Dataset / stage | Count and purpose |
|---|---|
| How2Sign source training annotation | 31,165 records; source manually realigned English CSV hash `67a8f4fb6dc066f38006b89ac84c27a4b05d7fa07d5e6952f997042724d7a847` recorded in preparation evidence |
| Unique weak-target generation requests | 30,102; expanded back to source records; generated labels are not expert ASL annotations |
| Fine-tune-1 unique-request split | 27,079 / 1,516 / 1,507 |
| Frozen corrected causal parents | 18,947 / 500 / 500; source-video exclusions and held-out IDs preserved |
| Semantic v2.2 eligible parents | Train 18,632; filtered action-building counts are not locked-test evaluation counts |
| CausalGloss recovery continuation | 439,103 action examples from 18,632 train parents; development and diagnostic/confirmation each have 250 frozen parents |
| Locked500 text evaluation | All 500 frozen test parents; three main seeds; fixed k=1,3 single seed and k=5,7 three seeds; learned k=1…8 each three seeds |
| CLERC Epee v0.3 | 401 source prompts / 1,200 human-gloss reference clips; revision `93bc8aa0a6526e86af80db5f59af8d133292c9b5` |
| ChineseSpeech-ASLGloss-v1 **cleaned** | 19,924 synthetic utterances: 18,924 / 500 / 500; 36.537778 h reported; 23 training cases removed from original 19,947 |

Official access routes: [How2Sign](https://how2sign.github.io/) and [CLERC Epee](https://huggingface.co/datasets/CLERC-DATA/epee). Follow their licenses and keep attribution. The archived original dataset-building scripts and source IDs define the transformations. Regenerating stochastic pseudo-targets from the same base model does not promise byte-identical labels: preserve the frozen generated targets when lawful access is available.

## Selected model identity

Backbone: `Qwen/Qwen2.5-7B-Instruct` at `a09a35458c702b33eeacc393d103063234e8bc28`. It is downloaded upstream, not part of the LoRA artifact.

| CausalGloss seed | Selected adapter | SHA-256 of adapter_model.safetensors |
|---|---|---|
| 20260829 | `adapter_final` | `2037ff8e5f984530bc36eb36bda6755356481f62334736547eff830f88f7718c` |
| 20260904 | `adapter_final` | `f2e2dba69ab39d7a295d938940917bd62ba67031d1af7992ec993ef67093da07` |
| 20260905 | `adapter_final` | `65e020c048aca6a058000a1121e74f5ec8915464c0346f8b6ff8ba5bfedd179c` |

Each is 20,200,056 bytes. Keep each adapter's own config; the evaluator loads its tokenizer from the **pinned backbone**, not from an assumed independent tokenizer bundled with every adapter. See the artifact register for baseline/initialization identities and selection reports. Do not guess selection from `best` in a directory name.

## Audio layout

The paper's 500-test utterances reuse the locked500 speech assets; the full-corpus audio directory alone therefore is not a standalone copy of all test waveforms. The two inventories include format/duplicate variants (e.g. generated and resampled WAVs), so file count is not utterance count. Follow `audio_layout_notes`, waveform paths in the frozen TTS manifests, cleaning exclusions, and sample-rate validation scripts. A copied JSON manifest does not imply all referenced audio was copied.

## What remains before a public data/model release

Choose an approved artifact host and redistribution scope; review underlying data/model/output terms; preserve attribution and required notices; create a checksum-verified snapshot including all frozen manifests and referenced waveform variants; then add real download URLs. Neither a GitHub source repo nor an inventory protects against a server being deleted. An independent full artifact backup has **not** been made by this preparation.
