# Final-paper data and models

`provenance/artifact_manifest.json` contains **239** final-paper assets and necessary production inputs, including **27** selected LoRA groups. It excludes unreported comparator weights, general training histories and unrelated pilot data. It is an access/integrity index, **not** an upload of the artifacts or permission to redistribute them.

- Main CausalGloss seeds: 20260829/20260904/20260905, all selected `adapter_final`.
- Backbone: `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`.
- Fixed k=1/3 are single-seed; fixed k=5/7 have three seeds. Use the recovered finite k=3 checkpoint.
- Learned baseline: final independent 20260918 SimAlign training, three selected adapters, decode k=1…8.
- Speech lexical control is the selected seed2 model, distinct from the removed exploratory CLERC lexical comparator.
- Five trained component ablations retain their own Stage A initialization and Stage B selected weights. No-recovery/guard-only reuse the selected prerecovery model with their respective frozen prompts.

## Distinct dataset counts

| Stage | Count / split |
|---|---|
| Original How2Sign source training records | 31,165; 30,102 unique weak-target requests |
| Upstream weak-gloss LoRA | 27,079 training pairs |
| Frozen corrected causal parents | 18,947 / 500 / 500 |
| Final semantic-policy eligible training parents | 18,632 |
| Stage A Treebank actions | 534,877 |
| Stage B full balanced + recovery actions | 439,103 = 367,267 + 71,836 |
| Development / diagnostic-confirmation | 250 / 250 frozen parents; not locked500 |
| CLERC Epee v0.3 | 401 prompts, 1,200 nested human references; revision `93bc8aa0a6526e86af80db5f59af8d133292c9b5` |
| Cleaned ChineseSpeech-ASLGloss-v1 | 19,924 synthetic utterances, 18,924/500/500, 36.54 h |

Frozen targets, splits, alignments and prefix scores are reproducibility inputs. Generating new stochastic pseudo-targets does not guarantee byte-identical data. An earlier split-only parent manifest or a v2.1 preprocessing output can be necessary without requiring the corresponding old model experiment.

## Access

Official sources: [How2Sign](https://how2sign.github.io/), [CLERC Epee](https://huggingface.co/datasets/CLERC-DATA/epee). Review upstream licenses separately from the author's code license; see [THIRD_PARTY.md](../THIRD_PARTY.md).

For assets already on the authorized research server:

```bash
python scripts/artifacts.py list --role paper_main_selected_final_seed20260829
python scripts/artifacts.py list --category corpus_manifests
```

Inspect the list first, then use the bounded `fetch` mode with your own SSH connection and a private destination outside the repository. The utility never displays key contents or overwrites a mismatching existing file. Raw verification inputs have a separate explicit 83-file allowlist in `scripts/verification/fetch_verification_inputs.py`.

Paths and available hashes are recorded per asset; null SHA values explicitly mean not content-hashed in the inventory. Audio directories are counted, not per-waveform verified. Corpus test audio reuses the locked500 assets, so copying one corpus directory is not necessarily a standalone waveform backup. No public artifact download URL or completed independent full backup is claimed.
