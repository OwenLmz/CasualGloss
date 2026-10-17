# Environments

`analysis.txt` provides CPU metric and result-processing dependencies. `core-recorded.txt` pins the direct text-policy packages recorded from the server environment; install a hardware-compatible PyTorch separately (recorded build: 2.11.0+cu128). These recorded versions are not a cross-platform installer or a complete historical environment specification.

`data-builders.txt` adds SoundFile and NLTK 3.9.2 to the analysis requirements for the original waveform QC/finalizer and recovery-manifest producer. These CPU data checks need no Torch, transformers or GPU. NLTK's Treebank detokenizer here does not need Punkt models; the TransLLaMA inference tokenizer is a separate dependency below.

`plots.txt` is for portable Fig.1 rendering. `metrics.txt` provides optional speech-frontend BLEU/chrF reporting; final Table 1 BLEU is calculated by `scripts/verification/verify_paper_results.py`. Metric definitions are documented in [`docs/EVALUATION.md`](../docs/EVALUATION.md).

Speech synthesis, SeamlessStreaming/fairseq2/SimulEval, and text-policy inference use separate environments. Do not merge incompatible CUDA/PyTorch requirements. `locks/` contains complete package recordings of the three server environments and exact editable-source commits. A recording can include unrelated transitive packages; it does not establish that every listed package is essential or was present in every historical run.

Run `scripts/reproduction/preflight.py` in the appropriate environment. It checks installed packages, CUDA and selected fixed model/data identities without downloading or changing them.

The retained TransLLaMA tokenizer uses English `punkt_tab` with recorded NLTK 3.9.2, not WordNet or legacy `punkt`. Install its exact official resource outside the checkout with `python scripts/reproduction/setup_nltk.py --destination /private/nltk_data`, then run the same command with `--verify-only` for an offline, read-only SHA/size check. `--archive /path/to/punkt_tab.zip` permits offline installation. Set `NLTK_DATA` to that destination or pass it as `--nltk-data` to preflight/checkpoint commands. The stdlib-only setup pins the official ZIP and all four English model files in `provenance/nltk_resources.json`, preserves the original ZIP bytes/README, extracts only English, and refuses mismatching existing files or any destination overlapping the source checkout or original server data. Public tokenizer resources are not duplicated in the private experimental dataset bundles.
