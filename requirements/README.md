# Environments

`analysis.txt` is for CPU result verification. `core-recorded.txt` pins the direct text-policy packages observed on the authorized reproduction server; install a hardware-compatible PyTorch separately (recorded build: 2.11.0+cu128). It is not a proven cross-platform dependency solution.

`data-builders.txt` adds SoundFile and NLTK 3.9.2 to the analysis requirements for the original waveform QC/finalizer and recovery-manifest producer. These CPU data checks need no Torch, transformers or GPU. NLTK's Treebank detokenizer here does not need Punkt models; the TransLLaMA inference tokenizer is a separate dependency below.

On 2026-10-17 a new isolated Python 3.10 environment (no system/user site
packages, no Torch) with NumPy 2.2.6, SoundFile 0.13.1 and NLTK 3.9.2 passed
146 root tests, 19 metric tests, full private-asset installation/verification,
waveform/manifest rebuilds and frozen-prediction scoring. This is a tested CPU
route, not a fresh GPU/ML environment installation. See
[`clean_clone_acceptance_20261017.json`](../results/clean_clone_acceptance_20261017.json).

`plots.txt` is for portable Fig.1 rendering. `metrics.txt` provides optional speech-frontend BLEU/chrF reporting; final Table 1 BLEU is calculated by the independent verifier, not the removed obsolete baseline script.

Speech synthesis, SeamlessStreaming/fairseq2/SimulEval, and text-policy inference use separate environments. Do not merge incompatible CUDA/PyTorch requirements. `locks/` contains complete read-only recordings of the current three server environments and exact editable-source commits. Those recordings are evidence, not a claim that every listed transitive package is essential or that a clean installation was tested.

Run `scripts/reproduction/preflight.py` in the appropriate environment. It checks installed packages, CUDA and selected fixed model/data identities without downloading or changing them. The final verification report must state whether an existing environment was reused or a new installation was actually tested.

The retained TransLLaMA tokenizer uses English `punkt_tab` with recorded NLTK 3.9.2, not WordNet or legacy `punkt`. Install its exact official resource outside the checkout with `python scripts/reproduction/setup_nltk.py --destination /private/nltk_data`, then run the same command with `--verify-only` for an offline, read-only SHA/size check. `--archive /path/to/punkt_tab.zip` permits offline installation. Set `NLTK_DATA` to that destination or pass it as `--nltk-data` to preflight/checkpoint commands. The stdlib-only setup pins the official ZIP and all four English model files in `provenance/nltk_resources.json`, preserves the original ZIP bytes/README, extracts only English, and refuses mismatching existing files or any destination overlapping the source checkout or original server data. Public tokenizer resources are not duplicated in the private experimental dataset bundles.
