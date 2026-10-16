# Environments

`analysis.txt` is for CPU result verification. `core-recorded.txt` pins the direct text-policy packages observed on the authorized reproduction server; install a hardware-compatible PyTorch separately (recorded build: 2.11.0+cu128). It is not a proven cross-platform dependency solution.

`plots.txt` is for portable Fig.1 rendering. `metrics.txt` provides optional speech-frontend BLEU/chrF reporting; final Table 1 BLEU is calculated by the independent verifier, not the removed obsolete baseline script.

Speech synthesis, SeamlessStreaming/fairseq2/SimulEval, and text-policy inference use separate environments. Do not merge incompatible CUDA/PyTorch requirements. `locks/` contains complete read-only recordings of the current three server environments and exact editable-source commits. Those recordings are evidence, not a claim that every listed transitive package is essential or that a clean installation was tested.

Run `scripts/reproduction/preflight.py` in the appropriate environment. It checks installed packages, CUDA and selected fixed model/data identities without downloading or changing them. The final verification report must state whether an existing environment was reused or a new installation was actually tested.
