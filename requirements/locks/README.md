# Recorded environments

These three `*.recorded.txt` files were produced by read-only `pip freeze --all`
from the existing policy, TTS and SeamlessStreaming environments on 2026-10-16.
All three use Python 3.10.12. `capture_metadata.json` records the collection
provenance and the currently resolved mBERT cache snapshot.

These files describe the environment at collection time, not every historical
run or a portable installation recipe. The CLERC FP16 protocol used a separate
overlay; per-experiment inference precision is retained in the experiment matrix.
Use the precision, backend and batch settings specified for the intended run.

Do not blindly install the entire recordings as ordinary requirements. CUDA
wheel selection, the PyTorch/fairseq2 ABI pair, and the two editable upstream
commits require separate environments. Unrelated transitive packages appear
because this is an unmodified environment recording, not additional project
code or a request to execute those packages.

- Policy: PyTorch `2.11.0+cu128`, Transformers `4.57.6`, PEFT `0.18.0`.
- TTS: Qwen3-TTS upstream commit `022e286b98fbec7e1e916cb940cdf532cd9f488e`.
- Seamless: upstream commit `90e2b57ac4d82fa2bfaa25caeffe39ceb8b2ebec`,
  PyTorch `2.2.2`, fairseq2/fairseq2n `0.2.1`, SimulEval `1.1.4`.
- mBERT cache currently resolves to
  `3f076fdb1ab68d5b2880cb87a0886f315b8146f8`. The original alignment config
  did not pin this revision, so this is not represented as a historical pin.

Use `scripts/reproduction/preflight.py` in the intended environment to inspect
packages, CUDA, selected adapter hashes, data paths and all four base-model
shards before running an experiment.
