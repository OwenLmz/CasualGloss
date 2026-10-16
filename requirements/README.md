# Environments

`analysis.txt` is for CPU result verification. `core-recorded.txt` documents the selected text-policy environment, but is not a fully solved cross-platform lockfile. Exact runtime freezes and per-run metadata are retained with the original sources. Install PyTorch separately for the actual hardware. Current audit hardware is not necessarily the original training hardware.

Speech synthesis, SeamlessStreaming/fairseq2/SimulEval, and text-policy inference used separate environments. Do not merge their incompatible requirements into one environment. The archived speech environment freeze/config/patch files and pinned upstream revisions are the reference. A newly installed full speech environment has not been certified by this source release.
