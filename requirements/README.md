# Environments

`analysis.txt` is for CPU result verification. `core-recorded.txt` documents the selected text-policy environment, but is not a fully solved cross-platform lockfile. This minimal source tree does not bundle all historical environment freezes or run metadata. Install PyTorch separately for the actual hardware.

`plots.txt` is for portable Fig.1 rendering. `metrics.txt` is optional for the original SacreBLEU post-processing script; it is not needed by the independent numerical verifier.

Speech synthesis, SeamlessStreaming/fairseq2/SimulEval, and text-policy inference used separate environments. Do not merge incompatible requirements into one environment. Follow the final model identities and upstream speech runtimes; a newly installed complete speech environment has not been certified by this source release.
