# CausalGloss — paper code and reproducibility archive

**CausalGloss: Multi-Evidence Action Control for a Simultaneous Mandarin Speech-to-ASL Gloss Cascade**
Mingze Li and Satoshi Nakamura · ICASSP 2027 submission

The GitHub repository name is **CasualGloss**; the method and paper name are **CausalGloss**. This archive concerns the submitted gloss-generation research only. It does **not** include the later G2P-DDM / gloss-to-pose / video extension, and does not claim conference acceptance.

## Start here

- Learning the implementation: [中文学习路线](docs/LEARNING_GUIDE_ZH.md)
- Finding the code behind each paper claim: [paper-to-code map](docs/PAPER_TO_CODE.md)
- Running code without changing the archived algorithms: [reproduction guide](docs/REPRODUCTION.md)
- Finding datasets, frozen splits, model adapters, and access conditions: [data and model guide](docs/DATA_AND_MODELS.md)
- What was actually checked, and what remains unverified: [verification report](docs/VERIFICATION.md), [release status](docs/RELEASE_STATUS.md)

## Repository organization

```text
archive/original/SN_Research/    Original server scripts/configs/protocols; original hierarchy
archive/original/workspace/      Original local analysis/plot/corpus-building code
scripts/                        New navigation, dispatch and verification utilities
provenance/                     Byte-level source identities and external-artifact inventories
results/                        Recomputed aggregate results; no source transcripts or media
docs/                           Scientific, learning and reproduction documentation
tests/                          Checks on the release tooling and numeric verification
requirements/                   Separate analysis/core/speech environment guidance
```

The historical directory names are preserved deliberately: a recovery launcher imports the preceding policy trainer, and later data builders reuse earlier semantic-scoring helpers. Flattening these directories would lose dependencies and experiment identity. New wrappers are separate from the byte-identical originals.

## Lightweight checks

Python 3.10+ is recommended. Source inspection itself requires only the standard library:

```bash
python scripts/paper.py list
python scripts/check_release.py
python -m unittest discover -s tests -p "test_*.py"
```

Do not run historical shell launchers blindly: they contain original absolute paths, device selections and expensive training commands. Use [the reproduction guide](docs/REPRODUCTION.md) and explicit input/output paths instead. Training or neural inference additionally needs the pinned model, selected adapters, lawful data access, and a compatible GPU environment. This is not a claim that cloning the source alone reproduces the paper.

## Scientific boundaries

The How2Sign-derived targets are **weak, model-generated pseudo-glosses**, not expert ASL gold labels. CLERC Epee provides a separate human-gloss evaluation. Automatic overlap/latency improvements do not establish expert ASL comprehension or non-manual correctness. The synthetic-speech cascade still over-generates; this negative result is preserved. See [results and limitations](docs/RESULTS_AND_LIMITATIONS.md).

The authoritative submitted PDF is identified by SHA-256 in [the paper anchor](provenance/paper_anchor.json). The PDF itself is not redistributed here. No submitted manuscript was edited during this release preparation.

## Access and licensing

Code, underlying datasets, generated assets, and model weights are distinct assets with distinct rights. See [THIRD_PARTY.md](THIRD_PARTY.md). Large manifests, audio/video, model backbones and LoRA binaries are not embedded in Git. Their locations and identities are recorded where verified; a manifest entry is not a public download URL. No new blanket open-source license has been chosen on the authors' behalf.
