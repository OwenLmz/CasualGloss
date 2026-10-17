# Figure 1

Portable content-equivalent redraw of the three submitted panels, not a
pixel-identical replacement. The submitted manuscript is not changed.

- **a:** locked-500 quality–latency frontier. Fixed k=1,3 have one seed;
  fixed k=5,7 and learned k=1…8 have three seeds (mean ± sample SD).
  CausalGloss shows its three final seeds and mean ± sample SD.
- **b,c:** CausalGloss minus learned k=1…8, with the original matched-seed,
  paired-parent hierarchical 95% confidence intervals (20,000 resamples).

```bash
python figures/build_transllama_figure.py \
  --input-report results/paper_results.json \
  --output-dir /your/output/figure1 \
  --qa-output /your/output/figure1-qa
```

Requires Python, matplotlib, numpy and pandas. Exports editable PDF/SVG plus
600-dpi PNG/TIFF. Displayed text requests at least 8 pt.

The input report contains the paper's numerical aggregates. The metric script
`scripts/verification/verify_paper_results.py` can generate the same report schema
from prediction files; see [`docs/EVALUATION.md`](../docs/EVALUATION.md).
`figure1_seed_summary.json` contains only numeric seed means and source
hashes, not sentences or predictions.

For the optional external alignment gate, add
`--qa-tools /path/to/nature-figure/scripts`. It calls
`require_matplotlib_panel_alignment` before export (1.5-pt tolerance), preserving
alignment JSON. No skill/helper code is bundled or needed for ordinary plotting.
Without this option the script explicitly reports that alignment QA was not run.
