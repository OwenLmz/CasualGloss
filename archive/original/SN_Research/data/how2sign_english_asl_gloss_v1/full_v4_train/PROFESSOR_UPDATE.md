# Progress update: full How2Sign pseudo-gloss construction

- I scaled the frozen v4 English-to-pseudo-ASL-gloss pipeline to the complete How2Sign manually realigned train transcription.
- All **31,165** valid official sentence records from **2,192** videos were retained.
- Exact normalized-English deduplication produced **30,102** unique generation requests and avoided **1,063** redundant model calls.
- Generation used fixed `Qwen2.5-7B-Instruct` revision `a09a354...`, the frozen v4 prompt, greedy decoding, batch size 4, and four RTX 5090 GPUs.
- Final validation found **0** empty glosses, **0** invalid formats, and **0** record-mapping errors after transparent repair.
- **95** invalid primary outputs were retained for audit and replaced with an explicitly marked deterministic fallback.
- The final engineering pipeline status is **True**.
- The final dataset contains pseudo labels, not gold ASL annotations. Automatic QC and the 200-row review queue support triage, but qualified ASL evaluation remains necessary.
