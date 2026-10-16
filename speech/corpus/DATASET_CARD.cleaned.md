# ChineseSpeech-ASLPseudoGloss-v1 (cleaned release)

## Scope

- Version: 2026-09-06-cleaned-v1
- Utterances: 19,924
- Duration: 36.54 hours
- Speakers: 4 synthetic Mandarin voices
- Splits: {'test': 500, 'train': 18924, 'validation': 500}
- Target: weak pseudo-ASL gloss, not expert gold ASL

## Construction

The source text is derived from manually realigned How2Sign English translations. English
sentences are deterministically translated into spoken Mandarin with the pinned Qwen3-4B
model, then synthesized with four pinned Qwen3-TTS voices. Audio is retained at the original
sample rate and normalized to mono 16 kHz. The frozen 500-item validation and 500-item test
sets are unchanged.

## Quality control

The raw corpus contained 19,947 records. A frozen, model-result-independent acoustic
audit flagged 23 training records (0.12%) for
duration, silence-ratio, or possible-clipping anomalies. Those records remain in the raw
release but are excluded from this cleaned manifest. The exclusion IDs and reasons are stored
in `reports/audio_exclusions.cleaned.json`.

## Intended use

This corpus supports reproducible research on Chinese-speech-to-ASL-pseudo-gloss streaming
systems. It does not measure expert ASL translation quality and must not be represented as a
gold ASL corpus.

## Limitations

- Mandarin speech is synthetic and does not represent real-speaker or noisy conditions.
- ASL targets are LLM-generated weak pseudo labels rather than expert annotations.
- Acoustic proxy checks do not establish human-perceived speech naturalness.
- Redistribution remains subject to the licenses of the underlying source and model assets.
