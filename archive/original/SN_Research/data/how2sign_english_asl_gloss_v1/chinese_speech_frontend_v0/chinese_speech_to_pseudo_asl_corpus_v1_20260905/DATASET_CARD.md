# ChineseSpeech-ASLPseudoGloss-v1

## Scope

- Version: 2026-09-05
- Utterances: 19,947
- Duration: 36.89 hours
- Speakers: 4 synthetic Mandarin voices
- Splits: {'test': 500, 'train': 18947, 'validation': 500}
- Target: weak pseudo-ASL gloss, not expert gold ASL

## Construction

The source text is derived from manually realigned How2Sign English translations. English
sentences are deterministically translated into spoken Mandarin with the pinned Qwen3-4B
model, then synthesized with four pinned Qwen3-TTS voices. Audio is retained at the original
sample rate and normalized to mono 16 kHz. The frozen 500-item test set is reused unchanged.

## Intended use

This corpus supports reproducible research on Chinese-speech-to-ASL-pseudo-gloss streaming
systems. It does not measure expert ASL translation quality and should not be represented as
a gold ASL corpus.

## Limitations

- Mandarin speech is synthetic and does not represent real-speaker or noisy conditions.
- ASL targets are LLM-generated weak pseudo labels.
- Redistribution remains subject to the licenses of the underlying source and model assets.
