# Chinese Speech Frontend v0

This directory contains the reproducible Task 5 prototype for:

```text
Chinese speech
  -> incremental committed English text
  -> Fine-tune 2 causal action model
  -> incremental pseudo ASL gloss
```

## Scientific status

- The How2Sign English transcriptions are official manually realigned annotations.
- The Chinese text is a generated research-side source, not an official How2Sign annotation.
- The Chinese audio is synthetic and is not a substitute for real-speaker evaluation.
- The ASL gloss targets are Qwen-generated weak pseudo labels, not gold ASL gloss.
- Fine-tune 2 timing is pseudo-aligned text/gloss timing, not verified sign-video timing.

## Frontend selection

The selected Task 5 v0 frontend is **Meta SeamlessStreaming 2.5B monotonic
S2TT**. It was selected only after a paired comparison on the same locked 100
synthetic Mandarin utterances. The retained Qwen3 cascade uses AlignAtt
Mandarin ASR followed by Qwen3-4B translation with a two-update Local Agreement
policy.

| Metric | Qwen3 cascade | SeamlessStreaming |
| --- | ---: | ---: |
| English BLEU | 18.5595 | **24.0454** |
| English chrF | 44.3342 | **47.4752** |
| Mean first English commit | 3.8488 s | **2.9648 s** |
| Pseudo-gloss sequence similarity | 0.3419 | **0.3878** |
| Pseudo-gloss token F1 | 0.3667 | **0.4051** |
| Mean first gloss write | 3.9664 s | **3.0352 s** |

Both end-to-end systems completed 100/100 samples with no premature `<END>`.
For SeamlessStreaming minus the Qwen3 cascade, a paired 5,000-resample
bootstrap produced a sequence-similarity delta of `+0.0459`, 95% CI
`[+0.0076, +0.0867]`, and a first-gloss-write delta of `-0.9312 s`, 95% CI
`[-1.1656, -0.7040]`. The token-F1 CI includes zero, so that individual gain
is not claimed as statistically conclusive.

## Locked versions

- SimulStreaming code: `077ea37d5ab4ff98bc567e4507f140dc4e5d5ad6`
- Qwen3-TTS code: `022e286b98fbec7e1e916cb940cdf532cd9f488e`
- Qwen3-TTS model revision: `85e237c12c027371202489a0ec509ded67b5e4b5`
- Whisper large-v3 SHA-256: `e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb`
- Seamless Communication code: `90e2b57ac4d82fa2bfaa25caeffe39ceb8b2ebec`
- SeamlessStreaming model revision: `82746da73c2927b962e53ad2f723babf1ae0e0b5`
- Qwen3-4B-Instruct-2507 revision: `cdbee75f17c01a7cc42f958dc650907174af0554`
- Fine-tune 2 base model revision: `a09a35458c702b33eeacc393d103063234e8bc28`

## Directory layout

```text
configs/       Fixed experiment settings
manifests/     Text/audio mappings and model outputs
audio/         Generated original and normalized 16 kHz WAV files
models/        Pinned local model snapshots/checkpoints
outputs/       Incremental frontend and end-to-end predictions
reports/       Validation, metrics and provenance reports
scripts/       Reproducible build, inference and evaluation code
third_party/   Pinned official repositories
logs/          Installation and inference logs
```

## Run order

1. `build_task5_pilot_manifest.py`
2. `generate_qwen3_tts.py --limit 1` and `validate_audio_manifest.py`
3. `generate_qwen3_tts.py --limit 100` and `validate_audio_manifest.py`
4. `run_simulstreaming_zh_asr_batch.py` and `run_qwen3_streaming_mt.py`
5. `build_seamlessstreaming_tsv.py` and `run_seamlessstreaming_eval.sh`
6. `convert_seamlessstreaming_results.py`
7. `evaluate_streaming_frontend.py` for each candidate
8. `run_speech_to_gloss_e2e.py` for each candidate using the same Fine-tune 2 adapter
9. `compare_task5_frontends.py`

## Primary outputs

- Selected incremental English: `outputs/task5_pilot_100.seamlessstreaming.jsonl`
- Selected end-to-end result: `outputs/task5_pilot_100.seamlessstreaming_ft2_e2e.jsonl`
- Frontend evaluation: `reports/task5_pilot_100.seamlessstreaming.evaluation.json`
- End-to-end evaluation: `reports/task5_pilot_100.seamlessstreaming_ft2_e2e.json`
- Model-selection report: `reports/task5_frontend_model_selection_100.json`

## Causal-alignment v2.2 integration

`run_speech_to_gloss_e2e.py` now uses the same transition semantics as the
locked v2.2 free-running evaluator:

- `WRITE(gloss)` commits one gloss token and keeps the current English prefix;
- `<WAIT>` consumes the next committed English token;
- after the English source is complete, inference continues at the full prefix
  until `<END>` or the predeclared action limit;
- multiple gloss tokens may therefore be emitted as a burst at one source
  prefix.

An `<END>` action is counted as a valid completion only after all committed
English source tokens have been observed. Premature `<END>` is retained as a
separate error and is not accepted as successful completion.

The runner records both source-token positions and audio-read timestamps for
every gloss write.  Its report includes speech-time reference-length AL, LAAL,
AP and DAL in addition to first-write latency.  These latency values exclude
model computation time and must be reported together with sequence similarity,
token F1, completion rate and output-length error so that under-generation is
not mistaken for a latency improvement.

The report also includes a compute-adjusted serial-online upper bound. Each
action begins only after its audio timestamp is available and the preceding
model call has finished; measured inference time is then added before the
action is considered emitted. This separates algorithmic waiting from runtime
overhead and exposes policies that appear early only because model-call cost
was ignored.
