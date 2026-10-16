#!/usr/bin/env bash
set -euo pipefail

SERVER_ROOT="${SERVER_ROOT:-/123090274/SN_Research}"
TASK_ROOT="${TASK_ROOT:-${SERVER_ROOT}/data/how2sign_english_asl_gloss_v1/chinese_speech_frontend_v0}"
FULL_TRAIN="${SERVER_ROOT}/data/how2sign_english_asl_gloss_v1/full_v4_train"
RUN_ROOT="${RUN_ROOT:-${TASK_ROOT}/chinese_speech_to_pseudo_asl_corpus_v1_20260905}"
SCRIPT_ROOT="${RUN_ROOT}/scripts"
CONFIG="${RUN_ROOT}/config_full_corpus.json"
ML_PY="${ML_PY:-${SERVER_ROOT}/venvs/wlasl-cu128-env/bin/python}"
TTS_PY="${TTS_PY:-${SERVER_ROOT}/envs/task5-qwen3-tts-cu128/bin/python}"
TRANSLATION_MODEL="${TASK_ROOT}/models/Qwen3-4B-Instruct-2507"
TTS_MODEL="${TASK_ROOT}/models/Qwen3-TTS-12Hz-0.6B-CustomVoice"
PARENTS="${FULL_TRAIN}/causal_alignment_v1_corrected_full/manifests/parent_splits_full.jsonl"
ALL_RECORDS="${FULL_TRAIN}/manifests/how2sign_train_all_records_31165.jsonl"
LOCKED_TEST="${TASK_ROOT}/paper_e2e_locked500_20260904/manifests/locked500.tts.jsonl"
EXPECTED=19947
SHARDS=4

mkdir -p "${RUN_ROOT}"/{manifests,shards,reports,logs,scripts}

if [[ ! -s "${RUN_ROOT}/manifests/full.source.jsonl" ]]; then
  "${ML_PY}" "${SCRIPT_ROOT}/prepare_full_corpus_manifest.py" \
    --parents "${PARENTS}" \
    --all-records "${ALL_RECORDS}" \
    --locked-test-manifest "${LOCKED_TEST}" \
    --config "${CONFIG}" \
    --output "${RUN_ROOT}/manifests/full.source.jsonl" \
    --report "${RUN_ROOT}/reports/manifest_build.json" \
    --audio-subdir "chinese_speech_to_pseudo_asl_corpus_v1_20260905"
fi

"${ML_PY}" "${SCRIPT_ROOT}/shard_jsonl.py" \
  --input "${RUN_ROOT}/manifests/full.source.jsonl" \
  --output-dir "${RUN_ROOT}/shards" \
  --prefix source \
  --shards "${SHARDS}"

declare -a pids=()
declare -a labels=()
for shard in 0 1 2 3; do
  report="${RUN_ROOT}/reports/translation.shard${shard}.json"
  if [[ -s "${report}" ]] && jq -e '.ok == true' "${report}" >/dev/null; then
    continue
  fi
  "${ML_PY}" "${SCRIPT_ROOT}/translate_full_corpus_en_zh.py" \
    --input "${RUN_ROOT}/shards/source.shard${shard}.jsonl" \
    --output "${RUN_ROOT}/shards/translated.shard${shard}.jsonl" \
    --report "${report}" \
    --model-path "${TRANSLATION_MODEL}" \
    --model-revision cdbee75f17c01a7cc42f958dc650907174af0554 \
    --device "cuda:${shard}" \
    --batch-size 8 \
    > "${RUN_ROOT}/logs/translation.shard${shard}.log" 2>&1 &
  pids+=("$!")
  labels+=("translation-shard${shard}")
done

for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "FAILED: ${labels[$i]}" >&2
    exit 1
  fi
done

"${ML_PY}" "${TASK_ROOT}/scripts/merge_frontend_shards.py" \
  --manifest "${RUN_ROOT}/manifests/full.source.jsonl" \
  --shard "${RUN_ROOT}/shards/translated.shard0.jsonl" \
  --shard "${RUN_ROOT}/shards/translated.shard1.jsonl" \
  --shard "${RUN_ROOT}/shards/translated.shard2.jsonl" \
  --shard "${RUN_ROOT}/shards/translated.shard3.jsonl" \
  --output "${RUN_ROOT}/manifests/full.translated.jsonl" \
  --report "${RUN_ROOT}/reports/translation_merge.json" \
  --status-field source_text_zh_status \
  --completed-value completed

"${ML_PY}" "${SCRIPT_ROOT}/shard_jsonl.py" \
  --input "${RUN_ROOT}/manifests/full.translated.jsonl" \
  --output-dir "${RUN_ROOT}/shards" \
  --prefix translated \
  --shards "${SHARDS}"

pids=()
labels=()
for shard in 0 1 2 3; do
  report="${RUN_ROOT}/reports/tts.shard${shard}.json"
  if [[ -s "${report}" ]] && jq -e '.ok == true' "${report}" >/dev/null; then
    continue
  fi
  "${TTS_PY}" "${SCRIPT_ROOT}/generate_full_corpus_tts.py" \
    --config "${CONFIG}" \
    --manifest "${RUN_ROOT}/shards/translated.shard${shard}.jsonl" \
    --root "${TASK_ROOT}" \
    --model-path "${TTS_MODEL}" \
    --output-manifest "${RUN_ROOT}/shards/tts.shard${shard}.jsonl" \
    --report "${report}" \
    --device "cuda:${shard}" \
    > "${RUN_ROOT}/logs/tts.shard${shard}.log" 2>&1 &
  pids+=("$!")
  labels+=("tts-shard${shard}")
done

for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "FAILED: ${labels[$i]}" >&2
    exit 1
  fi
done

"${ML_PY}" "${TASK_ROOT}/scripts/merge_frontend_shards.py" \
  --manifest "${RUN_ROOT}/manifests/full.translated.jsonl" \
  --shard "${RUN_ROOT}/shards/tts.shard0.jsonl" \
  --shard "${RUN_ROOT}/shards/tts.shard1.jsonl" \
  --shard "${RUN_ROOT}/shards/tts.shard2.jsonl" \
  --shard "${RUN_ROOT}/shards/tts.shard3.jsonl" \
  --output "${RUN_ROOT}/manifests/full.tts.jsonl" \
  --report "${RUN_ROOT}/reports/tts_merge.json" \
  --status-field tts_status \
  --completed-value completed

"${TTS_PY}" "${TASK_ROOT}/scripts/validate_audio_manifest.py" \
  --manifest "${RUN_ROOT}/manifests/full.tts.jsonl" \
  --root "${TASK_ROOT}" \
  --report "${RUN_ROOT}/reports/audio_validation.json" \
  --expected-count "${EXPECTED}"

"${ML_PY}" "${SCRIPT_ROOT}/summarize_full_corpus.py" \
  --manifest "${RUN_ROOT}/manifests/full.tts.jsonl" \
  --root "${TASK_ROOT}" \
  --config "${CONFIG}" \
  --audio-validation "${RUN_ROOT}/reports/audio_validation.json" \
  --report "${RUN_ROOT}/reports/corpus_summary.json" \
  --dataset-card "${RUN_ROOT}/DATASET_CARD.md" \
  --expected-count "${EXPECTED}"

{
  date -u
  hostname
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  "${ML_PY}" --version
  "${TTS_PY}" --version
} > "${RUN_ROOT}/reports/runtime_environment.txt"

sha256sum \
  "${CONFIG}" \
  "${RUN_ROOT}/manifests/full.source.jsonl" \
  "${RUN_ROOT}/manifests/full.translated.jsonl" \
  "${RUN_ROOT}/manifests/full.tts.jsonl" \
  > "${RUN_ROOT}/reports/core_sha256sums.txt"

echo "FULL_CHINESE_SPEECH_CORPUS_BUILD_COMPLETE"
