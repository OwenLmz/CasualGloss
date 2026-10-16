#!/usr/bin/env bash
set -euo pipefail

SERVER_ROOT="${SERVER_ROOT:-/123090274/SN_Research}"
TASK_ROOT="${TASK_ROOT:-${SERVER_ROOT}/data/how2sign_english_asl_gloss_v1/chinese_speech_frontend_v0}"
RUN_ROOT="${RUN_ROOT:-${TASK_ROOT}/paper_e2e_locked500_20260904}"
ML_PY="${ML_PY:-${SERVER_ROOT}/venvs/wlasl-cu128-env/bin/python}"
TTS_PY="${TTS_PY:-${SERVER_ROOT}/envs/task5-qwen3-tts-cu128/bin/python}"
BASE_MODEL="${BASE_MODEL:-${SERVER_ROOT}/huggingface_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}"
V1_ADAPTER="${V1_ADAPTER:-${SERVER_ROOT}/data/how2sign_english_asl_gloss_v1/full_v4_train/fine_tuning_v2_causal_full_seed2/runs/qwen2_5_7b_lora_causal_action_full_seed2_v1/adapter_best}"
V23_ADAPTER="${V23_ADAPTER:-${SERVER_ROOT}/data/how2sign_english_asl_gloss_v1/full_v4_train/state_recovery_v2_3_full/runs/qwen2_5_7b_lora_state_recovery_v2_3_full/adapter_final}"
V1_CONFIG="${V1_CONFIG:-${SERVER_ROOT}/data/how2sign_english_asl_gloss_v1/full_v4_train/fine_tuning_v2_causal_full_seed2/configs/qwen2_5_7b_lora_causal_action_full_seed2_v1.json}"
V23_CONFIG="${V23_CONFIG:-${SERVER_ROOT}/data/how2sign_english_asl_gloss_v1/full_v4_train/state_recovery_v2_3_full/configs/qwen2_5_7b_lora_state_recovery_v2_3_full.json}"
SCRIPTS="${TASK_ROOT}/scripts"

mkdir -p "${RUN_ROOT}"/{outputs,reports,logs,artifacts/tables,artifacts/figures}

"${ML_PY}" "${SCRIPTS}/merge_frontend_shards.py" \
  --manifest "${RUN_ROOT}/manifests/locked500.translated.jsonl" \
  --shard "${RUN_ROOT}/shards/locked500.tts.shard0.jsonl" \
  --shard "${RUN_ROOT}/shards/locked500.tts.shard1.jsonl" \
  --output "${RUN_ROOT}/manifests/locked500.tts.jsonl" \
  --report "${RUN_ROOT}/reports/tts_merge.json" \
  --status-field tts_status \
  --completed-value completed

"${TTS_PY}" "${SCRIPTS}/validate_audio_manifest.py" \
  --manifest "${RUN_ROOT}/manifests/locked500.tts.jsonl" \
  --root "${TASK_ROOT}" \
  --report "${RUN_ROOT}/reports/audio_validation.json" \
  --expected-count 500

for shard in 0 1; do
  "${ML_PY}" "${SCRIPTS}/build_seamlessstreaming_tsv.py" \
    --manifest "${RUN_ROOT}/shards/locked500.tts.shard${shard}.jsonl" \
    --output "${RUN_ROOT}/shards/locked500.seamless.shard${shard}.tsv"
done

pid0=""
pid1=""
if [[ ! -s "${RUN_ROOT}/outputs/seamless_shard0/instances.log" || ! -s "${RUN_ROOT}/outputs/seamless_shard0/metrics.tsv" ]]; then
  INPUT_TSV="${RUN_ROOT}/shards/locked500.seamless.shard0.tsv" \
  OUTPUT_DIR="${RUN_ROOT}/outputs/seamless_shard0" \
  START_INDEX=0 END_INDEX=250 GPU_ID=0 \
    bash "${SCRIPTS}/run_seamlessstreaming_eval.sh" \
    > "${RUN_ROOT}/logs/seamless_shard0.log" 2>&1 &
  pid0=$!
fi

if [[ ! -s "${RUN_ROOT}/outputs/seamless_shard1/instances.log" || ! -s "${RUN_ROOT}/outputs/seamless_shard1/metrics.tsv" ]]; then
  INPUT_TSV="${RUN_ROOT}/shards/locked500.seamless.shard1.tsv" \
  OUTPUT_DIR="${RUN_ROOT}/outputs/seamless_shard1" \
  START_INDEX=0 END_INDEX=250 GPU_ID=1 \
    bash "${SCRIPTS}/run_seamlessstreaming_eval.sh" \
    > "${RUN_ROOT}/logs/seamless_shard1.log" 2>&1 &
  pid1=$!
fi

status0=0
status1=0
if [[ -n "${pid0}" ]]; then wait "${pid0}" || status0=$?; fi
if [[ -n "${pid1}" ]]; then wait "${pid1}" || status1=$?; fi
if [[ "${status0}" -ne 0 || "${status1}" -ne 0 ]]; then
  echo "SeamlessStreaming shard failure: shard0=${status0}, shard1=${status1}" >&2
  exit 1
fi

for shard in 0 1; do
  "${ML_PY}" "${SCRIPTS}/convert_seamlessstreaming_results.py" \
    --input-manifest "${RUN_ROOT}/shards/locked500.tts.shard${shard}.jsonl" \
    --instances-log "${RUN_ROOT}/outputs/seamless_shard${shard}/instances.log" \
    --metrics-tsv "${RUN_ROOT}/outputs/seamless_shard${shard}/metrics.tsv" \
    --output-jsonl "${RUN_ROOT}/shards/locked500.frontend.shard${shard}.jsonl" \
    --output-report "${RUN_ROOT}/reports/frontend_conversion_shard${shard}.json" \
    --model-revision 82746da73c2927b962e53ad2f723babf1ae0e0b5
done

"${ML_PY}" "${SCRIPTS}/merge_frontend_shards.py" \
  --manifest "${RUN_ROOT}/manifests/locked500.tts.jsonl" \
  --shard "${RUN_ROOT}/shards/locked500.frontend.shard0.jsonl" \
  --shard "${RUN_ROOT}/shards/locked500.frontend.shard1.jsonl" \
  --output "${RUN_ROOT}/outputs/locked500.seamlessstreaming.jsonl" \
  --report "${RUN_ROOT}/reports/frontend_merge.json"

"${ML_PY}" "${SCRIPTS}/evaluate_streaming_frontend.py" \
  --predictions "${RUN_ROOT}/outputs/locked500.seamlessstreaming.jsonl" \
  --output-report "${RUN_ROOT}/reports/frontend_evaluation.json" \
  --output-preview "${RUN_ROOT}/reports/frontend_worst_first.jsonl" \
  --expected-count 500 \
  --status-field streaming_frontend_status \
  --hypothesis-field streaming_frontend_hypothesis_en \
  --first-commit-field streaming_frontend_first_commit_audio_seconds \
  --last-commit-field streaming_frontend_last_commit_audio_seconds \
  --rtf-field streaming_frontend_real_time_factor \
  --event-count-field streaming_frontend_commit_event_count \
  --method-label "Meta SeamlessStreaming 2.5B monotonic S2TT"

"${ML_PY}" "${SCRIPTS}/run_speech_to_gloss_e2e.py" \
  --frontend-jsonl "${RUN_ROOT}/outputs/locked500.seamlessstreaming.jsonl" \
  --config "${V1_CONFIG}" \
  --base-model-path "${BASE_MODEL}" \
  --adapter-dir "${V1_ADAPTER}" \
  --output-jsonl "${RUN_ROOT}/outputs/v1.jsonl" \
  --output-report "${RUN_ROOT}/reports/v1.json" \
  --device cuda:0 \
  --overwrite > "${RUN_ROOT}/logs/v1_e2e.log" 2>&1 &
pid0=$!

"${ML_PY}" "${SCRIPTS}/run_speech_to_gloss_e2e.py" \
  --frontend-jsonl "${RUN_ROOT}/outputs/locked500.seamlessstreaming.jsonl" \
  --config "${V23_CONFIG}" \
  --base-model-path "${BASE_MODEL}" \
  --adapter-dir "${V23_ADAPTER}" \
  --output-jsonl "${RUN_ROOT}/outputs/v2_3.jsonl" \
  --output-report "${RUN_ROOT}/reports/v2_3.json" \
  --device cuda:1 \
  --overwrite > "${RUN_ROOT}/logs/v2_3_e2e.log" 2>&1 &
pid1=$!

status0=0
status1=0
wait "${pid0}" || status0=$?
wait "${pid1}" || status1=$?
if [[ "${status0}" -ne 0 || "${status1}" -ne 0 ]]; then
  echo "Speech-to-gloss failure: v1=${status0}, v2.3=${status1}" >&2
  exit 1
fi

"${ML_PY}" "${SCRIPTS}/compare_causal_e2e_versions.py" \
  --baseline "${RUN_ROOT}/outputs/v1.jsonl" \
  --candidate "${RUN_ROOT}/outputs/v2_3.jsonl" \
  --baseline-name v1 \
  --candidate-name v2_3 \
  --bootstrap-samples 10000 \
  --seed 20260904 \
  --quality-noninferiority-margin 0.02 \
  --completion-noninferiority-margin 0.01 \
  --length-error-tolerance 0.02 \
  --scientific-status paired_locked500_synthetic_mandarin_to_pseudo_gloss_not_gold_asl \
  --output "${RUN_ROOT}/reports/v2_3_vs_v1.json"

"${ML_PY}" "${SCRIPTS}/build_locked500_e2e_artifacts.py" \
  --v1-report "${RUN_ROOT}/reports/v1.json" \
  --v2-3-report "${RUN_ROOT}/reports/v2_3.json" \
  --comparison "${RUN_ROOT}/reports/v2_3_vs_v1.json" \
  --frontend-report "${RUN_ROOT}/reports/frontend_evaluation.json" \
  --translation-report "${RUN_ROOT}/reports/translation.json" \
  --audio-report "${RUN_ROOT}/reports/audio_validation.json" \
  --output-root "${RUN_ROOT}/artifacts"

{
  date -u
  hostname
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  "${ML_PY}" --version
} > "${RUN_ROOT}/reports/runtime_environment.txt"

find "${RUN_ROOT}" -type f ! -name sha256sums.txt -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "${RUN_ROOT}/reports/sha256sums.txt"

echo "LOCKED500_E2E_PIPELINE_COMPLETE"
