#!/usr/bin/env bash
set -euo pipefail

SERVER_ROOT="${SERVER_ROOT:-/123090274/SN_Research}"
TASK_ROOT="${TASK_ROOT:-${SERVER_ROOT}/data/how2sign_english_asl_gloss_v1/chinese_speech_frontend_v0}"
PYTHON_ENV="${PYTHON_ENV:-${SERVER_ROOT}/envs/task5-seamlessstreaming-cu121}"
INPUT_TSV="${INPUT_TSV:-${TASK_ROOT}/manifests/task5_pilot_100.seamlessstreaming.tsv}"
OUTPUT_DIR="${OUTPUT_DIR:-${TASK_ROOT}/outputs/seamlessstreaming_eval}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-1}"
GPU_ID="${GPU_ID:-0}"

mkdir -p "${OUTPUT_DIR}"

export FAIRSEQ2_CACHE_DIR="${TASK_ROOT}/models/fairseq2_cache"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

"${PYTHON_ENV}/bin/streaming_evaluate" \
  --task s2tt \
  --data-file "${INPUT_TSV}" \
  --audio-root-dir "${TASK_ROOT}" \
  --output "${OUTPUT_DIR}" \
  --tgt-lang eng \
  --ref-field tgt_text \
  --start-index "${START_INDEX}" \
  --end-index "${END_INDEX}" \
  --source-segment-size 320 \
  --no-strip-silence \
  --dtype fp16 \
  --no-progress-bar
