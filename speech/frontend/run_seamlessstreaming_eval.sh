#!/usr/bin/env bash
set -euo pipefail

: "${TASK_ROOT:?Set TASK_ROOT to the recorded speech frontend data root}"
: "${PYTHON_ENV:?Set PYTHON_ENV to the SeamlessStreaming environment directory}"
: "${INPUT_TSV:?Set INPUT_TSV to an explicit final-paper input manifest}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new experiment output directory}"
START_INDEX="${START_INDEX:-0}"
: "${END_INDEX:?Set END_INDEX explicitly for the recorded final-paper shard}"
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
