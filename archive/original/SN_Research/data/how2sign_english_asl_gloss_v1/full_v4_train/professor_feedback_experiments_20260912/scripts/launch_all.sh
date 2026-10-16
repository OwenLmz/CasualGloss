#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/123090274/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train}"
ROOT="${ROOT:-${DATA_ROOT}/professor_feedback_experiments_20260912}"
PYTHON_BIN="${PYTHON_BIN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/python}"
V1="${DATA_ROOT}/causal_alignment_v1_corrected_full"
V22="${DATA_ROOT}/semantic_causal_alignment_v2_2_full"
V21="${DATA_ROOT}/semantic_causal_alignment_v2_1_full"
FROZEN="${DATA_ROOT}/state_recovery_v2_3_full/manifests"

mkdir -p "${ROOT}/logs" "${ROOT}/reports" "${ROOT}/pids"

nohup "${PYTHON_BIN}" "${ROOT}/scripts/prepare_experiment_manifests.py" \
  --v22-pairs "${V22}/outputs/semantic_causal_aligned_pairs_v2_2_full.jsonl" \
  --v1-pairs "${V1}/outputs/causal_aligned_pairs_full.jsonl" \
  --v1-validation-actions "${V1}/manifests/validation_actions.jsonl" \
  --development-pairs "${FROZEN}/development_pairs.jsonl" \
  --confirmation-pairs "${FROZEN}/confirmation_pairs.jsonl" \
  --output-root "${ROOT}" > "${ROOT}/logs/preparation.log" 2>&1 &
echo $! > "${ROOT}/pids/preparation.pid"

CUDA_VISIBLE_DEVICES=0 nohup bash "${ROOT}/scripts/run_learned_baseline.sh" \
  > "${ROOT}/logs/learned_baseline.log" 2>&1 &
echo $! > "${ROOT}/pids/learned_baseline.pid"

nohup bash "${ROOT}/scripts/run_queue.sh" 1 no_semantic no_rescue \
  > "${ROOT}/logs/gpu1_queue.log" 2>&1 &
echo $! > "${ROOT}/pids/gpu1_queue.pid"

nohup bash "${ROOT}/scripts/run_queue.sh" 2 no_negation no_occurrence \
  > "${ROOT}/logs/gpu2_queue.log" 2>&1 &
echo $! > "${ROOT}/pids/gpu2_queue.pid"

nohup bash "${ROOT}/scripts/run_queue.sh" 3 no_burst \
  > "${ROOT}/logs/gpu3_queue.log" 2>&1 &
echo $! > "${ROOT}/pids/gpu3_queue.pid"

printf 'launched_utc=%s\n' "$(date -u +%FT%TZ)" > "${ROOT}/reports/LAUNCHED.txt"
cat "${ROOT}"/pids/*.pid
