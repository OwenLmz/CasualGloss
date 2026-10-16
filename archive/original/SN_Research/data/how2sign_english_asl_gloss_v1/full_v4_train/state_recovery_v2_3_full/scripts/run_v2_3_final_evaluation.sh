#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/123090274/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train}"
ROOT="${ROOT:-${DATA_ROOT}/state_recovery_v2_3_full}"
V22_ROOT="${DATA_ROOT}/fine_tuning_v2_semantic_v2_2_full"
CLERC_ROOT="${DATA_ROOT}/clerc_epee_external_eval_v1"
PYTHON_BIN="${PYTHON_BIN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/python}"
RUN="${ROOT}/runs/qwen2_5_7b_lora_state_recovery_v2_3_full"
ADAPTER="${RUN}/adapter_final"
CONFIG="${ROOT}/configs/qwen2_5_7b_lora_state_recovery_v2_3_full.json"
FINAL_ROOT="${ROOT}/final_evaluation_v2_3"

LOCKED_OUTPUT="${FINAL_ROOT}/locked500/v2_3"
EPEE_OUTPUT="${FINAL_ROOT}/clerc_epee/v2_3"
REPORTS="${FINAL_ROOT}/reports"

export HF_HOME="${HF_HOME:-/123090274/SN_Research/huggingface_cache}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${FINAL_ROOT}/logs" "${REPORTS}"

test -s "${ROOT}/reports/confirmation_comparison.json"
test -s "${ADAPTER}/adapter_model.safetensors"
test -s "${CONFIG}"

echo "[v2.3-final] step 1/3: one-shot locked-500 and Epee evaluations"
if [[ ! -s "${LOCKED_OUTPUT}/rollout_metrics.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${V22_ROOT}/scripts/evaluate_free_running_rollout.py" \
    --config "${CONFIG}" \
    --parent-splits "${DATA_ROOT}/causal_alignment_v1_corrected_full/manifests/parent_splits_full.jsonl" \
    --causal-pairs "${DATA_ROOT}/causal_alignment_v1_corrected_full/outputs/causal_aligned_pairs_full.jsonl" \
    --adapter-dir "${ADAPTER}" \
    --output-dir "${LOCKED_OUTPUT}" \
    --stage v2_3_final_locked500 \
    --split test \
    --device cuda:0 \
    --generation-batch-size 32 \
    > "${FINAL_ROOT}/logs/locked500.log" 2>&1 &
  LOCKED_PID=$!
else
  LOCKED_PID=""
fi

if [[ ! -s "${EPEE_OUTPUT}/rollout_metrics.json" ]]; then
  CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" "${CLERC_ROOT}/scripts/evaluate_epee_free_running.py" \
    --config "${CONFIG}" \
    --manifest "${CLERC_ROOT}/manifests/epee_sources_401.jsonl" \
    --adapter-dir "${ADAPTER}" \
    --output-dir "${EPEE_OUTPUT}" \
    --stage v2_3_final_clerc_epee \
    --device cuda:0 \
    --generation-batch-size 32 \
    > "${FINAL_ROOT}/logs/clerc_epee.log" 2>&1 &
  EPEE_PID=$!
else
  EPEE_PID=""
fi

[[ -z "${LOCKED_PID}" ]] || wait "${LOCKED_PID}"
[[ -z "${EPEE_PID}" ]] || wait "${EPEE_PID}"
test -s "${LOCKED_OUTPUT}/sentence_rollouts.jsonl"
test -s "${EPEE_OUTPUT}/sentence_rollouts.jsonl"

echo "[v2.3-final] step 2/3: locked-500 paired 10,000-bootstrap comparison"
"${PYTHON_BIN}" "${ROOT}/scripts/compare_locked_v2_3.py" \
  --version "v1=${DATA_ROOT}/evaluation_v1_vs_semantic_v2_1_500/outputs/v1_500/sentence_rollouts.jsonl" \
  --version "v2_1=${DATA_ROOT}/evaluation_v1_vs_semantic_v2_1_500/outputs/v2_1_500/sentence_rollouts.jsonl" \
  --version "v2_2=${DATA_ROOT}/fine_tuning_v2_semantic_v2_2_full/runs/qwen2_5_7b_lora_semantic_causal_v2_2_1_treebank_full/rollout_locked_500/sentence_rollouts.jsonl" \
  --version "v2_3=${LOCKED_OUTPUT}/sentence_rollouts.jsonl" \
  --output-dir "${REPORTS}" \
  --bootstrap-repetitions 10000 \
  > "${FINAL_ROOT}/logs/compare_locked500.log" 2>&1

echo "[v2.3-final] step 3/3: Epee paired 10,000-bootstrap comparison"
"${PYTHON_BIN}" "${ROOT}/scripts/compare_epee_v2_3.py" \
  --manifest "${CLERC_ROOT}/manifests/epee_sources_401.jsonl" \
  --version "v1=${CLERC_ROOT}/outputs/v1/sentence_rollouts.jsonl" \
  --version "v2_2=${CLERC_ROOT}/outputs/v2_2/sentence_rollouts.jsonl" \
  --version "v2_3=${EPEE_OUTPUT}/sentence_rollouts.jsonl" \
  --output "${REPORTS}/clerc_epee_v2_3_comparison.json" \
  --bootstrap-repetitions 10000 \
  > "${FINAL_ROOT}/logs/compare_clerc_epee.log" 2>&1

test -s "${REPORTS}/locked500_v2_3_comparison.json"
test -s "${REPORTS}/clerc_epee_v2_3_comparison.json"
echo "[v2.3-final] completed"
