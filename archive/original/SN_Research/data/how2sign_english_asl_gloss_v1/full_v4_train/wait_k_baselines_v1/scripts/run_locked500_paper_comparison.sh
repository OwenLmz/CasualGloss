#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/123090274/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train}"
ROOT="${ROOT:-${DATA_ROOT}/wait_k_baselines_v1}"
PYTHON_BIN="${PYTHON_BIN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/python}"
OUTPUT_DIR="${DATA_ROOT}/causal_alignment_comparison_with_wait_k_v1"

V1="${DATA_ROOT}/evaluation_v1_vs_semantic_v2_1_500/outputs/v1_500/sentence_rollouts.jsonl"
V21="${DATA_ROOT}/evaluation_v1_vs_semantic_v2_1_500/outputs/v2_1_500/sentence_rollouts.jsonl"
V22="${DATA_ROOT}/fine_tuning_v2_semantic_v2_2_full/runs/qwen2_5_7b_lora_semantic_causal_v2_2_1_treebank_full/rollout_locked_500/sentence_rollouts.jsonl"
K1="${ROOT}/runs/qwen2_5_7b_lora_wait_k1_full_v1/rollout_locked500/sentence_rollouts.jsonl"
K3="${ROOT}/runs/qwen2_5_7b_lora_wait_k3_full_v1/rollout_locked500/sentence_rollouts.jsonl"
K5="${ROOT}/runs/qwen2_5_7b_lora_wait_k5_full_v1/rollout_locked500/sentence_rollouts.jsonl"
K7="${ROOT}/runs/qwen2_5_7b_lora_wait_k7_full_v1/rollout_locked500/sentence_rollouts.jsonl"

for path in "${V1}" "${V21}" "${V22}" "${K3}"; do
  test -s "${path}" || { echo "Missing locked rollout: ${path}" >&2; exit 1; }
done

EXTRA_SYSTEMS=()
if [[ "${INCLUDE_ALL_WAIT_K:-0}" == "1" ]]; then
  for path in "${K1}" "${K5}" "${K7}"; do
    test -s "${path}" || { echo "Missing optional locked rollout: ${path}" >&2; exit 1; }
  done
  EXTRA_SYSTEMS=(--system "wait_k1=${K1}" --system "wait_k5=${K5}" --system "wait_k7=${K7}")
fi

mkdir -p "${OUTPUT_DIR}/reports" "${OUTPUT_DIR}/figures"
REPORT="${OUTPUT_DIR}/reports/locked500_comparison.json"
if [[ ! -s "${REPORT}" ]]; then
  "${PYTHON_BIN}" "${ROOT}/scripts/compare_locked_rollouts.py" \
    --system "v1=${V1}" \
    --system "v2_1=${V21}" \
    --system "v2_2=${V22}" \
    --system "wait_k3=${K3}" \
    "${EXTRA_SYSTEMS[@]}" \
    --baseline v1 \
    --expected-count 500 \
    --bootstrap-repetitions 10000 \
    --seed 20260821 \
    --output-dir "${OUTPUT_DIR}" \
    > "${OUTPUT_DIR}/reports/run_comparison_stdout.json"
fi

if [[ ! -s "${OUTPUT_DIR}/figures/locked500_comparison.png" || ! -s "${OUTPUT_DIR}/figures/locked500_comparison.pdf" ]]; then
  "${PYTHON_BIN}" "${ROOT}/scripts/plot_locked_comparison.py" \
    --comparison "${REPORT}" \
    --output-prefix "${OUTPUT_DIR}/figures/locked500_comparison"
fi

test -s "${OUTPUT_DIR}/reports/locked500_summary.tsv"
test -s "${OUTPUT_DIR}/reports/reproducibility_manifest.json"
test -s "${OUTPUT_DIR}/figures/locked500_comparison.png"
echo "[locked500-paper-comparison] completed: ${OUTPUT_DIR}"
