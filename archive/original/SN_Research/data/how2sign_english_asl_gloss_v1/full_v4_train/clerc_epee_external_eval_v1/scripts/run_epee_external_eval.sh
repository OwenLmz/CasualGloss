#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/123090274/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train}"
ROOT="${ROOT:-${DATA_ROOT}/clerc_epee_external_eval_v1}"
PYTHON_BIN="${PYTHON_BIN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/python}"
COMMIT="93bc8aa0a6526e86af80db5f59af8d133292c9b5"
METADATA="${ROOT}/raw/epee_v0_3_${COMMIT}/metadata.csv"
MANIFEST="${ROOT}/manifests/epee_sources_401.jsonl"

V1_CONFIG="${DATA_ROOT}/fine_tuning_v2_causal_full/configs/qwen2_5_7b_lora_causal_action_full_v1.json"
V1_ADAPTER="${DATA_ROOT}/fine_tuning_v2_causal_full/runs/qwen2_5_7b_lora_causal_action_full_corrected_v1/adapter_best"
V22_CONFIG="${DATA_ROOT}/fine_tuning_v2_semantic_v2_2_full/configs/qwen2_5_7b_lora_semantic_causal_v2_2_1_full.json"
V22_ADAPTER="${DATA_ROOT}/fine_tuning_v2_semantic_v2_2_full/runs/qwen2_5_7b_lora_semantic_causal_v2_2_1_treebank_full/adapter_best"

mkdir -p "$(dirname "${METADATA}")" "${ROOT}/logs" "${ROOT}/reports" "${ROOT}/outputs"
test ! -e "${ROOT}/outputs/v1/sentence_rollouts.jsonl" || { echo "v1 output already exists" >&2; exit 1; }
test ! -e "${ROOT}/outputs/v2_2/sentence_rollouts.jsonl" || { echo "v2.2 output already exists" >&2; exit 1; }

"${PYTHON_BIN}" "${ROOT}/scripts/test_epee_protocol.py"

if [[ ! -s "${METADATA}" ]]; then
  curl -L --fail --silent --show-error \
    "https://huggingface.co/datasets/CLERC-DATA/epee/resolve/${COMMIT}/metadata.csv" \
    -o "${METADATA}"
fi

if [[ ! -s "${MANIFEST}" ]]; then
  "${PYTHON_BIN}" "${ROOT}/scripts/prepare_epee_external_eval.py" \
    --metadata "${METADATA}" \
    --output-dir "${ROOT}" \
    --dataset-commit "${COMMIT}" \
    > "${ROOT}/logs/dataset_audit.log" 2>&1
fi

test "$(wc -l < "${MANIFEST}")" -eq 401
test -s "${ROOT}/reports/dataset_audit.json"

sha256sum \
  "${ROOT}/scripts/prepare_epee_external_eval.py" \
  "${ROOT}/scripts/evaluate_epee_free_running.py" \
  "${ROOT}/scripts/compare_epee_multi_reference.py" \
  "${METADATA}" \
  "${V1_ADAPTER}/adapter_model.safetensors" \
  "${V22_ADAPTER}/adapter_model.safetensors" \
  > "${ROOT}/reports/pre_run_checksums.sha256"

export HF_HOME="${HF_HOME:-/123090274/SN_Research/huggingface_cache}"
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${ROOT}/scripts/evaluate_epee_free_running.py" \
  --config "${V1_CONFIG}" \
  --manifest "${MANIFEST}" \
  --adapter-dir "${V1_ADAPTER}" \
  --output-dir "${ROOT}/outputs/v1" \
  --stage v1_epee_external_native \
  --device cuda:0 \
  --generation-batch-size 32 \
  > "${ROOT}/logs/v1_rollout.log" 2>&1 &
V1_PID=$!

CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" "${ROOT}/scripts/evaluate_epee_free_running.py" \
  --config "${V22_CONFIG}" \
  --manifest "${MANIFEST}" \
  --adapter-dir "${V22_ADAPTER}" \
  --output-dir "${ROOT}/outputs/v2_2" \
  --stage v2_2_epee_external_native \
  --device cuda:0 \
  --generation-batch-size 32 \
  > "${ROOT}/logs/v2_2_rollout.log" 2>&1 &
V22_PID=$!

wait "${V1_PID}"
wait "${V22_PID}"

test "$(wc -l < "${ROOT}/outputs/v1/sentence_rollouts.jsonl")" -eq 401
test "$(wc -l < "${ROOT}/outputs/v2_2/sentence_rollouts.jsonl")" -eq 401

"${PYTHON_BIN}" "${ROOT}/scripts/compare_epee_multi_reference.py" \
  --manifest "${MANIFEST}" \
  --v1-rollouts "${ROOT}/outputs/v1/sentence_rollouts.jsonl" \
  --v2-2-rollouts "${ROOT}/outputs/v2_2/sentence_rollouts.jsonl" \
  --output-dir "${ROOT}/reports" \
  --bootstrap-repetitions 10000 \
  --seed 20260829 \
  --noninferiority-margin -0.03 \
  > "${ROOT}/logs/comparison.log" 2>&1

test -s "${ROOT}/reports/external_comparison.json"
test -s "${ROOT}/reports/summary.tsv"
test -s "${ROOT}/reports/external_quality_difference.png"
printf 'CLERC Epee external evaluation completed\n'
