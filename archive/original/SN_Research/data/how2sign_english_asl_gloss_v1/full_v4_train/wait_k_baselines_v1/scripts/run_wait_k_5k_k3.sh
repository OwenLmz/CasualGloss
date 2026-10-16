#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/123090274/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train}"
ROOT="${ROOT:-${DATA_ROOT}/wait_k_baselines_v1}"
PAIRS_ROOT="${DATA_ROOT}/causal_alignment_v1_corrected_full/manifests"
CAUSAL_PAIRS="${DATA_ROOT}/causal_alignment_v1_corrected_full/outputs/causal_aligned_pairs_full.jsonl"
FT1_ADAPTER="${DATA_ROOT}/fine_tuning_v1/runs/qwen2_5_7b_lora_how2sign_full_v1/adapter_best"
SHARED="${DATA_ROOT}/fine_tuning_v2_semantic_v2_2_full/scripts"
PYTHON_BIN="${PYTHON_BIN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/python}"
TORCHRUN="${TORCHRUN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/torchrun}"
CONFIG="${ROOT}/configs/qwen2_5_7b_wait_k3_5k.json"
MANIFESTS="${ROOT}/manifests/k3/5k"
RUN="${ROOT}/runs/qwen2_5_7b_lora_wait_k3_5k_v1"

export HF_HOME="${HF_HOME:-/123090274/SN_Research/huggingface_cache}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${ROOT}/logs" "${ROOT}/reports" "${RUN}"

"${PYTHON_BIN}" "${ROOT}/scripts/build_wait_k_actions.py" \
  --pairs-root "${PAIRS_ROOT}" --output-root "${ROOT}/manifests" \
  --wait-k 3 --scale-name 5k --train-limit 5000 --validation-limit 500 --test-limit 0
"${PYTHON_BIN}" "${ROOT}/scripts/validate_wait_k_actions.py" \
  --pairs-root "${PAIRS_ROOT}" --manifest-root "${MANIFESTS}" --wait-k 3 \
  --train-limit 5000 --validation-limit 500 --test-limit 0 \
  --output-report "${ROOT}/reports/wait_k3_5k_validation.json"

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${SHARED}/smoke_test_causal_finetuning.py" \
  --config "${CONFIG}" --manifest "${MANIFESTS}/train_actions.jsonl" \
  --source-adapter-dir "${FT1_ADAPTER}" --device cuda:0

CUDA_VISIBLE_DEVICES=0 "${TORCHRUN}" --standalone --nproc_per_node=1 \
  "${SHARED}/train_qwen_lora_causal_ddp.py" \
  --config "${CONFIG}" --train-manifest "${MANIFESTS}/train_actions.jsonl" \
  --validation-manifest "${MANIFESTS}/validation_actions.jsonl" \
  --test-manifest "${MANIFESTS}/test_actions.jsonl" \
  --source-adapter-dir "${FT1_ADAPTER}" --output-dir "${RUN}"

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${SHARED}/evaluate_causal_action_model.py" \
  --config "${CONFIG}" --manifest "${MANIFESTS}/validation_actions.jsonl" \
  --adapter-dir "${RUN}/adapter_best" --output-dir "${RUN}/evaluation_validation500" \
  --stage wait_k3_5k_validation_gold_states --device cuda:0 --generation-batch-size 16

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${SHARED}/evaluate_free_running_rollout.py" \
  --config "${CONFIG}" --parent-splits "${PAIRS_ROOT}/validation_pairs.jsonl" \
  --causal-pairs "${CAUSAL_PAIRS}" --adapter-dir "${RUN}/adapter_best" \
  --output-dir "${RUN}/rollout_validation500" --stage wait_k3_5k_validation_rollout \
  --split validation \
  --device cuda:0 --generation-batch-size 16

test -s "${RUN}/adapter_best/adapter_model.safetensors"
test -s "${RUN}/evaluation_validation500/metrics_summary.json"
test -s "${RUN}/rollout_validation500/rollout_metrics.json"
echo "[wait-k3-5k] completed without touching locked test"
