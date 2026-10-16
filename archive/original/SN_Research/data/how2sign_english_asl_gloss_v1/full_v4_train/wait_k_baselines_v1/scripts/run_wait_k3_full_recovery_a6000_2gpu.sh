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
CONFIG="${ROOT}/configs/qwen2_5_7b_wait_k3_full_recovery_a6000_2gpu.json"
MANIFESTS="${ROOT}/manifests/k3/full"
RUN="${ROOT}/runs/qwen2_5_7b_lora_wait_k3_full_v1_recovery_a6000_2gpu"

export HF_HOME="${HF_HOME:-/123090274/SN_Research/huggingface_cache}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${ROOT}/logs" "${ROOT}/reports" "${RUN}"

test -s "${MANIFESTS}/train_actions.jsonl"
test -s "${MANIFESTS}/validation_actions.jsonl"
test -s "${MANIFESTS}/test_actions.jsonl"
test -s "${ROOT}/reports/wait_k3_full_validation.json"

CUDA_VISIBLE_DEVICES=0,1 "${TORCHRUN}" --standalone --nproc_per_node=2 \
  "${SHARED}/train_qwen_lora_causal_ddp.py" \
  --config "${CONFIG}" --train-manifest "${MANIFESTS}/train_actions.jsonl" \
  --validation-manifest "${MANIFESTS}/validation_actions.jsonl" \
  --test-manifest "${MANIFESTS}/test_actions.jsonl" \
  --source-adapter-dir "${FT1_ADAPTER}" --output-dir "${RUN}"

"${PYTHON_BIN}" "${ROOT}/scripts/plot_training_history.py" \
  --history "${RUN}/training_history.jsonl" \
  --title "Wait-3 full-data Fine-tune 2 recovery (2 x A6000)" \
  --output-prefix "${RUN}/figures/training_history"

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${SHARED}/evaluate_causal_action_model.py" \
  --config "${CONFIG}" --manifest "${MANIFESTS}/test_actions.jsonl" \
  --adapter-dir "${RUN}/adapter_best" --output-dir "${RUN}/evaluation_locked500" \
  --stage "wait_k3_full_locked500_gold_states" --device cuda:0 --generation-batch-size 16
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${SHARED}/evaluate_free_running_rollout.py" \
  --config "${CONFIG}" --parent-splits "${PAIRS_ROOT}/test_pairs.jsonl" \
  --causal-pairs "${CAUSAL_PAIRS}" --adapter-dir "${RUN}/adapter_best" \
  --output-dir "${RUN}/rollout_locked500" --stage "wait_k3_full_locked500_rollout" \
  --device cuda:0 --generation-batch-size 16

test -s "${RUN}/adapter_best/adapter_model.safetensors"
test -s "${RUN}/figures/training_history.png"
test -s "${RUN}/evaluation_locked500/metrics_summary.json"
test -s "${RUN}/rollout_locked500/rollout_metrics.json"
echo "[wait-k3-full-recovery-a6000-2gpu] completed"
