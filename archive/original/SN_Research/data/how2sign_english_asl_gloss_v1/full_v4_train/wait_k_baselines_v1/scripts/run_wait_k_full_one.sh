#!/usr/bin/env bash
set -euo pipefail

K="${1:?usage: run_wait_k_full_one.sh K}"
case "${K}" in 1|3|5|7) ;; *) echo "K must be one of 1,3,5,7" >&2; exit 2 ;; esac

DATA_ROOT="${DATA_ROOT:-/123090274/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train}"
ROOT="${ROOT:-${DATA_ROOT}/wait_k_baselines_v1}"
PAIRS_ROOT="${DATA_ROOT}/causal_alignment_v1_corrected_full/manifests"
CAUSAL_PAIRS="${DATA_ROOT}/causal_alignment_v1_corrected_full/outputs/causal_aligned_pairs_full.jsonl"
FT1_ADAPTER="${DATA_ROOT}/fine_tuning_v1/runs/qwen2_5_7b_lora_how2sign_full_v1/adapter_best"
SHARED="${DATA_ROOT}/fine_tuning_v2_semantic_v2_2_full/scripts"
PYTHON_BIN="${PYTHON_BIN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/python}"
TORCHRUN="${TORCHRUN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/torchrun}"
CONFIG="${ROOT}/configs/qwen2_5_7b_wait_k${K}_full.json"
MANIFESTS="${ROOT}/manifests/k${K}/full"
RUN="${ROOT}/runs/qwen2_5_7b_lora_wait_k${K}_full_v1"

export HF_HOME="${HF_HOME:-/123090274/SN_Research/huggingface_cache}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${ROOT}/logs" "${ROOT}/reports" "${RUN}"

"${PYTHON_BIN}" "${ROOT}/scripts/build_wait_k_actions.py" \
  --pairs-root "${PAIRS_ROOT}" --output-root "${ROOT}/manifests" \
  --wait-k "${K}" --scale-name full --train-limit -1 --validation-limit -1 --test-limit -1
"${PYTHON_BIN}" "${ROOT}/scripts/validate_wait_k_actions.py" \
  --pairs-root "${PAIRS_ROOT}" --manifest-root "${MANIFESTS}" --wait-k "${K}" \
  --train-limit -1 --validation-limit -1 --test-limit -1 \
  --output-report "${ROOT}/reports/wait_k${K}_full_validation.json"

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${SHARED}/smoke_test_causal_finetuning.py" \
  --config "${CONFIG}" --manifest "${MANIFESTS}/train_actions.jsonl" \
  --source-adapter-dir "${FT1_ADAPTER}" --device cuda:0
CUDA_VISIBLE_DEVICES=0 "${TORCHRUN}" --standalone --nproc_per_node=1 \
  "${SHARED}/train_qwen_lora_causal_ddp.py" \
  --config "${CONFIG}" --train-manifest "${MANIFESTS}/train_actions.jsonl" \
  --validation-manifest "${MANIFESTS}/validation_actions.jsonl" \
  --test-manifest "${MANIFESTS}/test_actions.jsonl" \
  --source-adapter-dir "${FT1_ADAPTER}" --output-dir "${RUN}"

"${PYTHON_BIN}" "${ROOT}/scripts/plot_training_history.py" \
  --history "${RUN}/training_history.jsonl" \
  --title "Wait-${K} full-data Fine-tune 2" \
  --output-prefix "${RUN}/figures/training_history"

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${SHARED}/evaluate_causal_action_model.py" \
  --config "${CONFIG}" --manifest "${MANIFESTS}/test_actions.jsonl" \
  --adapter-dir "${RUN}/adapter_best" --output-dir "${RUN}/evaluation_locked500" \
  --stage "wait_k${K}_full_locked500_gold_states" --device cuda:0 --generation-batch-size 16
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${SHARED}/evaluate_free_running_rollout.py" \
  --config "${CONFIG}" --parent-splits "${PAIRS_ROOT}/test_pairs.jsonl" \
  --causal-pairs "${CAUSAL_PAIRS}" --adapter-dir "${RUN}/adapter_best" \
  --output-dir "${RUN}/rollout_locked500" --stage "wait_k${K}_full_locked500_rollout" \
  --device cuda:0 --generation-batch-size 16

test -s "${RUN}/adapter_best/adapter_model.safetensors"
test -s "${RUN}/figures/training_history.png"
test -s "${RUN}/evaluation_locked500/metrics_summary.json"
test -s "${RUN}/rollout_locked500/rollout_metrics.json"
echo "[wait-k${K}-full] completed"
