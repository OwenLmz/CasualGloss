#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/123090274/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train}"
ROOT="${ROOT:-${DATA_ROOT}/fine_tuning_v2_semantic_v2_2_full}"
CAUSAL_ROOT="${CAUSAL_ROOT:-${DATA_ROOT}/semantic_causal_alignment_v2_2_full}"
FT1_ROOT="${FT1_ROOT:-${DATA_ROOT}/fine_tuning_v1}"
PYTHON_BIN="${PYTHON_BIN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/python}"
TORCHRUN="${TORCHRUN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/torchrun}"
CONFIG="${ROOT}/configs/qwen2_5_7b_lora_semantic_causal_v2_2_1_full.json"
RUN="${ROOT}/runs/qwen2_5_7b_lora_semantic_causal_v2_2_1_treebank_full"
SOURCE_ADAPTER="${FT1_ROOT}/runs/qwen2_5_7b_lora_how2sign_full_v1/adapter_best"
LOCKED_V1_ROOT="${DATA_ROOT}/causal_alignment_v1_corrected_full"
MANIFESTS="${CAUSAL_ROOT}/manifests_treebank_v2_2_1"

export HF_HOME="${HF_HOME:-/123090274/SN_Research/huggingface_cache}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${ROOT}/logs" "${ROOT}/reports" "${RUN}"

echo "[semantic-v2.2.1-full] step 1/4: GPU smoke test"
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${ROOT}/scripts/smoke_test_causal_finetuning.py" \
  --config "${CONFIG}" \
  --manifest "${MANIFESTS}/train_actions.jsonl" \
  --source-adapter-dir "${SOURCE_ADAPTER}" \
  --device cuda:0

echo "[semantic-v2.2.1-full] step 2/4: one-epoch one-GPU LoRA training"
CUDA_VISIBLE_DEVICES=0 "${TORCHRUN}" --standalone --nproc_per_node=1 \
  "${ROOT}/scripts/train_qwen_lora_causal_ddp.py" \
  --config "${CONFIG}" \
  --train-manifest "${MANIFESTS}/train_actions.jsonl" \
  --validation-manifest "${MANIFESTS}/validation_actions.jsonl" \
  --test-manifest "${MANIFESTS}/test_actions.jsonl" \
  --source-adapter-dir "${SOURCE_ADAPTER}" \
  --output-dir "${RUN}"

echo "[semantic-v2.2.1-full] step 3/4: gold-state evaluation on 496 scored test parents"
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${ROOT}/scripts/evaluate_causal_action_model.py" \
  --config "${CONFIG}" \
  --manifest "${MANIFESTS}/test_actions.jsonl" \
  --adapter-dir "${RUN}/adapter_best" \
  --output-dir "${RUN}/evaluation_locked_test_496" \
  --stage semantic_v2_2_1_full_gold_states \
  --device cuda:0 \
  --generation-batch-size 32

echo "[semantic-v2.2.1-full] step 4/4: autonomous rollout on unchanged v1 locked 500 parents"
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${ROOT}/scripts/evaluate_free_running_rollout.py" \
  --config "${CONFIG}" \
  --parent-splits "${LOCKED_V1_ROOT}/manifests/test_pairs.jsonl" \
  --causal-pairs "${LOCKED_V1_ROOT}/outputs/causal_aligned_pairs_full.jsonl" \
  --adapter-dir "${RUN}/adapter_best" \
  --output-dir "${RUN}/rollout_locked_500" \
  --stage semantic_v2_2_1_full_free_running_locked_500 \
  --device cuda:0 \
  --generation-batch-size 32

test -s "${RUN}/adapter_best/adapter_model.safetensors"
test -s "${RUN}/evaluation_locked_test_496/metrics_summary.json"
test -s "${RUN}/rollout_locked_500/rollout_metrics.json"
echo "[semantic-v2.2.1-full] completed"
