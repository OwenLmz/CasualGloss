#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/123090274/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train}"
ROOT="${ROOT:-${DATA_ROOT}/fine_tuning_v2_semantic_v2_1_full}"
CAUSAL_ROOT="${CAUSAL_ROOT:-${DATA_ROOT}/semantic_causal_alignment_v2_1_full}"
FT1_ROOT="${FT1_ROOT:-${DATA_ROOT}/fine_tuning_v1}"
PYTHON_BIN="${PYTHON_BIN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/python}"
TORCHRUN="${TORCHRUN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/torchrun}"
CONFIG="${ROOT}/configs/qwen2_5_7b_lora_semantic_causal_v2_1_full.json"
RUN="${ROOT}/runs/qwen2_5_7b_lora_semantic_causal_v2_1_full"
SOURCE_ADAPTER="${FT1_ROOT}/runs/qwen2_5_7b_lora_how2sign_full_v1/adapter_best"

export HF_HOME="${HF_HOME:-/123090274/SN_Research/huggingface_cache}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${ROOT}/logs" "${ROOT}/reports" "${RUN}"

echo "[semantic-v2.1-full] step 1/5: GPU smoke test"
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${ROOT}/scripts/smoke_test_causal_finetuning.py" \
  --config "${CONFIG}" \
  --manifest "${CAUSAL_ROOT}/manifests/train_actions.jsonl" \
  --source-adapter-dir "${SOURCE_ADAPTER}" \
  --device cuda:0

echo "[semantic-v2.1-full] step 2/5: one-epoch two-GPU LoRA training"
"${TORCHRUN}" --standalone --nproc_per_node=2 \
  "${ROOT}/scripts/train_qwen_lora_causal_ddp.py" \
  --config "${CONFIG}" \
  --train-manifest "${CAUSAL_ROOT}/manifests/train_actions.jsonl" \
  --validation-manifest "${CAUSAL_ROOT}/manifests/validation_actions.jsonl" \
  --test-manifest "${CAUSAL_ROOT}/manifests/test_actions.jsonl" \
  --source-adapter-dir "${SOURCE_ADAPTER}" \
  --output-dir "${RUN}"

echo "[semantic-v2.1-full] step 3/5: locked-test gold-state and free-running evaluation"
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${ROOT}/scripts/evaluate_causal_action_model.py" \
  --config "${CONFIG}" \
  --manifest "${CAUSAL_ROOT}/manifests/test_actions.jsonl" \
  --adapter-dir "${RUN}/adapter_best" \
  --output-dir "${RUN}/evaluation_locked_test" \
  --stage semantic_v2_1_full_gold_states \
  --device cuda:0 \
  --generation-batch-size 16 \
  > "${ROOT}/logs/evaluation_locked_test.log" 2>&1 &
ACTION_PID=$!

CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" "${ROOT}/scripts/evaluate_free_running_rollout.py" \
  --config "${CONFIG}" \
  --parent-splits "${CAUSAL_ROOT}/manifests/test_pairs.jsonl" \
  --causal-pairs "${CAUSAL_ROOT}/outputs/semantic_causal_aligned_pairs_v2_1_full.jsonl" \
  --adapter-dir "${RUN}/adapter_best" \
  --output-dir "${RUN}/rollout_locked_test" \
  --stage semantic_v2_1_full_free_running \
  --device cuda:0 \
  --generation-batch-size 16 \
  > "${ROOT}/logs/rollout_locked_test.log" 2>&1 &
ROLLOUT_PID=$!
wait "${ACTION_PID}"
wait "${ROLLOUT_PID}"

echo "[semantic-v2.1-full] step 4/5: result summary"
"${PYTHON_BIN}" "${ROOT}/scripts/summarize_semantic_v2_1_full.py" \
  --history "${RUN}/training_history.jsonl" \
  --training-summary "${RUN}/training_summary.json" \
  --alignment-summary "${CAUSAL_ROOT}/reports/semantic_alignment_v2_1_full_summary.json" \
  --quality-gate "${CAUSAL_ROOT}/reports/quality_gate_v2_1_full.json" \
  --action-metrics "${RUN}/evaluation_locked_test/metrics_summary.json" \
  --rollout-metrics "${RUN}/rollout_locked_test/rollout_metrics.json" \
  --output "${ROOT}/reports/final_results.json"

echo "[semantic-v2.1-full] step 5/5: core checks"
test -s "${RUN}/adapter_best/adapter_model.safetensors"
test -s "${ROOT}/reports/final_results.json"
echo "[semantic-v2.1-full] completed"
