#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/123090274/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train}"
ROOT="${ROOT:-${DATA_ROOT}/fine_tuning_v2_semantic_v2_2_full}"
CAUSAL_ROOT="${CAUSAL_ROOT:-${DATA_ROOT}/semantic_causal_alignment_v2_2_full}"
PYTHON_BIN="${PYTHON_BIN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/python}"
TORCHRUN="${TORCHRUN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/torchrun}"
CONFIG="${ROOT}/configs/qwen2_5_7b_lora_semantic_causal_v2_2_2_recovery_full.json"
SOURCE_RUN="${ROOT}/runs/qwen2_5_7b_lora_semantic_causal_v2_2_1_treebank_full"
RUN="${ROOT}/runs/qwen2_5_7b_lora_semantic_causal_v2_2_2_recovery_full"
TRAIN_MANIFEST="${ROOT}/manifests_recovery_calibration_v2_2_2/train_actions.jsonl"
LOCKED_V1_ROOT="${DATA_ROOT}/causal_alignment_v1_corrected_full"
MANIFESTS="${CAUSAL_ROOT}/manifests_treebank_v2_2_1"

export HF_HOME="${HF_HOME:-/123090274/SN_Research/huggingface_cache}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${ROOT}/logs" "${ROOT}/reports" "${RUN}"
test -s "${TRAIN_MANIFEST}"

SOURCE_SELECTED="best"
if [[ -s "${SOURCE_RUN}/validation_checkpoint_selection.json" ]]; then
  SOURCE_SELECTED=$("${PYTHON_BIN}" -c "import json; print(json.load(open('${SOURCE_RUN}/validation_checkpoint_selection.json'))['selected_candidate'])")
fi
if [[ "${SOURCE_SELECTED}" == "final" ]]; then
  SOURCE_ADAPTER="${SOURCE_RUN}/adapter_final"
else
  SOURCE_ADAPTER="${SOURCE_RUN}/adapter_best"
fi
test -s "${SOURCE_ADAPTER}/adapter_model.safetensors"
echo "[semantic-v2.2.2-recovery] source checkpoint: ${SOURCE_SELECTED}"

echo "[semantic-v2.2.2-recovery] step 1/5: low-rate continuation calibration"
CUDA_VISIBLE_DEVICES=0 "${TORCHRUN}" --standalone --nproc_per_node=1 \
  "${ROOT}/scripts/train_qwen_lora_causal_ddp.py" \
  --config "${CONFIG}" \
  --train-manifest "${TRAIN_MANIFEST}" \
  --validation-manifest "${MANIFESTS}/validation_actions.jsonl" \
  --test-manifest "${MANIFESTS}/test_actions.jsonl" \
  --source-adapter-dir "${SOURCE_ADAPTER}" \
  --output-dir "${RUN}"

echo "[semantic-v2.2.2-recovery] step 2/5: validation rollout for adapter_best"
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${ROOT}/scripts/evaluate_free_running_rollout.py" \
  --config "${CONFIG}" \
  --parent-splits "${MANIFESTS}/validation_pairs.jsonl" \
  --causal-pairs "${CAUSAL_ROOT}/outputs/semantic_causal_aligned_pairs_v2_2_full.jsonl" \
  --adapter-dir "${RUN}/adapter_best" \
  --output-dir "${RUN}/rollout_validation_best" \
  --stage semantic_v2_2_2_recovery_validation_best \
  --device cuda:0 \
  --generation-batch-size 32

echo "[semantic-v2.2.2-recovery] step 3/5: validation rollout for adapter_final"
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${ROOT}/scripts/evaluate_free_running_rollout.py" \
  --config "${CONFIG}" \
  --parent-splits "${MANIFESTS}/validation_pairs.jsonl" \
  --causal-pairs "${CAUSAL_ROOT}/outputs/semantic_causal_aligned_pairs_v2_2_full.jsonl" \
  --adapter-dir "${RUN}/adapter_final" \
  --output-dir "${RUN}/rollout_validation_final" \
  --stage semantic_v2_2_2_recovery_validation_final \
  --device cuda:0 \
  --generation-batch-size 32

"${PYTHON_BIN}" "${DATA_ROOT}/causal_alignment_latency_audit_v2_2/scripts/select_validation_rollout_checkpoint.py" \
  --rollout best="${RUN}/rollout_validation_best/sentence_rollouts.jsonl" \
  --rollout final="${RUN}/rollout_validation_final/sentence_rollouts.jsonl" \
  --output "${RUN}/validation_checkpoint_selection.json"

SELECTED=$("${PYTHON_BIN}" -c "import json; print(json.load(open('${RUN}/validation_checkpoint_selection.json'))['selected_candidate'])")
if [[ "${SELECTED}" == "final" ]]; then
  SELECTED_ADAPTER="${RUN}/adapter_final"
else
  SELECTED_ADAPTER="${RUN}/adapter_best"
fi
echo "[semantic-v2.2.2-recovery] selected checkpoint: ${SELECTED}"

echo "[semantic-v2.2.2-recovery] step 4/5: selected gold-state evaluation"
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${ROOT}/scripts/evaluate_causal_action_model.py" \
  --config "${CONFIG}" \
  --manifest "${MANIFESTS}/test_actions.jsonl" \
  --adapter-dir "${SELECTED_ADAPTER}" \
  --output-dir "${RUN}/evaluation_locked_test_496_selected" \
  --stage semantic_v2_2_2_recovery_selected_gold_states \
  --device cuda:0 \
  --generation-batch-size 32

echo "[semantic-v2.2.2-recovery] step 5/5: selected autonomous locked-500 rollout"
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${ROOT}/scripts/evaluate_free_running_rollout.py" \
  --config "${CONFIG}" \
  --parent-splits "${LOCKED_V1_ROOT}/manifests/test_pairs.jsonl" \
  --causal-pairs "${LOCKED_V1_ROOT}/outputs/causal_aligned_pairs_full.jsonl" \
  --adapter-dir "${SELECTED_ADAPTER}" \
  --output-dir "${RUN}/rollout_locked_500_selected" \
  --stage semantic_v2_2_2_recovery_selected_free_running_locked_500 \
  --device cuda:0 \
  --generation-batch-size 32

test -s "${RUN}/evaluation_locked_test_496_selected/metrics_summary.json"
test -s "${RUN}/rollout_locked_500_selected/rollout_metrics.json"
echo "[semantic-v2.2.2-recovery] completed"
