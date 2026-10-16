#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/123090274/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train}"
ROOT="${ROOT:-${DATA_ROOT}/state_recovery_v2_3_full}"
V22_ROOT="${DATA_ROOT}/fine_tuning_v2_semantic_v2_2_full"
ALIGN_ROOT="${DATA_ROOT}/semantic_causal_alignment_v2_2_full"
EVAL_PAIRS="${DATA_ROOT}/causal_alignment_v1_corrected_full/outputs/causal_aligned_pairs_full.jsonl"
PYTHON_BIN="${PYTHON_BIN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/python}"
TORCHRUN="${TORCHRUN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/torchrun}"
CONFIG="${ROOT}/configs/qwen2_5_7b_lora_state_recovery_v2_3_screen.json"
OLD_CONFIG="${V22_ROOT}/configs/qwen2_5_7b_lora_semantic_causal_v2_2_1_full.json"
SOURCE_RUN="${V22_ROOT}/runs/qwen2_5_7b_lora_semantic_causal_v2_2_1_treebank_full"
SOURCE_ADAPTER="${SOURCE_RUN}/adapter_best"
RUN="${ROOT}/runs/qwen2_5_7b_lora_state_recovery_v2_3_screen"

export HF_HOME="${HF_HOME:-/123090274/SN_Research/huggingface_cache}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${ROOT}/logs" "${ROOT}/reports" "${RUN}"
test -s "${SOURCE_ADAPTER}/adapter_model.safetensors"

echo "[v2.3-screen] step 1/6: freeze manifests and protocol"
"${PYTHON_BIN}" "${ROOT}/scripts/prepare_v2_3_manifests.py" \
  --causal-pairs "${ALIGN_ROOT}/outputs/semantic_causal_aligned_pairs_v2_2_full.jsonl" \
  --evaluation-pairs "${EVAL_PAIRS}" \
  --balanced-train-actions "${V22_ROOT}/manifests_wait_balanced_v2_2_1/train_actions.jsonl" \
  --validation-actions "${ALIGN_ROOT}/manifests_treebank_v2_2_1/validation_actions.jsonl" \
  --output-root "${ROOT}"

echo "[v2.3-screen] step 2/6: v2.2 native development baseline"
if [[ ! -s "${ROOT}/baselines/v2_2_native_development/rollout_metrics.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${V22_ROOT}/scripts/evaluate_free_running_rollout.py" \
    --config "${OLD_CONFIG}" \
    --parent-splits "${ROOT}/manifests/development_pairs.jsonl" \
    --causal-pairs "${EVAL_PAIRS}" \
    --adapter-dir "${SOURCE_ADAPTER}" \
    --output-dir "${ROOT}/baselines/v2_2_native_development" \
    --stage v2_2_native_v2_3_development \
    --split v2_3_development \
    --device cuda:0 \
    --generation-batch-size 32
fi

echo "[v2.3-screen] step 3/6: prompt-guard development ablation"
if [[ ! -s "${ROOT}/baselines/v2_2_prompt_guard_development/rollout_metrics.json" ]]; then
  CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" "${V22_ROOT}/scripts/evaluate_free_running_rollout.py" \
    --config "${CONFIG}" \
    --parent-splits "${ROOT}/manifests/development_pairs.jsonl" \
    --causal-pairs "${EVAL_PAIRS}" \
    --adapter-dir "${SOURCE_ADAPTER}" \
    --output-dir "${ROOT}/baselines/v2_2_prompt_guard_development" \
    --stage v2_2_prompt_guard_v2_3_development \
    --split v2_3_development \
    --device cuda:0 \
    --generation-batch-size 32
fi

echo "[v2.3-screen] step 4/6: two-GPU recovery continuation"
if [[ ! -s "${RUN}/training_summary.json" ]]; then
  CUDA_VISIBLE_DEVICES=0,1 "${TORCHRUN}" --standalone --nproc_per_node=2 \
    "${V22_ROOT}/scripts/train_qwen_lora_causal_ddp.py" \
    --config "${CONFIG}" \
    --train-manifest "${ROOT}/manifests/train_actions_screen.jsonl" \
    --validation-manifest "${ROOT}/manifests/development_actions.jsonl" \
    --test-manifest "${ROOT}/manifests/confirmation_actions.jsonl" \
    --source-adapter-dir "${SOURCE_ADAPTER}" \
    --output-dir "${RUN}"
fi

echo "[v2.3-screen] step 5/6: autonomous development rollouts"
for candidate in best final; do
  output="${RUN}/rollout_development_${candidate}"
  if [[ ! -s "${output}/rollout_metrics.json" ]]; then
    CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${V22_ROOT}/scripts/evaluate_free_running_rollout.py" \
      --config "${CONFIG}" \
      --parent-splits "${ROOT}/manifests/development_pairs.jsonl" \
      --causal-pairs "${EVAL_PAIRS}" \
      --adapter-dir "${RUN}/adapter_${candidate}" \
      --output-dir "${output}" \
      --stage "v2_3_screen_${candidate}_development" \
      --split v2_3_development \
      --device cuda:0 \
      --generation-batch-size 32
  fi
done

echo "[v2.3-screen] step 6/6: frozen gate"
"${PYTHON_BIN}" "${ROOT}/scripts/assess_screen_gate.py" \
  --baseline "${ROOT}/baselines/v2_2_prompt_guard_development/rollout_metrics.json" \
  --candidate best="${RUN}/rollout_development_best/rollout_metrics.json" \
  --candidate final="${RUN}/rollout_development_final/rollout_metrics.json" \
  --output "${ROOT}/reports/screen_gate.json"

test -s "${ROOT}/reports/screen_gate.json"
echo "[v2.3-screen] completed"
