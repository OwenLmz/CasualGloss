#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/123090274/SN_Research}"
DATA_ROOT="${PROJECT_ROOT}/data/how2sign_english_asl_gloss_v1/full_v4_train"
V23_ROOT="${DATA_ROOT}/state_recovery_v2_3_full"
V22_ROOT="${DATA_ROOT}/fine_tuning_v2_semantic_v2_2_full"
EXP_ROOT="${DATA_ROOT}/state_recovery_v2_3_multiseed_20260904"
PYTHON_BIN="${PROJECT_ROOT}/venvs/wlasl-cu128-env/bin/python"
TORCHRUN="${PROJECT_ROOT}/venvs/wlasl-cu128-env/bin/torchrun"
CONFIG="${EXP_ROOT}/configs/qwen2_5_7b_lora_state_recovery_v2_3_seed20260904.json"
OLD_CONFIG="${V22_ROOT}/configs/qwen2_5_7b_lora_semantic_causal_v2_2_1_full.json"
SOURCE_ADAPTER="${V22_ROOT}/runs/qwen2_5_7b_lora_semantic_causal_v2_2_1_treebank_full/adapter_best"
EVAL_PAIRS="${DATA_ROOT}/causal_alignment_v1_corrected_full/outputs/causal_aligned_pairs_full.jsonl"
RUN="${EXP_ROOT}/runs/qwen2_5_7b_lora_state_recovery_v2_3_seed20260904"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/huggingface_cache}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${EXP_ROOT}/logs" "${EXP_ROOT}/reports" "${RUN}"

test -s "${CONFIG}"
test -s "${SOURCE_ADAPTER}/adapter_model.safetensors"
test -s "${V23_ROOT}/manifests/train_actions_full.jsonl"
test -s "${V23_ROOT}/manifests/development_actions.jsonl"
test -s "${V23_ROOT}/manifests/confirmation_actions.jsonl"

echo "[seed-20260904] step 1/5: two-GPU v2.3 recovery training"
if [[ ! -s "${RUN}/training_summary.json" ]]; then
  CUDA_VISIBLE_DEVICES=0,1 "${TORCHRUN}" --standalone --nproc_per_node=2 \
    "${V22_ROOT}/scripts/train_qwen_lora_causal_ddp.py" \
    --config "${CONFIG}" \
    --train-manifest "${V23_ROOT}/manifests/train_actions_full.jsonl" \
    --validation-manifest "${V23_ROOT}/manifests/development_actions.jsonl" \
    --test-manifest "${V23_ROOT}/manifests/confirmation_actions.jsonl" \
    --source-adapter-dir "${SOURCE_ADAPTER}" \
    --output-dir "${RUN}"
fi

echo "[seed-20260904] step 2/5: development rollouts"
for candidate in best final; do
  output="${RUN}/rollout_development_${candidate}"
  if [[ ! -s "${output}/rollout_metrics.json" ]]; then
    CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${V22_ROOT}/scripts/evaluate_free_running_rollout.py" \
      --config "${CONFIG}" \
      --parent-splits "${V23_ROOT}/manifests/development_pairs.jsonl" \
      --causal-pairs "${EVAL_PAIRS}" \
      --adapter-dir "${RUN}/adapter_${candidate}" \
      --output-dir "${output}" \
      --stage "v2_3_seed20260904_${candidate}_development" \
      --split v2_3_development \
      --device cuda:0 \
      --generation-batch-size 32
  fi
done

echo "[seed-20260904] step 3/5: checkpoint selection on development only"
"${PYTHON_BIN}" "${V23_ROOT}/scripts/assess_screen_gate.py" \
  --baseline "${V23_ROOT}/baselines/v2_2_prompt_guard_development/rollout_metrics.json" \
  --candidate best="${RUN}/rollout_development_best/rollout_metrics.json" \
  --candidate final="${RUN}/rollout_development_final/rollout_metrics.json" \
  --output "${EXP_ROOT}/reports/development_selection_seed20260904.json"

SELECTED=$("${PYTHON_BIN}" -c "import json; print(json.load(open('${EXP_ROOT}/reports/development_selection_seed20260904.json'))['selected_candidate'])")
SELECTED_ADAPTER="${RUN}/adapter_${SELECTED}"
test -s "${SELECTED_ADAPTER}/adapter_model.safetensors"

echo "[seed-20260904] step 4/5: frozen confirmation rollout"
if [[ ! -s "${RUN}/rollout_confirmation_selected/rollout_metrics.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${V22_ROOT}/scripts/evaluate_free_running_rollout.py" \
    --config "${CONFIG}" \
    --parent-splits "${V23_ROOT}/manifests/confirmation_pairs.jsonl" \
    --causal-pairs "${EVAL_PAIRS}" \
    --adapter-dir "${SELECTED_ADAPTER}" \
    --output-dir "${RUN}/rollout_confirmation_selected" \
    --stage "v2_3_seed20260904_${SELECTED}_confirmation" \
    --split v2_3_confirmation \
    --device cuda:0 \
    --generation-batch-size 32
fi

echo "[seed-20260904] step 5/5: paired confirmation analysis"
"${PYTHON_BIN}" "${V23_ROOT}/scripts/compare_confirmation.py" \
  --native-baseline "${V23_ROOT}/baselines/v2_2_native_confirmation/sentence_rollouts.jsonl" \
  --prompt-baseline "${V23_ROOT}/baselines/v2_2_prompt_guard_confirmation/sentence_rollouts.jsonl" \
  --candidate "${RUN}/rollout_confirmation_selected/sentence_rollouts.jsonl" \
  --selected-checkpoint "${SELECTED}" \
  --output "${EXP_ROOT}/reports/confirmation_comparison_seed20260904.json" \
  --bootstrap-repetitions 10000

test -s "${EXP_ROOT}/reports/confirmation_comparison_seed20260904.json"
echo "[seed-20260904] completed"
