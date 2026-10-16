#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/123090274/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train}"
ROOT="${ROOT:-${DATA_ROOT}/state_recovery_v2_3_full}"
V22_ROOT="${DATA_ROOT}/fine_tuning_v2_semantic_v2_2_full"
EVAL_PAIRS="${DATA_ROOT}/causal_alignment_v1_corrected_full/outputs/causal_aligned_pairs_full.jsonl"
PYTHON_BIN="${PYTHON_BIN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/python}"
TORCHRUN="${TORCHRUN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/torchrun}"
CONFIG="${ROOT}/configs/qwen2_5_7b_lora_state_recovery_v2_3_full.json"
OLD_CONFIG="${V22_ROOT}/configs/qwen2_5_7b_lora_semantic_causal_v2_2_1_full.json"
SOURCE_RUN="${V22_ROOT}/runs/qwen2_5_7b_lora_semantic_causal_v2_2_1_treebank_full"
SOURCE_ADAPTER="${SOURCE_RUN}/adapter_best"
RUN="${ROOT}/runs/qwen2_5_7b_lora_state_recovery_v2_3_full"

export HF_HOME="${HF_HOME:-/123090274/SN_Research/huggingface_cache}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${ROOT}/logs" "${ROOT}/reports" "${ROOT}/baselines" "${RUN}"
test -s "${ROOT}/reports/screen_gate.json"
test -s "${SOURCE_ADAPTER}/adapter_model.safetensors"
test -s "${ROOT}/manifests/train_actions_full.jsonl"

SCREEN_PASSED=$("${PYTHON_BIN}" -c "import json; print(str(json.load(open('${ROOT}/reports/screen_gate.json'))['full_training_gate_passed']).lower())")
if [[ "${SCREEN_PASSED}" != "true" ]]; then
  echo "v2.3 screen gate did not pass; refusing full training" >&2
  exit 2
fi

echo "[v2.3-full] step 1/6: two-GPU full recovery continuation from original v2.2"
if [[ ! -s "${RUN}/training_summary.json" ]]; then
  CUDA_VISIBLE_DEVICES=0,1 "${TORCHRUN}" --standalone --nproc_per_node=2 \
    "${V22_ROOT}/scripts/train_qwen_lora_causal_ddp.py" \
    --config "${CONFIG}" \
    --train-manifest "${ROOT}/manifests/train_actions_full.jsonl" \
    --validation-manifest "${ROOT}/manifests/development_actions.jsonl" \
    --test-manifest "${ROOT}/manifests/confirmation_actions.jsonl" \
    --source-adapter-dir "${SOURCE_ADAPTER}" \
    --output-dir "${RUN}"
fi

echo "[v2.3-full] step 2/6: development rollouts for best and final"
for candidate in best final; do
  output="${RUN}/rollout_development_${candidate}"
  if [[ ! -s "${output}/rollout_metrics.json" ]]; then
    CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${V22_ROOT}/scripts/evaluate_free_running_rollout.py" \
      --config "${CONFIG}" \
      --parent-splits "${ROOT}/manifests/development_pairs.jsonl" \
      --causal-pairs "${EVAL_PAIRS}" \
      --adapter-dir "${RUN}/adapter_${candidate}" \
      --output-dir "${output}" \
      --stage "v2_3_full_${candidate}_development" \
      --split v2_3_development \
      --device cuda:0 \
      --generation-batch-size 32
  fi
done

echo "[v2.3-full] step 3/6: select checkpoint on development only"
"${PYTHON_BIN}" "${ROOT}/scripts/assess_screen_gate.py" \
  --baseline "${ROOT}/baselines/v2_2_prompt_guard_development/rollout_metrics.json" \
  --candidate best="${RUN}/rollout_development_best/rollout_metrics.json" \
  --candidate final="${RUN}/rollout_development_final/rollout_metrics.json" \
  --output "${ROOT}/reports/full_development_selection.json"

SELECTED=$("${PYTHON_BIN}" -c "import json; print(json.load(open('${ROOT}/reports/full_development_selection.json'))['selected_candidate'])")
SELECTED_ADAPTER="${RUN}/adapter_${SELECTED}"
test -s "${SELECTED_ADAPTER}/adapter_model.safetensors"

echo "[v2.3-full] step 4/6: fixed confirmation baselines"
if [[ ! -s "${ROOT}/baselines/v2_2_native_confirmation/rollout_metrics.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${V22_ROOT}/scripts/evaluate_free_running_rollout.py" \
    --config "${OLD_CONFIG}" \
    --parent-splits "${ROOT}/manifests/confirmation_pairs.jsonl" \
    --causal-pairs "${EVAL_PAIRS}" \
    --adapter-dir "${SOURCE_ADAPTER}" \
    --output-dir "${ROOT}/baselines/v2_2_native_confirmation" \
    --stage v2_2_native_v2_3_confirmation \
    --split v2_3_confirmation \
    --device cuda:0 \
    --generation-batch-size 32
fi
if [[ ! -s "${ROOT}/baselines/v2_2_prompt_guard_confirmation/rollout_metrics.json" ]]; then
  CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" "${V22_ROOT}/scripts/evaluate_free_running_rollout.py" \
    --config "${CONFIG}" \
    --parent-splits "${ROOT}/manifests/confirmation_pairs.jsonl" \
    --causal-pairs "${EVAL_PAIRS}" \
    --adapter-dir "${SOURCE_ADAPTER}" \
    --output-dir "${ROOT}/baselines/v2_2_prompt_guard_confirmation" \
    --stage v2_2_prompt_guard_v2_3_confirmation \
    --split v2_3_confirmation \
    --device cuda:0 \
    --generation-batch-size 32
fi

echo "[v2.3-full] step 5/6: selected model on frozen confirmation"
if [[ ! -s "${RUN}/rollout_confirmation_selected/rollout_metrics.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${V22_ROOT}/scripts/evaluate_free_running_rollout.py" \
    --config "${CONFIG}" \
    --parent-splits "${ROOT}/manifests/confirmation_pairs.jsonl" \
    --causal-pairs "${EVAL_PAIRS}" \
    --adapter-dir "${SELECTED_ADAPTER}" \
    --output-dir "${RUN}/rollout_confirmation_selected" \
    --stage "v2_3_full_${SELECTED}_confirmation" \
    --split v2_3_confirmation \
    --device cuda:0 \
    --generation-batch-size 32
fi

echo "[v2.3-full] step 6/6: paired 10,000-bootstrap confirmation analysis"
"${PYTHON_BIN}" "${ROOT}/scripts/compare_confirmation.py" \
  --native-baseline "${ROOT}/baselines/v2_2_native_confirmation/sentence_rollouts.jsonl" \
  --prompt-baseline "${ROOT}/baselines/v2_2_prompt_guard_confirmation/sentence_rollouts.jsonl" \
  --candidate "${RUN}/rollout_confirmation_selected/sentence_rollouts.jsonl" \
  --selected-checkpoint "${SELECTED}" \
  --output "${ROOT}/reports/confirmation_comparison.json" \
  --bootstrap-repetitions 10000

test -s "${ROOT}/reports/confirmation_comparison.json"
echo "[v2.3-full] completed"
