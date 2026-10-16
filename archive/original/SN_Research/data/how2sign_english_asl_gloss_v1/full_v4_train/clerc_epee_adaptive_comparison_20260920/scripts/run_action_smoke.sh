#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=/123090274/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train
EXP_ROOT=${DATA_ROOT}/clerc_epee_adaptive_comparison_20260920
MANIFEST=${EXP_ROOT}/manifests/smoke_sources_24.jsonl
ENV_ROOT=/123090274/SN_Research/venvs/clerc-v100-fp16-20260920/site-packages

export PYTHONPATH=${ENV_ROOT}
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_HOME=/123090274/SN_Research/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export NLTK_DATA=${DATA_ROOT}/transllama_learned_adaptive_20260918/nltk_data
export TOKENIZERS_PARALLELISM=false

mkdir -p "${EXP_ROOT}/logs/smoke_action"

run_one() {
  local gpu=$1
  local stage=$2
  local family=$3
  local config=$4
  local adapter=$5
  local output=${EXP_ROOT}/outputs/smoke/${stage}
  local log=${EXP_ROOT}/logs/smoke_action/${stage}.log
  CUDA_VISIBLE_DEVICES=${gpu} python3 "${EXP_ROOT}/scripts/evaluate_action_epee_fp16.py" \
    --config "${config}" \
    --manifest "${MANIFEST}" \
    --adapter-dir "${adapter}" \
    --output-dir "${output}" \
    --stage "${stage}" \
    --system-family "${family}" \
    --expected-count 24 \
    --generation-batch-size 16 >"${log}" 2>&1
}

run_one 0 lexical_v1 lexical \
  "${DATA_ROOT}/fine_tuning_v2_causal_full/configs/qwen2_5_7b_lora_causal_action_full_v1.json" \
  "${DATA_ROOT}/fine_tuning_v2_causal_full/runs/qwen2_5_7b_lora_causal_action_full_corrected_v1/adapter_best" &
p0=$!

run_one 1 causalgloss_20260829 causalgloss \
  "${DATA_ROOT}/state_recovery_v2_3_full/configs/qwen2_5_7b_lora_state_recovery_v2_3_full.json" \
  "${DATA_ROOT}/state_recovery_v2_3_full/runs/qwen2_5_7b_lora_state_recovery_v2_3_full/adapter_final" &
p1=$!

run_one 2 causalgloss_20260904 causalgloss \
  "${DATA_ROOT}/state_recovery_v2_3_multiseed_20260904/configs/qwen2_5_7b_lora_state_recovery_v2_3_seed20260904.json" \
  "${DATA_ROOT}/state_recovery_v2_3_multiseed_20260904/runs/qwen2_5_7b_lora_state_recovery_v2_3_seed20260904/adapter_final" &
p2=$!

run_one 3 causalgloss_20260905 causalgloss \
  "${DATA_ROOT}/state_recovery_v2_3_multiseed_20260905/configs/qwen2_5_7b_lora_state_recovery_v2_3_seed20260905.json" \
  "${DATA_ROOT}/state_recovery_v2_3_multiseed_20260905/runs/qwen2_5_7b_lora_state_recovery_v2_3_seed20260905/adapter_final" &
p3=$!

status=0
for pid in "${p0}" "${p1}" "${p2}" "${p3}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
