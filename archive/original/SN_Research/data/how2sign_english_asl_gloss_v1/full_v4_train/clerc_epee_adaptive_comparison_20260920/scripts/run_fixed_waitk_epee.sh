#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=/123090274/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train
EXP_ROOT=${DATA_ROOT}/clerc_epee_adaptive_comparison_20260920
MANIFEST=${DATA_ROOT}/clerc_epee_external_eval_v1/manifests/epee_sources_401.jsonl
SMOKE_MANIFEST=${EXP_ROOT}/manifests/smoke_sources_24.jsonl
ENV_ROOT=/123090274/SN_Research/venvs/clerc-v100-fp16-20260920/site-packages
RUN_ROOT=${EXP_ROOT}/fixed_waitk_extension_20260920

export PYTHONPATH=${ENV_ROOT}:${EXP_ROOT}/scripts
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_HOME=/123090274/SN_Research/huggingface_cache
export TRANSFORMERS_OFFLINE=1
export NLTK_DATA=${DATA_ROOT}/transllama_learned_adaptive_20260918/nltk_data
export TOKENIZERS_PARALLELISM=false

mkdir -p "${RUN_ROOT}/logs/smoke" "${RUN_ROOT}/logs/full" "${RUN_ROOT}/reports/validation"
date -u +%Y-%m-%dT%H:%M:%SZ >"${RUN_ROOT}/RUNNING.started_at_utc.txt"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv \
  >"${RUN_ROOT}/reports/inference_environment.txt"

STAGES=(
  waitk1_seed20260821
  waitk3_seed20260821
  waitk5_seed20260821
  waitk7_seed20260821
  waitk5_seed20260904
  waitk5_seed20260905
  waitk7_seed20260904
  waitk7_seed20260905
)

CONFIGS=(
  "${DATA_ROOT}/wait_k_baselines_v1/configs/qwen2_5_7b_wait_k1_full.json"
  "${DATA_ROOT}/wait_k_baselines_v1/configs/qwen2_5_7b_wait_k3_full_recovery_a6000_2gpu.json"
  "${DATA_ROOT}/wait_k_baselines_v1/configs/qwen2_5_7b_wait_k5_full.json"
  "${DATA_ROOT}/wait_k_baselines_v1/configs/qwen2_5_7b_wait_k7_full.json"
  "${DATA_ROOT}/wait_k_baselines_multiseed_20260914/configs/qwen2_5_7b_wait_k5_full_seed20260904.json"
  "${DATA_ROOT}/wait_k_baselines_multiseed_20260914/configs/qwen2_5_7b_wait_k5_full_seed20260905.json"
  "${DATA_ROOT}/wait_k_baselines_multiseed_20260914/configs/qwen2_5_7b_wait_k7_full_seed20260904.json"
  "${DATA_ROOT}/wait_k_baselines_multiseed_20260914/configs/qwen2_5_7b_wait_k7_full_seed20260905.json"
)

ADAPTERS=(
  "${DATA_ROOT}/wait_k_baselines_v1/runs/qwen2_5_7b_lora_wait_k1_full_v1/adapter_best"
  "${DATA_ROOT}/wait_k_baselines_v1/runs/qwen2_5_7b_lora_wait_k3_full_v1_recovery_a6000_2gpu/adapter_best"
  "${DATA_ROOT}/wait_k_baselines_v1/runs/qwen2_5_7b_lora_wait_k5_full_v1/adapter_best"
  "${DATA_ROOT}/wait_k_baselines_v1/runs/qwen2_5_7b_lora_wait_k7_full_v1/adapter_best"
  "${DATA_ROOT}/wait_k_baselines_multiseed_20260914/runs/qwen2_5_7b_lora_wait_k5_full_seed20260904/adapter_best"
  "${DATA_ROOT}/wait_k_baselines_multiseed_20260914/runs/qwen2_5_7b_lora_wait_k5_full_seed20260905/adapter_best"
  "${DATA_ROOT}/wait_k_baselines_multiseed_20260914/runs/qwen2_5_7b_lora_wait_k7_full_seed20260904/adapter_best"
  "${DATA_ROOT}/wait_k_baselines_multiseed_20260914/runs/qwen2_5_7b_lora_wait_k7_full_seed20260905/adapter_best"
)

run_one() {
  local gpu=$1
  local phase=$2
  local expected=$3
  local manifest=$4
  local index=$5
  local stage=${STAGES[$index]}
  local output=${RUN_ROOT}/outputs/${phase}/${stage}
  if [[ -f "${output}/sentence_rollouts.jsonl" ]]; then
    echo "Reusing complete rollout ${phase}/${stage}"
    return
  fi
  CUDA_VISIBLE_DEVICES=${gpu} python3 "${EXP_ROOT}/scripts/evaluate_fixed_waitk_epee_fp16.py" \
    --config "${CONFIGS[$index]}" \
    --manifest "${manifest}" \
    --adapter-dir "${ADAPTERS[$index]}" \
    --output-dir "${output}" \
    --stage "${stage}" \
    --system-family fixed_wait_k \
    --expected-count "${expected}" \
    --generation-batch-size 16 >"${RUN_ROOT}/logs/${phase}/${stage}.log" 2>&1
}

run_wave() {
  local phase=$1
  local expected=$2
  local manifest=$3
  local base=$4
  local pids=()
  for offset in 0 1 2 3; do
    local index=$((base + offset))
    run_one "${offset}" "${phase}" "${expected}" "${manifest}" "${index}" &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ ${failed} -ne 0 ]]; then
    return 1
  fi
}

validate_phase() {
  local phase=$1
  local expected=$2
  local manifest=$3
  for stage in "${STAGES[@]}"; do
    python3 "${EXP_ROOT}/scripts/validate_rollouts.py" \
      --manifest "${manifest}" \
      --rollout "${RUN_ROOT}/outputs/${phase}/${stage}/sentence_rollouts.jsonl" \
      --expected-count "${expected}" \
      --output "${RUN_ROOT}/reports/validation/${phase}_${stage}.json"
  done
}

run_wave smoke 24 "${SMOKE_MANIFEST}" 0
run_wave smoke 24 "${SMOKE_MANIFEST}" 4
validate_phase smoke 24 "${SMOKE_MANIFEST}"
date -u +%Y-%m-%dT%H:%M:%SZ >"${RUN_ROOT}/SMOKE_GATE_PASSED.at_utc.txt"

run_wave full 401 "${MANIFEST}" 0
run_wave full 401 "${MANIFEST}" 4
validate_phase full 401 "${MANIFEST}"

ANALYSIS_ARGS=(
  --manifest "${MANIFEST}"
  --causal "20260829=${EXP_ROOT}/outputs/full/causalgloss_20260829/sentence_rollouts.jsonl"
  --causal "20260904=${EXP_ROOT}/outputs/full/causalgloss_20260904/sentence_rollouts.jsonl"
  --causal "20260905=${EXP_ROOT}/outputs/full/causalgloss_20260905/sentence_rollouts.jsonl"
  --fixed "1:20260821=${RUN_ROOT}/outputs/full/waitk1_seed20260821/sentence_rollouts.jsonl"
  --fixed "3:20260821=${RUN_ROOT}/outputs/full/waitk3_seed20260821/sentence_rollouts.jsonl"
  --fixed "5:20260821=${RUN_ROOT}/outputs/full/waitk5_seed20260821/sentence_rollouts.jsonl"
  --fixed "5:20260904=${RUN_ROOT}/outputs/full/waitk5_seed20260904/sentence_rollouts.jsonl"
  --fixed "5:20260905=${RUN_ROOT}/outputs/full/waitk5_seed20260905/sentence_rollouts.jsonl"
  --fixed "7:20260821=${RUN_ROOT}/outputs/full/waitk7_seed20260821/sentence_rollouts.jsonl"
  --fixed "7:20260904=${RUN_ROOT}/outputs/full/waitk7_seed20260904/sentence_rollouts.jsonl"
  --fixed "7:20260905=${RUN_ROOT}/outputs/full/waitk7_seed20260905/sentence_rollouts.jsonl"
  --expected-count 401
  --bootstrap-repetitions 20000
  --bootstrap-seed 20260921
  --primary-k 7
)

python3 "${EXP_ROOT}/scripts/compare_fixed_waitk_epee.py" \
  "${ANALYSIS_ARGS[@]}" --output-dir "${RUN_ROOT}/reports/full_analysis"
python3 "${EXP_ROOT}/scripts/compare_fixed_waitk_epee.py" \
  "${ANALYSIS_ARGS[@]}" --output-dir "${RUN_ROOT}/reports/full_analysis_repeat"

cmp "${RUN_ROOT}/reports/full_analysis/comparison_report.json" \
  "${RUN_ROOT}/reports/full_analysis_repeat/comparison_report.json"
sha256sum "${RUN_ROOT}/reports/full_analysis/comparison_report.json" \
  "${RUN_ROOT}/reports/full_analysis_repeat/comparison_report.json" \
  >"${RUN_ROOT}/reports/analysis_repeat_hashes.sha256"

date -u +%Y-%m-%dT%H:%M:%SZ >"${RUN_ROOT}/COMPLETE.at_utc.txt"

