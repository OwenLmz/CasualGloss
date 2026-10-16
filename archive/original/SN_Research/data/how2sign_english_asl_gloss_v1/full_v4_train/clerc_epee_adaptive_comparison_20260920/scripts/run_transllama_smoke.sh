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

mkdir -p "${EXP_ROOT}/logs/smoke_transllama"

run_one() {
  local gpu=$1
  local seed=$2
  local config=${DATA_ROOT}/transllama_learned_adaptive_20260918/configs/final_seed_${seed}.json
  local adapter=${DATA_ROOT}/transllama_learned_adaptive_20260918/runs/final_seeds/final_seed_${seed}/adapter_best
  local output=${EXP_ROOT}/outputs/smoke/transllama_${seed}
  local log=${EXP_ROOT}/logs/smoke_transllama/transllama_${seed}.log
  CUDA_VISIBLE_DEVICES=${gpu} python3 "${EXP_ROOT}/scripts/evaluate_transllama_epee_fp16.py" \
    --config "${config}" \
    --manifest "${MANIFEST}" \
    --adapter-dir "${adapter}" \
    --output-dir "${output}" \
    --stage "transllama_${seed}" \
    --expected-count 24 \
    --k-values 1,5,8 \
    --generation-batch-size 16 \
    --action-max-new-tokens 16 >"${log}" 2>&1
}

run_one 0 20260829 &
p0=$!
run_one 1 20260904 &
p1=$!
run_one 2 20260905 &
p2=$!

status=0
for pid in "${p0}" "${p1}" "${p2}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
