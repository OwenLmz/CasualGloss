#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/123090274/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train}"
ROOT="${ROOT:-${DATA_ROOT}/wait_k_baselines_v1}"
mkdir -p "${ROOT}/logs"

for K in 1 3 5 7; do
  RUN="${ROOT}/runs/qwen2_5_7b_lora_wait_k${K}_full_v1"
  if [[ -s "${RUN}/rollout_locked500/rollout_metrics.json" ]]; then
    echo "[wait-k-suite] k=${K} already complete; skipping"
    continue
  fi
  bash "${ROOT}/scripts/run_wait_k_full_one.sh" "${K}" \
    > "${ROOT}/logs/run_wait_k${K}_full.log" 2>&1
done
echo "[wait-k-suite] all full baselines completed"
