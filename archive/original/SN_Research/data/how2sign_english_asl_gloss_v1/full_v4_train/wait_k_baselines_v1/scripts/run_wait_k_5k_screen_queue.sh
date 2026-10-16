#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/123090274/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train}"
ROOT="${ROOT:-${DATA_ROOT}/wait_k_baselines_v1}"
mkdir -p "${ROOT}/logs"

for K in 1 5 7; do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting wait-k${K} 5K development screen"
  bash "${ROOT}/scripts/run_wait_k_5k_one.sh" "${K}" \
    > "${ROOT}/logs/run_wait_k${K}_5k_pipeline.log" 2>&1
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] completed wait-k${K} 5K development screen"
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] all Wait-k 5K development screens completed"
