#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/123090274/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train}"
ROOT="${ROOT:-${DATA_ROOT}/wait_k_baselines_v1}"
PAIRS_ROOT="${DATA_ROOT}/causal_alignment_v1_corrected_full/manifests"
PYTHON_BIN="${PYTHON_BIN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/python}"

mkdir -p "${ROOT}/logs" "${ROOT}/reports"
for K in 1 3 5 7; do
  MANIFEST_ROOT="${ROOT}/manifests/k${K}/pilot500"
  "${PYTHON_BIN}" "${ROOT}/scripts/build_wait_k_actions.py" \
    --pairs-root "${PAIRS_ROOT}" \
    --output-root "${ROOT}/manifests" \
    --wait-k "${K}" \
    --scale-name pilot500 \
    --train-limit 500 \
    --validation-limit 100 \
    --test-limit 0 \
    > "${ROOT}/logs/build_wait_k${K}_pilot500.log"
  "${PYTHON_BIN}" "${ROOT}/scripts/validate_wait_k_actions.py" \
    --pairs-root "${PAIRS_ROOT}" \
    --manifest-root "${MANIFEST_ROOT}" \
    --wait-k "${K}" \
    --train-limit 500 \
    --validation-limit 100 \
    --test-limit 0 \
    --output-report "${ROOT}/reports/wait_k${K}_pilot500_validation.json" \
    > "${ROOT}/logs/validate_wait_k${K}_pilot500.log"
done
echo "[wait-k-pilot] all k values validated"
