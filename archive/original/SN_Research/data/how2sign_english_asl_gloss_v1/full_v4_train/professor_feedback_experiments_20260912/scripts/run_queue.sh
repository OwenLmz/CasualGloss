#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?gpu is required}"
shift
ROOT="${ROOT:-/123090274/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train/professor_feedback_experiments_20260912}"
for VARIANT in "$@"; do
  CUDA_VISIBLE_DEVICES="${GPU}" bash "${ROOT}/scripts/run_ablation.sh" "${VARIANT}" \
    > "${ROOT}/logs/${VARIANT}.log" 2>&1
done

