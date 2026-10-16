#!/usr/bin/env bash
set -euo pipefail

SEED="${1:?usage: queue_waitk5_after_k7.sh SEED GPU}"
GPU="${2:?usage: queue_waitk5_after_k7.sh SEED GPU}"

PROJECT="/123090274/SN_Research"
RUNNER="${PROJECT}/run_waitk_seed_20260914.sh"
ROOT="${PROJECT}/data/how2sign_english_asl_gloss_v1/full_v4_train/wait_k_baselines_multiseed_20260914"
K7_RUN="${ROOT}/runs/qwen2_5_7b_lora_wait_k7_full_seed${SEED}"

while pgrep -f "^bash ${RUNNER} 7 ${SEED} ${GPU}$" >/dev/null; do
  sleep 120
done

if ! test -s "${K7_RUN}/completed_at.txt"; then
  echo "[seed=${SEED}] k=7 did not complete; k=5 will not start" >&2
  exit 1
fi

exec "${RUNNER}" 5 "${SEED}" "${GPU}"
