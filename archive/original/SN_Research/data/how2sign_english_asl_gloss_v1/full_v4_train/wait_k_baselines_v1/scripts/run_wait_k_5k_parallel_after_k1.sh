#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/123090274/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train}"
ROOT="${ROOT:-${DATA_ROOT}/wait_k_baselines_v1}"
K1_RUN="${ROOT}/runs/qwen2_5_7b_lora_wait_k1_5k_v1"
OLD_QUEUE_PID="${OLD_QUEUE_PID:-3761}"
LOG="${ROOT}/logs/run_wait_k_5k_parallel_after_k1.log"

mkdir -p "${ROOT}/logs"
exec >> "${LOG}" 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for the active k=1 pipeline"

while pgrep -f "run_wait_k_5k_one.sh 1" >/dev/null; do
  sleep 60
done

for artifact in \
  "${K1_RUN}/adapter_best/adapter_model.safetensors" \
  "${K1_RUN}/figures/training_history.png" \
  "${K1_RUN}/evaluation_validation500/metrics_summary.json" \
  "${K1_RUN}/rollout_validation500/rollout_metrics.json"; do
  if ! test -s "${artifact}"; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] k=1 ended without required artifact: ${artifact}"
    exit 4
  fi
done
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] k=1 artifacts verified"

# The old queue was stopped while its k=1 child finished. Terminate it before
# starting the parallel jobs so it cannot launch duplicate k=5 or k=7 runs.
if kill -0 "${OLD_QUEUE_PID}" 2>/dev/null; then
  kill -TERM "${OLD_QUEUE_PID}" 2>/dev/null || true
  kill -CONT "${OLD_QUEUE_PID}" 2>/dev/null || true
  sleep 2
fi

if pgrep -af "wait_k(5|7)_5k|run_wait_k_5k_one.*(5|7)" >/dev/null; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] an existing k=5 or k=7 process was detected; refusing duplicates"
  pgrep -af "wait_k(5|7)_5k|run_wait_k_5k_one.*(5|7)" || true
  exit 5
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launching k=5 on GPU 0 and k=7 on GPU 1"
CUDA_DEVICE=0 bash "${ROOT}/scripts/run_wait_k_5k_one_device.sh" 5 \
  > "${ROOT}/logs/run_wait_k5_5k_pipeline.parallel.log" 2>&1 &
PID5=$!
CUDA_DEVICE=1 bash "${ROOT}/scripts/run_wait_k_5k_one_device.sh" 7 \
  > "${ROOT}/logs/run_wait_k7_5k_pipeline.parallel.log" 2>&1 &
PID7=$!

STATUS5=0
STATUS7=0
wait "${PID5}" || STATUS5=$?
wait "${PID7}" || STATUS7=$?
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] k=5 status=${STATUS5}; k=7 status=${STATUS7}"

if [[ "${STATUS5}" -ne 0 || "${STATUS7}" -ne 0 ]]; then
  exit 6
fi
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] parallel Wait-k 5K screens completed"
