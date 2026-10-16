#!/usr/bin/env bash
set -euo pipefail

K="${1:?usage: run_waitk_seed.sh K SEED GPU}"
SEED="${2:?usage: run_waitk_seed.sh K SEED GPU}"
GPU="${3:?usage: run_waitk_seed.sh K SEED GPU}"

case "${K}" in 5|7) ;; *) echo "K must be 5 or 7" >&2; exit 2 ;; esac
case "${SEED}" in 20260904|20260905) ;; *) echo "unexpected seed" >&2; exit 2 ;; esac
case "${GPU}" in 0|1) ;; *) echo "GPU must be 0 or 1" >&2; exit 2 ;; esac

DATA_ROOT="/123090274/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train"
BASE_ROOT="${DATA_ROOT}/wait_k_baselines_v1"
ROOT="${DATA_ROOT}/wait_k_baselines_multiseed_20260914"
PAIRS_ROOT="${DATA_ROOT}/causal_alignment_v1_corrected_full/manifests"
CAUSAL_PAIRS="${DATA_ROOT}/causal_alignment_v1_corrected_full/outputs/causal_aligned_pairs_full.jsonl"
FT1_ADAPTER="${DATA_ROOT}/fine_tuning_v1/runs/qwen2_5_7b_lora_how2sign_full_v1/adapter_best"
SHARED="${DATA_ROOT}/fine_tuning_v2_semantic_v2_2_full/scripts"
PYTHON_BIN="/123090274/SN_Research/venvs/wlasl-cu128-env/bin/python"
TORCHRUN="/123090274/SN_Research/venvs/wlasl-cu128-env/bin/torchrun"
BASE_CONFIG="${BASE_ROOT}/configs/qwen2_5_7b_wait_k${K}_full.json"
CONFIG="${ROOT}/configs/qwen2_5_7b_wait_k${K}_full_seed${SEED}.json"
MANIFESTS="${BASE_ROOT}/manifests/k${K}/full"
RUN="${ROOT}/runs/qwen2_5_7b_lora_wait_k${K}_full_seed${SEED}"
LOCK="${ROOT}/locks/k${K}_seed${SEED}.lock"

export HF_HOME="/123090274/SN_Research/huggingface_cache"
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${GPU}"

mkdir -p "${ROOT}/configs" "${ROOT}/runs" "${ROOT}/logs" "${ROOT}/locks" "${RUN}"

"${PYTHON_BIN}" - "${BASE_CONFIG}" "${CONFIG}" "${K}" "${SEED}" <<'PY'
import json
import pathlib
import sys

source, destination, wait_k, seed = sys.argv[1:]
with open(source, encoding="utf-8") as handle:
    base = json.load(handle)
config = dict(base)
config["experiment_name"] = f"qwen2_5_7b_lora_wait_k{wait_k}_full_seed{seed}"
config["seed"] = int(seed)

changed = {key for key in config if config[key] != base.get(key)}
if changed != {"experiment_name", "seed"}:
    raise RuntimeError(f"unexpected config changes: {sorted(changed)}")

path = pathlib.Path(destination)
path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY

test -s "${MANIFESTS}/train_actions.jsonl"
test -s "${MANIFESTS}/validation_actions.jsonl"
test -s "${MANIFESTS}/test_actions.jsonl"
test -s "${FT1_ADAPTER}/adapter_model.safetensors"

if ! mkdir "${LOCK}" 2>/dev/null; then
  echo "[k=${K} seed=${SEED}] lock exists; refusing duplicate run" >&2
  exit 3
fi
trap 'rmdir "${LOCK}" 2>/dev/null || true' EXIT

if test -s "${RUN}/training_summary.json" \
  && test -s "${RUN}/evaluation_locked500/metrics_summary.json" \
  && test -s "${RUN}/rollout_locked500/rollout_metrics.json"; then
  echo "[k=${K} seed=${SEED}] final artifacts already exist"
  exit 0
fi

{
  date --iso-8601=seconds
  hostname
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv
  sha256sum "${BASE_CONFIG}" "${CONFIG}" "${MANIFESTS}/train_actions.jsonl"
} > "${RUN}/execution_environment_start.txt"

"${PYTHON_BIN}" "${SHARED}/smoke_test_causal_finetuning.py" \
  --config "${CONFIG}" \
  --manifest "${MANIFESTS}/train_actions.jsonl" \
  --source-adapter-dir "${FT1_ADAPTER}" \
  --device cuda:0

echo "[k=${K} seed=${SEED}] training on physical GPU ${GPU}"
"${TORCHRUN}" --standalone --nproc_per_node=1 \
  "${SHARED}/train_qwen_lora_causal_ddp.py" \
  --config "${CONFIG}" \
  --train-manifest "${MANIFESTS}/train_actions.jsonl" \
  --validation-manifest "${MANIFESTS}/validation_actions.jsonl" \
  --test-manifest "${MANIFESTS}/test_actions.jsonl" \
  --source-adapter-dir "${FT1_ADAPTER}" \
  --output-dir "${RUN}"

"${PYTHON_BIN}" "${BASE_ROOT}/scripts/plot_training_history.py" \
  --history "${RUN}/training_history.jsonl" \
  --title "Wait-${K} full-data seed ${SEED}" \
  --output-prefix "${RUN}/figures/training_history"

"${PYTHON_BIN}" "${SHARED}/evaluate_causal_action_model.py" \
  --config "${CONFIG}" \
  --manifest "${MANIFESTS}/test_actions.jsonl" \
  --adapter-dir "${RUN}/adapter_best" \
  --output-dir "${RUN}/evaluation_locked500" \
  --stage "wait_k${K}_seed${SEED}_locked500_gold_states" \
  --device cuda:0 --generation-batch-size 16

"${PYTHON_BIN}" "${SHARED}/evaluate_free_running_rollout.py" \
  --config "${CONFIG}" \
  --parent-splits "${PAIRS_ROOT}/test_pairs.jsonl" \
  --causal-pairs "${CAUSAL_PAIRS}" \
  --adapter-dir "${RUN}/adapter_best" \
  --output-dir "${RUN}/rollout_locked500" \
  --stage "wait_k${K}_seed${SEED}_locked500_rollout" \
  --split test --device cuda:0 --generation-batch-size 16

test -s "${RUN}/training_summary.json"
test -s "${RUN}/evaluation_locked500/metrics_summary.json"
test -s "${RUN}/rollout_locked500/rollout_metrics.json"
date --iso-8601=seconds > "${RUN}/completed_at.txt"
echo "[k=${K} seed=${SEED}] complete"
