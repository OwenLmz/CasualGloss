#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?variant is required}"
DATA_ROOT="${DATA_ROOT:-/123090274/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train}"
ROOT="${ROOT:-${DATA_ROOT}/professor_feedback_experiments_20260912}"
PYTHON_BIN="${PYTHON_BIN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/python}"
TORCHRUN="${TORCHRUN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/torchrun}"
V1="${DATA_ROOT}/causal_alignment_v1_corrected_full"
FROZEN="${DATA_ROOT}/state_recovery_v2_3_full/manifests"
FT1_ADAPTER="${DATA_ROOT}/fine_tuning_v1/runs/qwen2_5_7b_lora_how2sign_full_v1/adapter_best"
EXP="${ROOT}/ablations/${VARIANT}"

export HF_HOME="${HF_HOME:-/123090274/SN_Research/huggingface_cache}"
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${EXP}/runs" "${EXP}/evaluation" "${EXP}/logs"

until [[ -s "${ROOT}/reports/PREPARATION_COMPLETE.json" ]]; do sleep 30; done
test -s "${EXP}/manifest_report.json"

if [[ ! -s "${EXP}/runs/stage_a/training_summary.json" ]]; then
  "${TORCHRUN}" --standalone --nproc_per_node=1 "${ROOT}/scripts/train_qwen_lora_causal_ddp.py" \
    --config "${EXP}/configs/stage_a.json" \
    --train-manifest "${EXP}/manifests/stage_a_train.jsonl" \
    --validation-manifest "${EXP}/manifests/development_actions.jsonl" \
    --test-manifest "${EXP}/manifests/confirmation_actions.jsonl" \
    --source-adapter-dir "${FT1_ADAPTER}" \
    --output-dir "${EXP}/runs/stage_a"
fi

if [[ ! -s "${EXP}/runs/stage_b/training_summary.json" ]]; then
  "${TORCHRUN}" --standalone --nproc_per_node=1 "${ROOT}/scripts/train_qwen_lora_causal_ddp.py" \
    --config "${EXP}/configs/stage_b.json" \
    --train-manifest "${EXP}/manifests/stage_b_train.jsonl" \
    --validation-manifest "${EXP}/manifests/development_actions.jsonl" \
    --test-manifest "${EXP}/manifests/confirmation_actions.jsonl" \
    --source-adapter-dir "${EXP}/runs/stage_a/adapter_best" \
    --output-dir "${EXP}/runs/stage_b"
fi

if [[ ! -s "${EXP}/evaluation/confirmation/rollout_metrics.json" ]]; then
  "${PYTHON_BIN}" "${ROOT}/scripts/evaluate_free_running_rollout.py" \
    --config "${EXP}/configs/stage_b.json" \
    --parent-splits "${FROZEN}/confirmation_pairs.jsonl" \
    --causal-pairs "${V1}/outputs/causal_aligned_pairs_full.jsonl" \
    --adapter-dir "${EXP}/runs/stage_b/adapter_best" \
    --output-dir "${EXP}/evaluation/confirmation" \
    --stage "component_ablation_${VARIANT}_confirmation250" \
    --split v2_3_confirmation --device cuda:0 --generation-batch-size 16
fi

printf 'complete\n' > "${EXP}/DONE"

