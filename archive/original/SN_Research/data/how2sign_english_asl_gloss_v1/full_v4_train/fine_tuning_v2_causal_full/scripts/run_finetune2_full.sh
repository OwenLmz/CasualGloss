#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/123090274/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train}"
ROOT="${ROOT:-${DATA_ROOT}/fine_tuning_v2_causal_full}"
CAUSAL_ROOT="${CAUSAL_ROOT:-${DATA_ROOT}/causal_alignment_v1_corrected_full}"
CONTROL_ROOT="${CONTROL_ROOT:-${DATA_ROOT}/fine_tuning_v2_causal_5k}"
CONTROL_DATA_ROOT="${CONTROL_DATA_ROOT:-${DATA_ROOT}/causal_alignment_v1_corrected_5k}"
FT1_ROOT="${FT1_ROOT:-${DATA_ROOT}/fine_tuning_v1}"
PYTHON_BIN="${PYTHON_BIN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/python}"
TORCHRUN="${TORCHRUN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/torchrun}"
CONFIG="${ROOT}/configs/qwen2_5_7b_lora_causal_action_full_v1.json"
RUN="${ROOT}/runs/qwen2_5_7b_lora_causal_action_full_corrected_v1"
RESULTS="${ROOT}/results/controlled_full"
SOURCE_ADAPTER="${FT1_ROOT}/runs/qwen2_5_7b_lora_how2sign_full_v1/adapter_best"
CONTROL_RUN="${CONTROL_ROOT}/runs/qwen2_5_7b_lora_causal_action_5k_corrected_v1"

export HF_HOME="${HF_HOME:-/123090274/SN_Research/huggingface_cache}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${ROOT}/logs" "${ROOT}/reports" "${RUN}" "${RESULTS}"

echo "[full-ft2] step 1/6: exact-comparison preflight and smoke test"
"${PYTHON_BIN}" "${ROOT}/scripts/audit_full_experiment_preflight.py" \
  --full-root "${CAUSAL_ROOT}" \
  --control-5k-root "${CONTROL_DATA_ROOT}" \
  --output-report "${ROOT}/reports/full_experiment_preflight.json"

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${ROOT}/scripts/smoke_test_causal_finetuning.py" \
  --config "${CONFIG}" \
  --manifest "${CAUSAL_ROOT}/manifests/train_actions.jsonl" \
  --source-adapter-dir "${SOURCE_ADAPTER}" \
  --device cuda:0

echo "[full-ft2] step 2/6: one-epoch four-GPU LoRA training from Fine-tune 1"
"${TORCHRUN}" --standalone --nproc_per_node=4 \
  "${ROOT}/scripts/train_qwen_lora_causal_ddp.py" \
  --config "${CONFIG}" \
  --train-manifest "${CAUSAL_ROOT}/manifests/train_actions.jsonl" \
  --validation-manifest "${CAUSAL_ROOT}/manifests/validation_actions.jsonl" \
  --test-manifest "${CAUSAL_ROOT}/manifests/test_actions.jsonl" \
  --source-adapter-dir "${SOURCE_ADAPTER}" \
  --output-dir "${RUN}"

echo "[full-ft2] step 3/6: parallel gold-state and free-running evaluation"
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${ROOT}/scripts/evaluate_causal_action_model.py" \
  --config "${CONFIG}" \
  --manifest "${CAUSAL_ROOT}/manifests/test_actions.jsonl" \
  --adapter-dir "${RUN}/adapter_best" \
  --output-dir "${RUN}/evaluation_full" \
  --stage finetune2_full_on_locked_test_gold_states \
  --device cuda:0 \
  --generation-batch-size 16 \
  > "${ROOT}/logs/evaluation_full.log" 2>&1 &
ACTION_PID=$!

CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" "${ROOT}/scripts/evaluate_free_running_rollout.py" \
  --config "${CONFIG}" \
  --parent-splits "${CAUSAL_ROOT}/manifests/parent_splits_full.jsonl" \
  --causal-pairs "${CAUSAL_ROOT}/outputs/causal_aligned_pairs_full.jsonl" \
  --adapter-dir "${RUN}/adapter_best" \
  --output-dir "${RUN}/rollout_full" \
  --stage finetune2_full_free_running_on_locked_test \
  --device cuda:0 \
  --generation-batch-size 16 \
  > "${ROOT}/logs/rollout_full.log" 2>&1 &
ROLLOUT_PID=$!

wait "${ACTION_PID}"
wait "${ROLLOUT_PID}"

echo "[full-ft2] step 4/6: paired 5K-versus-full bootstrap comparison"
"${PYTHON_BIN}" "${ROOT}/scripts/summarize_full_experiment.py" \
  --history "${RUN}/training_history.jsonl" \
  --training-summary "${RUN}/training_summary.json" \
  --data-report "${CAUSAL_ROOT}/reports/finalization_report.json" \
  --preflight-report "${ROOT}/reports/full_experiment_preflight.json" \
  --control-action "${CONTROL_RUN}/evaluation_finetune2_5k/metrics_summary.json" \
  --full-action "${RUN}/evaluation_full/metrics_summary.json" \
  --control-rollouts "${CONTROL_RUN}/rollout_finetune2_5k/sentence_rollouts.jsonl" \
  --full-rollouts "${RUN}/rollout_full/sentence_rollouts.jsonl" \
  --output-dir "${RESULTS}"

echo "[full-ft2] step 5/6: structural result validation"
"${PYTHON_BIN}" "${ROOT}/scripts/validate_full_experiment.py" \
  --root "${ROOT}" \
  --data-root "${CAUSAL_ROOT}" \
  --control-5k-root "${CONTROL_ROOT}" \
  --config "${CONFIG}" \
  --output-report "${ROOT}/reports/final_validation_report.json" \
  --hash-report "${ROOT}/reports/core_file_sha256.json"

echo "[full-ft2] step 6/6: completed"
echo "[full-ft2] result: ${RESULTS}/RESULTS.md"
echo "[full-ft2] decision: ${RESULTS}/metrics_comparison.json"
