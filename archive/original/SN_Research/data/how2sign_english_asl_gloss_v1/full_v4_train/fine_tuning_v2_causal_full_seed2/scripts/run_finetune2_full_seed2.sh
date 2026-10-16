#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/123090274/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train}"
ROOT="${ROOT:-${DATA_ROOT}/fine_tuning_v2_causal_full_seed2}"
CAUSAL_ROOT="${CAUSAL_ROOT:-${DATA_ROOT}/causal_alignment_v1_corrected_full}"
SEED1_FULL_ROOT="${SEED1_FULL_ROOT:-${DATA_ROOT}/fine_tuning_v2_causal_full}"
SEED2_5K_ROOT="${SEED2_5K_ROOT:-${DATA_ROOT}/fine_tuning_v2_causal_5k_seed2}"
CONTROL_DATA_ROOT="${CONTROL_DATA_ROOT:-${DATA_ROOT}/causal_alignment_v1_corrected_5k}"
FT1_ROOT="${FT1_ROOT:-${DATA_ROOT}/fine_tuning_v1}"
PYTHON_BIN="${PYTHON_BIN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/python}"
TORCHRUN="${TORCHRUN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/torchrun}"
CONFIG="${ROOT}/configs/qwen2_5_7b_lora_causal_action_full_seed2_v1.json"
SEED1_CONFIG="${SEED1_FULL_ROOT}/configs/qwen2_5_7b_lora_causal_action_full_v1.json"
RUN="${ROOT}/runs/qwen2_5_7b_lora_causal_action_full_seed2_v1"
RESULTS="${ROOT}/results/controlled_full_seed2"
STABILITY_RESULTS="${ROOT}/results/seed_stability_full"
SOURCE_ADAPTER="${FT1_ROOT}/runs/qwen2_5_7b_lora_how2sign_full_v1/adapter_best"
SEED1_FULL_RUN="${SEED1_FULL_ROOT}/runs/qwen2_5_7b_lora_causal_action_full_corrected_v1"
SEED2_5K_RUN="${SEED2_5K_ROOT}/runs/qwen2_5_7b_lora_causal_action_5k_seed2_v1"

export HF_HOME="${HF_HOME:-/123090274/SN_Research/huggingface_cache}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${ROOT}/logs" "${ROOT}/reports" "${RUN}" "${RESULTS}" "${STABILITY_RESULTS}"

echo "[full-seed2] step 1/7: locked-data and exact-seed preflights"
"${PYTHON_BIN}" "${ROOT}/scripts/audit_full_experiment_preflight.py" \
  --full-root "${CAUSAL_ROOT}" \
  --control-5k-root "${CONTROL_DATA_ROOT}" \
  --output-report "${ROOT}/reports/full_experiment_preflight.json"

"${PYTHON_BIN}" "${ROOT}/scripts/audit_full_seed2_preflight.py" \
  --seed1-config "${SEED1_CONFIG}" \
  --seed2-config "${CONFIG}" \
  --causal-root "${CAUSAL_ROOT}" \
  --seed1-root "${SEED1_FULL_ROOT}" \
  --source-adapter "${SOURCE_ADAPTER}" \
  --output-report "${ROOT}/reports/full_seed2_preflight.json"

echo "[full-seed2] step 2/7: GPU smoke test"
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${ROOT}/scripts/smoke_test_causal_finetuning.py" \
  --config "${CONFIG}" \
  --manifest "${CAUSAL_ROOT}/manifests/train_actions.jsonl" \
  --source-adapter-dir "${SOURCE_ADAPTER}" \
  --device cuda:0

echo "[full-seed2] step 3/7: one-epoch four-GPU LoRA training from Fine-tune 1"
"${TORCHRUN}" --standalone --nproc_per_node=4 \
  "${ROOT}/scripts/train_qwen_lora_causal_ddp.py" \
  --config "${CONFIG}" \
  --train-manifest "${CAUSAL_ROOT}/manifests/train_actions.jsonl" \
  --validation-manifest "${CAUSAL_ROOT}/manifests/validation_actions.jsonl" \
  --test-manifest "${CAUSAL_ROOT}/manifests/test_actions.jsonl" \
  --source-adapter-dir "${SOURCE_ADAPTER}" \
  --output-dir "${RUN}"

echo "[full-seed2] step 4/7: parallel gold-state and free-running evaluation"
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${ROOT}/scripts/evaluate_causal_action_model.py" \
  --config "${CONFIG}" \
  --manifest "${CAUSAL_ROOT}/manifests/test_actions.jsonl" \
  --adapter-dir "${RUN}/adapter_best" \
  --output-dir "${RUN}/evaluation_full_seed2" \
  --stage finetune2_full_seed2_on_locked_test_gold_states \
  --device cuda:0 \
  --generation-batch-size 16 \
  > "${ROOT}/logs/evaluation_full_seed2.log" 2>&1 &
ACTION_PID=$!

CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" "${ROOT}/scripts/evaluate_free_running_rollout.py" \
  --config "${CONFIG}" \
  --parent-splits "${CAUSAL_ROOT}/manifests/parent_splits_full.jsonl" \
  --causal-pairs "${CAUSAL_ROOT}/outputs/causal_aligned_pairs_full.jsonl" \
  --adapter-dir "${RUN}/adapter_best" \
  --output-dir "${RUN}/rollout_full_seed2" \
  --stage finetune2_full_seed2_free_running_on_locked_test \
  --device cuda:0 \
  --generation-batch-size 16 \
  > "${ROOT}/logs/rollout_full_seed2.log" 2>&1 &
ROLLOUT_PID=$!

wait "${ACTION_PID}"
wait "${ROLLOUT_PID}"

echo "[full-seed2] step 5/7: paired Seed-2 5K-versus-full scale comparison"
"${PYTHON_BIN}" "${ROOT}/scripts/summarize_full_experiment.py" \
  --history "${RUN}/training_history.jsonl" \
  --training-summary "${RUN}/training_summary.json" \
  --data-report "${CAUSAL_ROOT}/reports/finalization_report.json" \
  --preflight-report "${ROOT}/reports/full_experiment_preflight.json" \
  --control-action "${SEED2_5K_RUN}/evaluation_seed2_5k/metrics_summary.json" \
  --full-action "${RUN}/evaluation_full_seed2/metrics_summary.json" \
  --control-rollouts "${SEED2_5K_RUN}/rollout_seed2_5k/sentence_rollouts.jsonl" \
  --full-rollouts "${RUN}/rollout_full_seed2/sentence_rollouts.jsonl" \
  --output-dir "${RESULTS}"

echo "[full-seed2] step 6/7: paired full Seed-1-versus-Seed-2 stability comparison"
"${PYTHON_BIN}" "${ROOT}/scripts/summarize_full_seed_stability.py" \
  --history "${RUN}/training_history.jsonl" \
  --training-summary "${RUN}/training_summary.json" \
  --preflight-report "${ROOT}/reports/full_seed2_preflight.json" \
  --seed1-action "${SEED1_FULL_RUN}/evaluation_full/metrics_summary.json" \
  --seed2-action "${RUN}/evaluation_full_seed2/metrics_summary.json" \
  --seed1-rollouts "${SEED1_FULL_RUN}/rollout_full/sentence_rollouts.jsonl" \
  --seed2-rollouts "${RUN}/rollout_full_seed2/sentence_rollouts.jsonl" \
  --output-dir "${STABILITY_RESULTS}"

echo "[full-seed2] step 7/7: structural validation and SHA-256 audit"
"${PYTHON_BIN}" "${ROOT}/scripts/validate_full_seed2_experiment.py" \
  --root "${ROOT}" \
  --causal-root "${CAUSAL_ROOT}" \
  --seed1-full-root "${SEED1_FULL_ROOT}" \
  --seed2-5k-root "${SEED2_5K_ROOT}" \
  --config "${CONFIG}" \
  --output-report "${ROOT}/reports/final_validation_report.json" \
  --hash-report "${ROOT}/reports/core_file_sha256.json"

echo "[full-seed2] completed"
echo "[full-seed2] scale result: ${RESULTS}/RESULTS.md"
echo "[full-seed2] stability result: ${STABILITY_RESULTS}/RESULTS.md"
