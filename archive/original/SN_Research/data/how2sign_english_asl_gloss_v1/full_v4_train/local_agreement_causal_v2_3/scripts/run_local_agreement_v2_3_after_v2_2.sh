#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/123090274/SN_Research}"
DATA_ROOT="${PROJECT_ROOT}/data/how2sign_english_asl_gloss_v1/full_v4_train"
ROOT="${DATA_ROOT}/local_agreement_causal_v2_3"
CAUSAL_SCRIPTS="${DATA_ROOT}/fine_tuning_v2_semantic_v2_2_full/scripts"
AUDIT_ROOT="${DATA_ROOT}/causal_alignment_latency_audit_v2_2"
V1_ROOT="${DATA_ROOT}/causal_alignment_v1_corrected_full"
V1_TRAIN_ROOT="${DATA_ROOT}/fine_tuning_v2_causal_full"
FT1_ROOT="${DATA_ROOT}/fine_tuning_v1"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venvs/wlasl-cu128-env/bin/python}"
V2_FINAL_DECISION="${AUDIT_ROOT}/reports/locked500_predeclared_model_decision.json"
E2E_DECISION="${PROJECT_ROOT}/data/how2sign_english_asl_gloss_v1/chinese_speech_frontend_v0/reports/task5_pilot_100.causal_e2e_selection.json"
V1_LOCKED_ROLLOUT="${DATA_ROOT}/evaluation_v1_vs_semantic_v2_1_500/outputs/v1_500/sentence_rollouts.jsonl"

export HF_HOME="${PROJECT_ROOT}/huggingface_cache"
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${ROOT}/predictions" "${ROOT}/rollouts" "${ROOT}/reports" "${ROOT}/logs"

summarize() {
  "${PYTHON_BIN}" "${ROOT}/scripts/summarize_local_agreement_v2_3.py" \
    --root "${ROOT}" \
    --output-json "${ROOT}/reports/FINAL_V2_3_RESEARCH_SUMMARY.json" \
    --output-markdown "${ROOT}/reports/FINAL_V2_3_RESEARCH_SUMMARY.md"
}

echo "[local-agreement-v2.3] waiting until the predeclared v2.2 and speech-front-end evaluations release the GPU"
while [[ ! -s "${V2_FINAL_DECISION}" || ! -s "${E2E_DECISION}" ]]; do
  sleep 60
done

echo "[local-agreement-v2.3] measuring the original v1 model on validation"
V1_VALIDATION="${ROOT}/rollouts/v1_validation_baseline"
if [[ ! -s "${V1_VALIDATION}/rollout_metrics.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${CAUSAL_SCRIPTS}/evaluate_free_running_rollout.py" \
    --config "${V1_TRAIN_ROOT}/configs/qwen2_5_7b_lora_causal_action_full_v1.json" \
    --parent-splits "${V1_ROOT}/manifests/validation_pairs.jsonl" \
    --causal-pairs "${V1_ROOT}/outputs/causal_aligned_pairs_full.jsonl" \
    --adapter-dir "${V1_TRAIN_ROOT}/runs/qwen2_5_7b_lora_causal_action_full_corrected_v1/adapter_best" \
    --output-dir "${V1_VALIDATION}" \
    --stage v1_validation_baseline_for_v2_3 \
    --device cuda:0 --generation-batch-size 32
fi
MAX_LAAL=$("${PYTHON_BIN}" -c "import json; print(json.load(open('${V1_VALIDATION}/rollout_metrics.json'))['mean_length_adaptive_average_lagging_source_tokens'])")
echo "[local-agreement-v2.3] validation LAAL constraint inherited from v1 validation: ${MAX_LAAL}"

VALIDATION_PREDICTIONS="${ROOT}/predictions/finetune1_prefix_predictions_validation.jsonl"
if [[ ! -s "${ROOT}/reports/finetune1_prefix_predictions_validation.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${ROOT}/scripts/generate_finetune1_prefix_predictions.py" \
    --pairs "${V1_ROOT}/manifests/validation_pairs.jsonl" --split validation \
    --config "${FT1_ROOT}/configs/qwen2_5_7b_lora_full_v1.json" \
    --adapter-dir "${FT1_ROOT}/runs/qwen2_5_7b_lora_how2sign_full_v1/adapter_best" \
    --finetune1-scripts "${FT1_ROOT}/scripts" \
    --output "${VALIDATION_PREDICTIONS}" \
    --report "${ROOT}/reports/finetune1_prefix_predictions_validation.json" \
    --device cuda:0 --batch-size 16
fi

REPORT_ARGS=()
for WINDOW in 2 3 4; do
  for MIN_PREFIX in 1 2 3; do
    NAME="w${WINDOW}_m${MIN_PREFIX}"
    ROLLOUT="${ROOT}/rollouts/validation_${NAME}.jsonl"
    REPORT="${ROOT}/reports/validation_${NAME}.json"
    "${PYTHON_BIN}" "${ROOT}/scripts/evaluate_local_agreement_policy.py" \
      --predictions "${VALIDATION_PREDICTIONS}" \
      --causal-scripts "${CAUSAL_SCRIPTS}" \
      --stability-window "${WINDOW}" --minimum-prefix-tokens "${MIN_PREFIX}" \
      --output-rollouts "${ROLLOUT}" --output-report "${REPORT}"
    REPORT_ARGS+=("${REPORT}")
  done
done

"${PYTHON_BIN}" "${ROOT}/scripts/select_validation_local_agreement_policy.py" \
  --reports "${REPORT_ARGS[@]}" \
  --maximum-laal "${MAX_LAAL}" --maximum-conflict-ratio 0.20 \
  --output "${ROOT}/reports/validation_policy_selection.json"
SELECTED_WINDOW=$("${PYTHON_BIN}" -c "import json; print(json.load(open('${ROOT}/reports/validation_policy_selection.json'))['selected']['stability_window'])")
SELECTED_MIN=$("${PYTHON_BIN}" -c "import json; print(json.load(open('${ROOT}/reports/validation_policy_selection.json'))['selected']['minimum_prefix_tokens'])")
SELECTION_STATUS=$("${PYTHON_BIN}" -c "import json; print(json.load(open('${ROOT}/reports/validation_policy_selection.json'))['selection_status'])")

if [[ "${SELECTION_STATUS}" == no_candidate_passed* ]]; then
  echo "[local-agreement-v2.3] no policy passed validation constraints; test remains untouched"
  summarize
  exit 0
fi

echo "[local-agreement-v2.3] validation selected window=${SELECTED_WINDOW}, minimum-prefix=${SELECTED_MIN}"
TEST_PREDICTIONS="${ROOT}/predictions/finetune1_prefix_predictions_test.jsonl"
if [[ ! -s "${ROOT}/reports/finetune1_prefix_predictions_test.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${ROOT}/scripts/generate_finetune1_prefix_predictions.py" \
    --pairs "${V1_ROOT}/manifests/test_pairs.jsonl" --split test \
    --config "${FT1_ROOT}/configs/qwen2_5_7b_lora_full_v1.json" \
    --adapter-dir "${FT1_ROOT}/runs/qwen2_5_7b_lora_how2sign_full_v1/adapter_best" \
    --finetune1-scripts "${FT1_ROOT}/scripts" \
    --output "${TEST_PREDICTIONS}" \
    --report "${ROOT}/reports/finetune1_prefix_predictions_test.json" \
    --device cuda:0 --batch-size 16
fi

"${PYTHON_BIN}" "${ROOT}/scripts/evaluate_local_agreement_policy.py" \
  --predictions "${TEST_PREDICTIONS}" \
  --causal-scripts "${CAUSAL_SCRIPTS}" \
  --stability-window "${SELECTED_WINDOW}" --minimum-prefix-tokens "${SELECTED_MIN}" \
  --output-rollouts "${ROOT}/rollouts/test_selected.jsonl" \
  --output-report "${ROOT}/reports/test_selected.json"

"${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/compare_locked_rollouts.py" \
  --rollout v1="${V1_LOCKED_ROLLOUT}" \
  --rollout v2_3="${ROOT}/rollouts/test_selected.jsonl" \
  --baseline v1 --bootstrap-samples 5000 --seed 20260819 \
  --output-report "${ROOT}/reports/locked500_v1_vs_v2_3_comparison.json" \
  --output-paired-tsv "${ROOT}/reports/locked500_v1_vs_v2_3_paired.tsv"
"${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/select_latency_quality_model.py" \
  --comparison-report "${ROOT}/reports/locked500_v1_vs_v2_3_comparison.json" \
  --baseline v1 --candidate v2_3 \
  --output "${ROOT}/reports/locked500_v2_3_decision.json"

summarize
echo "[local-agreement-v2.3] completed"
