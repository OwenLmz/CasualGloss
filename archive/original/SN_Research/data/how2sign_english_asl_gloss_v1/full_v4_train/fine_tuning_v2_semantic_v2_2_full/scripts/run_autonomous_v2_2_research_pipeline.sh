#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/123090274/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train}"
ROOT="${DATA_ROOT}/fine_tuning_v2_semantic_v2_2_full"
CAUSAL_ROOT="${DATA_ROOT}/semantic_causal_alignment_v2_2_full"
AUDIT_ROOT="${DATA_ROOT}/causal_alignment_latency_audit_v2_2"
PYTHON_BIN="${PYTHON_BIN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/python}"
PRIMARY_RUN="${ROOT}/runs/qwen2_5_7b_lora_semantic_causal_v2_2_1_treebank_full"
RECOVERY_RUN="${ROOT}/runs/qwen2_5_7b_lora_semantic_causal_v2_2_2_recovery_full"
MANIFESTS="${CAUSAL_ROOT}/manifests_treebank_v2_2_1"
LOCKED_V1_ROOT="${DATA_ROOT}/causal_alignment_v1_corrected_full"
V1_ROLLOUT="${DATA_ROOT}/evaluation_v1_vs_semantic_v2_1_500/outputs/v1_500/sentence_rollouts.jsonl"
V2_1_ROLLOUT="${DATA_ROOT}/evaluation_v1_vs_semantic_v2_1_500/outputs/v2_1_500/sentence_rollouts.jsonl"
REPORTS="${AUDIT_ROOT}/reports"

export HF_HOME="${HF_HOME:-/123090274/SN_Research/huggingface_cache}"
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${REPORTS}" "${ROOT}/logs"

echo "[autonomous-v2.2] waiting for primary v2.2.1 training and built-in evaluation"
while [[ ! -s "${PRIMARY_RUN}/rollout_locked_500/rollout_metrics.json" ]]; do
  if ! pgrep -f '[r]un_finetune2_semantic_v2_2_1_full.sh' >/dev/null; then
    echo "[autonomous-v2.2][error] primary process exited before producing rollout metrics" >&2
    exit 1
  fi
  sleep 60
done

echo "[autonomous-v2.2] validation-only selection between primary best/final checkpoints"
for NAME in best final; do
  if [[ "${NAME}" == "best" ]]; then ADAPTER="${PRIMARY_RUN}/adapter_best"; else ADAPTER="${PRIMARY_RUN}/adapter_final"; fi
  if [[ ! -s "${PRIMARY_RUN}/rollout_validation_${NAME}/rollout_metrics.json" ]]; then
    CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${ROOT}/scripts/evaluate_free_running_rollout.py" \
      --config "${ROOT}/configs/qwen2_5_7b_lora_semantic_causal_v2_2_1_full.json" \
      --parent-splits "${MANIFESTS}/validation_pairs.jsonl" \
      --causal-pairs "${CAUSAL_ROOT}/outputs/semantic_causal_aligned_pairs_v2_2_full.jsonl" \
      --adapter-dir "${ADAPTER}" \
      --output-dir "${PRIMARY_RUN}/rollout_validation_${NAME}" \
      --stage "semantic_v2_2_1_validation_${NAME}" \
      --device cuda:0 --generation-batch-size 32
  fi
done

"${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/select_validation_rollout_checkpoint.py" \
  --rollout best="${PRIMARY_RUN}/rollout_validation_best/sentence_rollouts.jsonl" \
  --rollout final="${PRIMARY_RUN}/rollout_validation_final/sentence_rollouts.jsonl" \
  --output "${PRIMARY_RUN}/validation_checkpoint_selection.json"
PRIMARY_SELECTED=$("${PYTHON_BIN}" -c "import json; print(json.load(open('${PRIMARY_RUN}/validation_checkpoint_selection.json'))['selected_candidate'])")
if [[ "${PRIMARY_SELECTED}" == "final" ]]; then PRIMARY_ADAPTER="${PRIMARY_RUN}/adapter_final"; else PRIMARY_ADAPTER="${PRIMARY_RUN}/adapter_best"; fi

if [[ ! -s "${PRIMARY_RUN}/rollout_locked_500_selected/rollout_metrics.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${ROOT}/scripts/evaluate_free_running_rollout.py" \
    --config "${ROOT}/configs/qwen2_5_7b_lora_semantic_causal_v2_2_1_full.json" \
    --parent-splits "${LOCKED_V1_ROOT}/manifests/test_pairs.jsonl" \
    --causal-pairs "${LOCKED_V1_ROOT}/outputs/causal_aligned_pairs_full.jsonl" \
    --adapter-dir "${PRIMARY_ADAPTER}" \
    --output-dir "${PRIMARY_RUN}/rollout_locked_500_selected" \
    --stage semantic_v2_2_1_selected_free_running_locked_500 \
    --device cuda:0 --generation-batch-size 32
fi

echo "[autonomous-v2.2] primary paired and qualitative audit"
"${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/compare_locked_rollouts.py" \
  --rollout v1="${V1_ROLLOUT}" --rollout v2_1="${V2_1_ROLLOUT}" \
  --rollout v2_2_1="${PRIMARY_RUN}/rollout_locked_500_selected/sentence_rollouts.jsonl" \
  --baseline v1 --bootstrap-samples 5000 --seed 20260817 \
  --output-report "${REPORTS}/locked500_v1_v2_1_v2_2_1_comparison.json" \
  --output-paired-tsv "${REPORTS}/locked500_v1_v2_1_v2_2_1_paired.tsv"
"${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/summarize_paired_latency_cases.py" \
  --baseline-rollout "${V1_ROLLOUT}" \
  --candidate-rollout "${PRIMARY_RUN}/rollout_locked_500_selected/sentence_rollouts.jsonl" \
  --candidate-name v2_2_1 --top-k 20 \
  --output-report "${REPORTS}/v2_2_1_vs_v1_qualitative_cases.json" \
  --output-tsv "${REPORTS}/v2_2_1_vs_v1_qualitative_cases.tsv"

echo "[autonomous-v2.2] running separate recovery-calibration ablation"
if [[ ! -s "${RECOVERY_RUN}/rollout_locked_500_selected/rollout_metrics.json" ]]; then
  bash "${ROOT}/scripts/run_finetune2_semantic_v2_2_2_recovery_full.sh"
fi

echo "[autonomous-v2.2] selecting the v2.2 variant on validation only"
RECOVERY_SELECTED=$("${PYTHON_BIN}" -c "import json; print(json.load(open('${RECOVERY_RUN}/validation_checkpoint_selection.json'))['selected_candidate'])")
"${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/compare_locked_rollouts.py" \
  --rollout v2_2_1="${PRIMARY_RUN}/rollout_validation_${PRIMARY_SELECTED}/sentence_rollouts.jsonl" \
  --rollout v2_2_2="${RECOVERY_RUN}/rollout_validation_${RECOVERY_SELECTED}/sentence_rollouts.jsonl" \
  --baseline v2_2_1 --bootstrap-samples 5000 --seed 20260817 \
  --output-report "${REPORTS}/validation_v2_2_1_vs_v2_2_2_comparison.json" \
  --output-paired-tsv "${REPORTS}/validation_v2_2_1_vs_v2_2_2_paired.tsv"
"${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/select_latency_quality_model.py" \
  --comparison-report "${REPORTS}/validation_v2_2_1_vs_v2_2_2_comparison.json" \
  --baseline v2_2_1 --candidate v2_2_2 \
  --output "${REPORTS}/validation_v2_2_model_decision.json"
FINAL_V2_CANDIDATE=$("${PYTHON_BIN}" -c "import json; d=json.load(open('${REPORTS}/validation_v2_2_model_decision.json')); print(d.get('selected_candidate') or d.get('fallback'))")

echo "[autonomous-v2.2] final locked comparison and confirmatory test"
"${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/compare_locked_rollouts.py" \
  --rollout v1="${V1_ROLLOUT}" --rollout v2_1="${V2_1_ROLLOUT}" \
  --rollout v2_2_1="${PRIMARY_RUN}/rollout_locked_500_selected/sentence_rollouts.jsonl" \
  --rollout v2_2_2="${RECOVERY_RUN}/rollout_locked_500_selected/sentence_rollouts.jsonl" \
  --baseline v1 --bootstrap-samples 5000 --seed 20260817 \
  --output-report "${REPORTS}/locked500_final_model_comparison.json" \
  --output-paired-tsv "${REPORTS}/locked500_final_model_comparison_paired.tsv"
"${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/select_latency_quality_model.py" \
  --comparison-report "${REPORTS}/locked500_final_model_comparison.json" \
  --baseline v1 --candidate "${FINAL_V2_CANDIDATE}" \
  --output "${REPORTS}/locked500_predeclared_model_decision.json"
if [[ "${FINAL_V2_CANDIDATE}" == "v2_2_2" ]]; then
  FINAL_V2_ROLLOUT="${RECOVERY_RUN}/rollout_locked_500_selected/sentence_rollouts.jsonl"
else
  FINAL_V2_ROLLOUT="${PRIMARY_RUN}/rollout_locked_500_selected/sentence_rollouts.jsonl"
fi
"${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/analyze_policy_imitation_gap.py" \
  --label-pairs "${CAUSAL_ROOT}/outputs/semantic_causal_aligned_pairs_v2_2_full.jsonl" \
  --model-rollout "${FINAL_V2_ROLLOUT}" \
  --model-name "${FINAL_V2_CANDIDATE}" --split test \
  --bootstrap-samples 5000 --seed 20260820 \
  --output-report "${REPORTS}/final_v2_2_policy_imitation_gap.json" \
  --output-tsv "${REPORTS}/final_v2_2_policy_imitation_gap_cases.tsv"
"${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/summarize_paired_latency_cases.py" \
  --baseline-rollout "${V1_ROLLOUT}" \
  --candidate-rollout "${RECOVERY_RUN}/rollout_locked_500_selected/sentence_rollouts.jsonl" \
  --candidate-name v2_2_2 --top-k 20 \
  --output-report "${REPORTS}/v2_2_2_vs_v1_qualitative_cases.json" \
  --output-tsv "${REPORTS}/v2_2_2_vs_v1_qualitative_cases.tsv"

echo "[autonomous-v2.2] completed"
