#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/123090274/SN_Research}"
DATA_ROOT="${PROJECT_ROOT}/data/how2sign_english_asl_gloss_v1"
FRONTEND_ROOT="${DATA_ROOT}/chinese_speech_frontend_v0"
TRAIN_ROOT="${DATA_ROOT}/full_v4_train"
SEMANTIC_ROOT="${TRAIN_ROOT}/fine_tuning_v2_semantic_v2_2_full"
AUDIT_ROOT="${TRAIN_ROOT}/causal_alignment_latency_audit_v2_2"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venvs/wlasl-cu128-env/bin/python}"
BASE_MODEL="${PROJECT_ROOT}/huggingface_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
FRONTEND_INPUT="${FRONTEND_ROOT}/outputs/task5_pilot_100.seamlessstreaming.jsonl"
DECISION="${AUDIT_ROOT}/reports/locked500_predeclared_model_decision.json"
V1_RUN="${TRAIN_ROOT}/fine_tuning_v2_causal_full_seed2/runs/qwen2_5_7b_lora_causal_action_full_seed2_v1"
V1_CONFIG="${TRAIN_ROOT}/fine_tuning_v2_causal_full_seed2/configs/qwen2_5_7b_lora_causal_action_full_seed2_v1.json"
PRIMARY_RUN="${SEMANTIC_ROOT}/runs/qwen2_5_7b_lora_semantic_causal_v2_2_1_treebank_full"
RECOVERY_RUN="${SEMANTIC_ROOT}/runs/qwen2_5_7b_lora_semantic_causal_v2_2_2_recovery_full"
SUMMARIZER="${AUDIT_ROOT}/scripts/summarize_v2_2_research.py"
REPRODUCIBILITY_WRITER="${AUDIT_ROOT}/scripts/write_reproducibility_manifest.py"
RESULT_PLOTTER="${AUDIT_ROOT}/scripts/plot_causal_alignment_results.py"

summarize_research() {
  "${PYTHON_BIN}" "${RESULT_PLOTTER}" \
    --train-root "${SEMANTIC_ROOT}" \
    --comparison-report "${AUDIT_ROOT}/reports/locked500_final_model_comparison.json" \
    --decision "${AUDIT_ROOT}/reports/locked500_predeclared_model_decision.json" \
    --output-png "${AUDIT_ROOT}/reports/FINAL_V2_2_RESULTS.png" \
    --output-pdf "${AUDIT_ROOT}/reports/FINAL_V2_2_RESULTS.pdf"
  "${PYTHON_BIN}" "${SUMMARIZER}" \
    --project-root "${PROJECT_ROOT}" \
    --output-json "${AUDIT_ROOT}/reports/FINAL_V2_2_RESEARCH_SUMMARY.json" \
    --output-markdown "${AUDIT_ROOT}/reports/FINAL_V2_2_RESEARCH_SUMMARY.md"
  "${PYTHON_BIN}" "${REPRODUCIBILITY_WRITER}" \
    --project-root "${PROJECT_ROOT}" \
    --output "${AUDIT_ROOT}/reports/FINAL_V2_2_REPRODUCIBILITY_MANIFEST.json"
}

export HF_HOME="${PROJECT_ROOT}/huggingface_cache"
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${FRONTEND_ROOT}/outputs" "${FRONTEND_ROOT}/reports" "${FRONTEND_ROOT}/logs"

echo "[causal-e2e] waiting for locked-500 model decision"
while [[ ! -s "${DECISION}" ]]; do
  sleep 60
done

echo "[causal-e2e] rerunning v1 with corrected WRITE-holds/WAIT-reads semantics"
V1_OUTPUT="${FRONTEND_ROOT}/outputs/task5_pilot_100.seamlessstreaming_ft2_v1_corrected_state_machine.jsonl"
V1_REPORT="${FRONTEND_ROOT}/reports/task5_pilot_100.seamlessstreaming_ft2_v1_corrected_state_machine.json"
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${FRONTEND_ROOT}/scripts/run_speech_to_gloss_e2e.py" \
  --frontend-jsonl "${FRONTEND_INPUT}" \
  --config "${V1_CONFIG}" \
  --base-model-path "${BASE_MODEL}" \
  --adapter-dir "${V1_RUN}/adapter_best" \
  --output-jsonl "${V1_OUTPUT}" \
  --output-report "${V1_REPORT}" \
  --device cuda:0 --overwrite

SELECTED=$("${PYTHON_BIN}" -c "import json; d=json.load(open('${DECISION}')); print(d.get('selected_candidate') or d.get('fallback') or 'v1')")
echo "[causal-e2e] locked-test selection: ${SELECTED}"

if [[ "${SELECTED}" == "v1" ]]; then
  "${PYTHON_BIN}" -c "import json; p='${FRONTEND_ROOT}/reports/task5_pilot_100.causal_e2e_selection.json'; json.dump({'selected_model':'v1','candidate_run_skipped':True,'reason':'No v2.2 candidate passed the predeclared locked-test criteria.'},open(p,'w'),indent=2); print(p)"
  summarize_research
  echo "[causal-e2e] v1 retained; no candidate endpoint run required"
  exit 0
fi

if [[ "${SELECTED}" == "v2_2_1" ]]; then
  RUN="${PRIMARY_RUN}"
  CONFIG="${SEMANTIC_ROOT}/configs/qwen2_5_7b_lora_semantic_causal_v2_2_1_full.json"
elif [[ "${SELECTED}" == "v2_2_2" ]]; then
  RUN="${RECOVERY_RUN}"
  CONFIG="${SEMANTIC_ROOT}/configs/qwen2_5_7b_lora_semantic_causal_v2_2_2_recovery_full.json"
else
  echo "[causal-e2e][error] unsupported selected model: ${SELECTED}" >&2
  exit 1
fi

CHECKPOINT=$("${PYTHON_BIN}" -c "import json; print(json.load(open('${RUN}/validation_checkpoint_selection.json'))['selected_candidate'])")
ADAPTER="${RUN}/adapter_${CHECKPOINT}"
CANDIDATE_OUTPUT="${FRONTEND_ROOT}/outputs/task5_pilot_100.seamlessstreaming_ft2_${SELECTED}_corrected_state_machine.jsonl"
CANDIDATE_REPORT="${FRONTEND_ROOT}/reports/task5_pilot_100.seamlessstreaming_ft2_${SELECTED}_corrected_state_machine.json"

echo "[causal-e2e] evaluating ${SELECTED}/${CHECKPOINT} on the locked speech pilot"
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${FRONTEND_ROOT}/scripts/run_speech_to_gloss_e2e.py" \
  --frontend-jsonl "${FRONTEND_INPUT}" \
  --config "${CONFIG}" \
  --base-model-path "${BASE_MODEL}" \
  --adapter-dir "${ADAPTER}" \
  --output-jsonl "${CANDIDATE_OUTPUT}" \
  --output-report "${CANDIDATE_REPORT}" \
  --device cuda:0 --overwrite

"${PYTHON_BIN}" "${FRONTEND_ROOT}/scripts/compare_causal_e2e_versions.py" \
  --baseline "${V1_OUTPUT}" \
  --candidate "${CANDIDATE_OUTPUT}" \
  --baseline-name v1 \
  --candidate-name "${SELECTED}" \
  --bootstrap-samples 5000 \
  --seed 20260818 \
  --output "${FRONTEND_ROOT}/reports/task5_pilot_100.causal_e2e_selection.json"

sha256sum "${V1_OUTPUT}" "${CANDIDATE_OUTPUT}" \
  > "${FRONTEND_ROOT}/reports/task5_pilot_100.causal_e2e_outputs.sha256"
summarize_research
echo "[causal-e2e] completed"
