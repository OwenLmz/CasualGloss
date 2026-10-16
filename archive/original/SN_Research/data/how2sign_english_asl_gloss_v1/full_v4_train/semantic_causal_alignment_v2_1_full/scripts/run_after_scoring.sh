#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/123090274/SN_Research}"
DATA_ROOT="${PROJECT_ROOT}/data/how2sign_english_asl_gloss_v1/full_v4_train"
ROOT="${DATA_ROOT}/semantic_causal_alignment_v2_1_full"
PILOT_ROOT="${DATA_ROOT}/semantic_causal_alignment_v2_1_pilot500"
V2_ROOT="${DATA_ROOT}/semantic_causal_alignment_v2_pilot500"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venvs/wlasl-cu128-env/bin/python}"
CONFIG="${ROOT}/configs/semantic_causal_alignment_v2_1_full.json"

echo "[v2.1-full] waiting for both score reports"
while [[ ! -s "${ROOT}/reports/prefix_semantic_scores_shard_00_report.json" || ! -s "${ROOT}/reports/prefix_semantic_scores_shard_01_report.json" ]]; do
  sleep 30
done

"${PYTHON_BIN}" -c 'import json,sys; paths=sys.argv[1:]; reports=[json.load(open(p)) for p in paths]; assert all(r.get("ok") for r in reports), reports; print({"score_reports_ok": True, "completed": [r["completed_parent_count"] for r in reports]})' \
  "${ROOT}/reports/prefix_semantic_scores_shard_00_report.json" \
  "${ROOT}/reports/prefix_semantic_scores_shard_01_report.json"

if [[ -s "${ROOT}/shards/parents_delta_after_contraction_fix.jsonl" ]]; then
  echo "[v2.1-full] scoring reconciled four-parent delta"
  CUDA_VISIBLE_DEVICES=0 HF_HOME="${PROJECT_ROOT}/huggingface_cache" "${PYTHON_BIN}" \
    "${V2_ROOT}/scripts/score_prefix_semantics.py" \
    --input "${ROOT}/shards/parents_delta_after_contraction_fix.jsonl" \
    --config "${CONFIG}" \
    --output "${ROOT}/shards/prefix_semantic_scores_delta_after_contraction_fix.jsonl" \
    --report "${ROOT}/reports/prefix_semantic_scores_delta_after_contraction_fix_report.json" \
    --device cuda:0
fi

echo "[v2.1-full] finalizing semantic alignments and causal actions"
"${PYTHON_BIN}" "${PILOT_ROOT}/scripts/finalize_semantic_alignment_v2_1.py" \
  --parents "${ROOT}/manifests/parents_all_reconciled.jsonl" \
  --scores "${ROOT}/shards/prefix_semantic_scores_shard_00.jsonl" "${ROOT}/shards/prefix_semantic_scores_shard_01.jsonl" "${ROOT}/shards/prefix_semantic_scores_delta_after_contraction_fix.jsonl" \
  --config "${CONFIG}" \
  --output-pairs "${ROOT}/outputs/semantic_causal_aligned_pairs_v2_1_full.jsonl" \
  --output-actions "${ROOT}/manifests/semantic_actions_v2_1_full.jsonl" \
  --summary "${ROOT}/reports/semantic_alignment_v2_1_full_summary.json" \
  --preview "${ROOT}/previews/v2_1_alignment_all_full.tsv"

echo "[v2.1-full] applying predeclared full-data quality gate"
"${PYTHON_BIN}" "${PILOT_ROOT}/scripts/quality_gate_v2_1.py" \
  --pairs "${ROOT}/outputs/semantic_causal_aligned_pairs_v2_1_full.jsonl" \
  --actions "${ROOT}/manifests/semantic_actions_v2_1_full.jsonl" \
  --summary "${ROOT}/reports/semantic_alignment_v2_1_full_summary.json" \
  --config "${CONFIG}" \
  --stress-audit "${V2_ROOT}/reviews/semantic_alignment_technical_audit_50.jsonl" \
  --output-report "${ROOT}/reports/quality_gate_v2_1_full.json" \
  --stress-preview "${ROOT}/previews/stress50_v2_1_full.tsv" \
  --random-preview "${ROOT}/previews/random50_v2_1_full.tsv"

echo "[v2.1-full] exporting locked train/validation/test manifests"
"${PYTHON_BIN}" "${ROOT}/scripts/split_full_outputs.py" \
  --pairs "${ROOT}/outputs/semantic_causal_aligned_pairs_v2_1_full.jsonl" \
  --actions "${ROOT}/manifests/semantic_actions_v2_1_full.jsonl" \
  --output-root "${ROOT}"

echo "[v2.1-full] starting Full Fine-tune 2"
bash "${DATA_ROOT}/fine_tuning_v2_semantic_v2_1_full/scripts/run_finetune2_semantic_v2_1_full.sh"
echo "[v2.1-full] entire pipeline completed"
