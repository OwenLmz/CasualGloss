#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/123090274/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train}"
ROOT="${ROOT:-${DATA_ROOT}/causal_alignment_v1_corrected_full}"
PYTHON_BIN="${PYTHON_BIN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/python}"
HF_HOME="${HF_HOME:-/123090274/SN_Research/huggingface_cache}"
export HF_HOME
export TOKENIZERS_PARALLELISM=false

INPUT="${DATA_ROOT}/fine_tuning_v1/manifests/train.jsonl"
LOCKED="${DATA_ROOT}/causal_alignment_v1_corrected_5k/manifests/parent_splits_5000.jsonl"
CONFIG="${ROOT}/configs/causal_alignment_v1_corrected_full.json"

mkdir -p "${ROOT}/manifests" "${ROOT}/shards/candidates" "${ROOT}/outputs"
mkdir -p "${ROOT}/reports/logs" "${ROOT}/previews" "${ROOT}/reviews"

echo "[full] step 1/5: prepare all deterministic eligible parents and verify locked IDs"
"${PYTHON_BIN}" "${ROOT}/scripts/prepare_full_candidates.py" \
  --input "${INPUT}" \
  --locked-parent-splits "${LOCKED}" \
  --config "${CONFIG}" \
  --output "${ROOT}/manifests/candidate_source_full.jsonl" \
  --shard-dir "${ROOT}/shards/candidates" \
  --report "${ROOT}/reports/candidate_preparation_report.json"

echo "[full] step 2/5: four-GPU lexical/SimAlign causal alignment"
pids=()
for shard in 0 1 2 3; do
  shard_name="$(printf '%02d' "${shard}")"
  CUDA_VISIBLE_DEVICES="${shard}" "${PYTHON_BIN}" \
    "${ROOT}/scripts/build_causal_alignment_shard.py" \
    --input "${ROOT}/shards/candidates/candidate_shard_${shard_name}.jsonl" \
    --config "${CONFIG}" \
    --output-pairs "${ROOT}/shards/pairs_shard_${shard_name}.jsonl" \
    --output-actions "${ROOT}/shards/actions_shard_${shard_name}.jsonl" \
    --report "${ROOT}/reports/alignment_shard_${shard_name}_report.json" \
    > "${ROOT}/reports/logs/alignment_shard_${shard_name}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "[full] at least one alignment shard failed; inspect reports/logs" >&2
  exit 1
fi

echo "[full] step 3/5: filter alignments, lock evaluation parents, and quarantine held-out videos"
"${PYTHON_BIN}" "${ROOT}/scripts/finalize_full_dataset.py" \
  --shard-dir "${ROOT}/shards" \
  --locked-parent-splits "${LOCKED}" \
  --config "${CONFIG}" \
  --output-dir "${ROOT}/outputs" \
  --manifest-dir "${ROOT}/manifests" \
  --preview "${ROOT}/previews/causal_alignment_preview_100.tsv" \
  --review "${ROOT}/reviews/causal_alignment_review_100.tsv" \
  --report "${ROOT}/reports/finalization_report.json"

echo "[full] step 4/5: write concise research summaries"
"${PYTHON_BIN}" "${ROOT}/scripts/summarize_full_alignment.py" \
  --preparation-report "${ROOT}/reports/candidate_preparation_report.json" \
  --finalization-report "${ROOT}/reports/finalization_report.json" \
  --results "${ROOT}/RESULTS.md" \
  --professor-update "${ROOT}/PROFESSOR_UPDATE.md"

echo "[full] step 5/5: end-to-end validation and SHA-256 audit"
"${PYTHON_BIN}" "${ROOT}/scripts/validate_full_dataset.py" \
  --root "${ROOT}" \
  --locked-parent-splits "${LOCKED}" \
  --config "${CONFIG}" \
  --report "${ROOT}/reports/validation_report.json" \
  --hash-report "${ROOT}/reports/core_file_sha256.json"

echo "[full] completed successfully"
echo "[full] parents: ${ROOT}/outputs/causal_aligned_pairs_full.jsonl"
echo "[full] actions: ${ROOT}/outputs/prefix_action_examples_corrected_full.jsonl"
echo "[full] validation: ${ROOT}/reports/validation_report.json"
