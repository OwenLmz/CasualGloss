#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data/how2sign_english_asl_gloss_v1/full_v4_train}"
ROOT="${ROOT:-${DATA_ROOT}/causal_alignment_v1_corrected_5k}"
PYTHON_BIN="${PYTHON_BIN:-/123090274/SN_Research/venvs/wlasl-cu128-env/bin/python}"
HF_HOME="${HF_HOME:-/123090274/SN_Research/huggingface_cache}"
export HF_HOME
export TOKENIZERS_PARALLELISM=false

INPUT="${DATA_ROOT}/fine_tuning_v1/manifests/train.jsonl"
CONFIG="${ROOT}/configs/causal_alignment_v1_corrected_5k.json"

mkdir -p "${ROOT}/manifests" "${ROOT}/shards/candidates" "${ROOT}/outputs"
mkdir -p "${ROOT}/reports/logs" "${ROOT}/previews" "${ROOT}/reviews"

echo "[5k] step 1/4: deterministic 6,200-parent candidate selection"
"${PYTHON_BIN}" "${ROOT}/scripts/prepare_candidates.py" \
  --input "${INPUT}" \
  --config "${CONFIG}" \
  --output "${ROOT}/manifests/candidate_source_6200.jsonl" \
  --shard-dir "${ROOT}/shards/candidates" \
  --report "${ROOT}/reports/candidate_selection_report.json"

echo "[5k] step 2/4: four-way GPU lexical/SimAlign causal alignment"
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
  echo "[5k] at least one alignment shard failed; inspect reports/logs" >&2
  exit 1
fi

echo "[5k] step 3/4: structural filtering and video-disjoint 4,000/500/500 split"
"${PYTHON_BIN}" "${ROOT}/scripts/finalize_dataset.py" \
  --shard-dir "${ROOT}/shards" \
  --config "${CONFIG}" \
  --output-dir "${ROOT}/outputs" \
  --manifest-dir "${ROOT}/manifests" \
  --preview "${ROOT}/previews/causal_alignment_preview_50.tsv" \
  --review "${ROOT}/reviews/causal_alignment_review_50.tsv" \
  --report "${ROOT}/reports/finalization_report.json"

echo "[5k] step 4/4: end-to-end validation and SHA-256 audit"
"${PYTHON_BIN}" "${ROOT}/scripts/validate_dataset.py" \
  --root "${ROOT}" \
  --config "${CONFIG}" \
  --report "${ROOT}/reports/validation_report.json" \
  --hash-report "${ROOT}/reports/core_file_sha256.json"

echo "[5k] completed successfully"
echo "[5k] final parents: ${ROOT}/outputs/causal_aligned_pairs_5000.jsonl"
echo "[5k] action manifests: ${ROOT}/manifests/{train,validation,test}_actions.jsonl"
echo "[5k] validation: ${ROOT}/reports/validation_report.json"
