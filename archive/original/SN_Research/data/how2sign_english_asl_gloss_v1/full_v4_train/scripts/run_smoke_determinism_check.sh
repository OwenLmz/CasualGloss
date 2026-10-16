#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/123090274/SN_Research}"
D="$ROOT/data/how2sign_english_asl_gloss_v1/full_v4_train"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/venvs/wlasl-cu128-env/bin/python}"
HF_HOME="${HF_HOME:-$ROOT/huggingface_cache}"

mkdir -p "$D/outputs/smoke" "$D/previews/smoke" "$D/logs"

for gpu_id in 0 1; do
  run_name="batch4_repeat${gpu_id}"
  CUDA_VISIBLE_DEVICES="$gpu_id" HF_HOME="$HF_HOME" \
    "$PYTHON_BIN" "$D/scripts/generate_qwen_v4_pseudo_gloss.py" \
    --requests-jsonl "$D/manifests/shards/unique_requests.part000.jsonl" \
    --prompt-file "$D/prompts/asl_gloss_prompt_v4_deterministic_text_only.txt" \
    --output-jsonl "$D/outputs/smoke/predictions.limit32.$run_name.jsonl" \
    --preview-tsv "$D/previews/smoke/predictions.limit32.$run_name.tsv" \
    --device cuda:0 --batch-size 4 --max-new-tokens 192 --seed 20260720 \
    --limit 32 > "$D/logs/smoke_$run_name.log" 2>&1 &
  pids[$gpu_id]=$!
done

status=0
for gpu_id in 0 1; do
  if wait "${pids[$gpu_id]}"; then
    echo "[complete] repeat $gpu_id"
  else
    echo "[failed] repeat $gpu_id" >&2
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  exit 1
fi

"$PYTHON_BIN" "$D/scripts/validate_smoke_batch_consistency.py" \
  --first "$D/outputs/smoke/predictions.limit32.batch4_repeat0.jsonl" \
  --second "$D/outputs/smoke/predictions.limit32.batch4_repeat1.jsonl" \
  --report-json "$D/reports/smoke_batch4_repeatability_report.json"
