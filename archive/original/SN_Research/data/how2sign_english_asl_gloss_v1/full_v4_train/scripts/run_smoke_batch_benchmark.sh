#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/123090274/SN_Research}"
D="$ROOT/data/how2sign_english_asl_gloss_v1/full_v4_train"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/venvs/wlasl-cu128-env/bin/python}"
HF_HOME="${HF_HOME:-$ROOT/huggingface_cache}"

mkdir -p "$D/outputs/smoke" "$D/previews/smoke" "$D/logs"

(
  start_seconds=$SECONDS
  CUDA_VISIBLE_DEVICES=0 HF_HOME="$HF_HOME" \
    "$PYTHON_BIN" "$D/scripts/generate_qwen_v4_pseudo_gloss.py" \
    --requests-jsonl "$D/manifests/shards/unique_requests.part000.jsonl" \
    --prompt-file "$D/prompts/asl_gloss_prompt_v4_deterministic_text_only.txt" \
    --output-jsonl "$D/outputs/smoke/predictions.limit32.batch8.jsonl" \
    --preview-tsv "$D/previews/smoke/predictions.limit32.batch8.tsv" \
    --device cuda:0 --batch-size 8 --max-new-tokens 192 --seed 20260720 \
    --limit 32 > "$D/logs/smoke_batch8.log" 2>&1
  status=$?
  echo $((SECONDS - start_seconds)) > "$D/logs/smoke_batch8.seconds"
  exit "$status"
) &
p0=$!

(
  start_seconds=$SECONDS
  CUDA_VISIBLE_DEVICES=1 HF_HOME="$HF_HOME" \
    "$PYTHON_BIN" "$D/scripts/generate_qwen_v4_pseudo_gloss.py" \
    --requests-jsonl "$D/manifests/shards/unique_requests.part000.jsonl" \
    --prompt-file "$D/prompts/asl_gloss_prompt_v4_deterministic_text_only.txt" \
    --output-jsonl "$D/outputs/smoke/predictions.limit32.batch16.jsonl" \
    --preview-tsv "$D/previews/smoke/predictions.limit32.batch16.tsv" \
    --device cuda:0 --batch-size 16 --max-new-tokens 192 --seed 20260720 \
    --limit 32 > "$D/logs/smoke_batch16.log" 2>&1
  status=$?
  echo $((SECONDS - start_seconds)) > "$D/logs/smoke_batch16.seconds"
  exit "$status"
) &
p1=$!

wait "$p0"
s0=$?
wait "$p1"
s1=$?

echo "batch8_status=$s0 batch16_status=$s1"
wc -l "$D"/outputs/smoke/*.jsonl 2>/dev/null || true
echo -n "batch8_seconds="
cat "$D/logs/smoke_batch8.seconds" 2>/dev/null || true
echo -n "batch16_seconds="
cat "$D/logs/smoke_batch16.seconds" 2>/dev/null || true

if [[ "$s0" -ne 0 || "$s1" -ne 0 ]]; then
  exit 1
fi
