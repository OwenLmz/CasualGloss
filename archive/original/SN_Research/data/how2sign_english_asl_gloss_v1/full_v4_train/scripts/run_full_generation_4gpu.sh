#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/123090274/SN_Research}"
DATA_ROOT="$ROOT/data/how2sign_english_asl_gloss_v1/full_v4_train"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/venvs/wlasl-cu128-env/bin/python}"
HF_HOME="${HF_HOME:-$ROOT/huggingface_cache}"
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
REVISION="a09a35458c702b33eeacc393d103063234e8bc28"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
SEED="${SEED:-20260720}"

mkdir -p \
  "$DATA_ROOT/outputs/shards" \
  "$DATA_ROOT/previews/shards" \
  "$DATA_ROOT/reports" \
  "$DATA_ROOT/reviews" \
  "$DATA_ROOT/logs"

pids=()
for shard_id in 0 1 2 3; do
  shard_name=$(printf "part%03d" "$shard_id")
  echo "[launch] GPU $shard_id <- $shard_name"
  CUDA_VISIBLE_DEVICES="$shard_id" HF_HOME="$HF_HOME" "$PYTHON_BIN" \
    "$DATA_ROOT/scripts/generate_qwen_v4_pseudo_gloss.py" \
    --requests-jsonl "$DATA_ROOT/manifests/shards/unique_requests.$shard_name.jsonl" \
    --prompt-file "$DATA_ROOT/prompts/asl_gloss_prompt_v4_deterministic_text_only.txt" \
    --output-jsonl "$DATA_ROOT/outputs/shards/predictions.$shard_name.jsonl" \
    --preview-tsv "$DATA_ROOT/previews/shards/predictions.$shard_name.tsv" \
    --model-id "$MODEL_ID" \
    --revision "$REVISION" \
    --device cuda:0 \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --seed "$SEED" \
    > "$DATA_ROOT/logs/generate.$shard_name.log" 2>&1 &
  pids+=("$!")
done

generation_status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "[complete] shard $index"
  else
    echo "[failed] shard $index" >&2
    generation_status=1
  fi
done
if [[ "$generation_status" -ne 0 ]]; then
  echo "At least one generation shard failed. Re-run this script to resume." >&2
  exit 1
fi

"$PYTHON_BIN" "$DATA_ROOT/scripts/merge_qwen_v4_full_shards.py" \
  --requests-jsonl "$DATA_ROOT/manifests/unique_generation_requests.jsonl" \
  --shard-output "$DATA_ROOT/outputs/shards/predictions.part000.jsonl" \
  --shard-output "$DATA_ROOT/outputs/shards/predictions.part001.jsonl" \
  --shard-output "$DATA_ROOT/outputs/shards/predictions.part002.jsonl" \
  --shard-output "$DATA_ROOT/outputs/shards/predictions.part003.jsonl" \
  --output-jsonl "$DATA_ROOT/outputs/how2sign_train_unique_pseudo_gloss_30102.qwen2_5_7b_v4.raw.jsonl" \
  --report-json "$DATA_ROOT/reports/merge_shards_report.json"

"$PYTHON_BIN" "$DATA_ROOT/scripts/repair_invalid_qwen_v4_outputs.py" \
  --input-jsonl "$DATA_ROOT/outputs/how2sign_train_unique_pseudo_gloss_30102.qwen2_5_7b_v4.raw.jsonl" \
  --output-jsonl "$DATA_ROOT/outputs/how2sign_train_unique_pseudo_gloss_30102.qwen2_5_7b_v4.final.jsonl" \
  --report-json "$DATA_ROOT/reports/invalid_output_repair_report.json"

"$PYTHON_BIN" "$DATA_ROOT/scripts/expand_qwen_v4_full_records.py" \
  --all-records-jsonl "$DATA_ROOT/manifests/how2sign_train_all_records_31165.jsonl" \
  --unique-predictions-jsonl "$DATA_ROOT/outputs/how2sign_train_unique_pseudo_gloss_30102.qwen2_5_7b_v4.final.jsonl" \
  --output-jsonl "$DATA_ROOT/outputs/how2sign_train_all_records_31165.pseudo_gloss.qwen2_5_7b_v4.final.jsonl" \
  --report-json "$DATA_ROOT/reports/expand_all_records_report.json"

"$PYTHON_BIN" "$DATA_ROOT/scripts/validate_qwen_v4_full_dataset.py" \
  --all-records-jsonl "$DATA_ROOT/manifests/how2sign_train_all_records_31165.jsonl" \
  --unique-requests-jsonl "$DATA_ROOT/manifests/unique_generation_requests.jsonl" \
  --unique-predictions-jsonl "$DATA_ROOT/outputs/how2sign_train_unique_pseudo_gloss_30102.qwen2_5_7b_v4.final.jsonl" \
  --expanded-predictions-jsonl "$DATA_ROOT/outputs/how2sign_train_all_records_31165.pseudo_gloss.qwen2_5_7b_v4.final.jsonl" \
  --report-json "$DATA_ROOT/reports/full_dataset_validation_report.json" \
  --audit-jsonl "$DATA_ROOT/reviews/full_unique_pseudo_gloss_automatic_audit.jsonl" \
  --review-queue-tsv "$DATA_ROOT/reviews/full_unique_pseudo_gloss_review_queue_200.tsv" \
  --expected-all-count 31165 \
  --expected-unique-count 30102 \
  --expected-revision "$REVISION" \
  --expected-source-sha256 67a8f4fb6dc066f38006b89ac84c27a4b05d7fa07d5e6952f997042724d7a847 \
  --review-queue-size 200

"$PYTHON_BIN" "$DATA_ROOT/scripts/summarize_full_dataset.py" \
  --preparation-report "$DATA_ROOT/reports/preparation_report.json" \
  --repair-report "$DATA_ROOT/reports/invalid_output_repair_report.json" \
  --validation-report "$DATA_ROOT/reports/full_dataset_validation_report.json" \
  --results-md "$DATA_ROOT/RESULTS.md" \
  --professor-update-md "$DATA_ROOT/PROFESSOR_UPDATE.md"

echo "[done] Full How2Sign train pseudo-gloss pipeline completed."
