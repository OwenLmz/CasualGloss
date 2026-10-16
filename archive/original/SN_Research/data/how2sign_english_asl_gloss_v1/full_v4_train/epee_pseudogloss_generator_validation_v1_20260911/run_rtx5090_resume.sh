#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/123090274/SN_Research}"
FULL_ROOT="$ROOT/data/how2sign_english_asl_gloss_v1/full_v4_train"
EPEE_ROOT="$FULL_ROOT/clerc_epee_external_eval_v1"
EXP_ROOT="$FULL_ROOT/epee_pseudogloss_generator_validation_v1_20260911"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/venvs/wlasl-cu128-env/bin/python}"
HF_HOME="${HF_HOME:-$ROOT/huggingface_cache}"
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
REVISION="a09a35458c702b33eeacc393d103063234e8bc28"
PROMPT="$EXP_ROOT/configs/asl_gloss_prompt_v4_deterministic_text_only.txt"
METRICS_RUNNER="$EXP_ROOT/run_with_metrics_deps.py"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for required in \
  "$EXP_ROOT/manifests/epee_generator_requests_401.jsonl" \
  "$FULL_ROOT/manifests/unique_generation_requests.jsonl" \
  "$FULL_ROOT/outputs/how2sign_train_unique_pseudo_gloss_30102.qwen2_5_7b_v4.raw.jsonl" \
  "$PROMPT"; do
  [[ -s "$required" ]] || { echo "Missing required checkpoint: $required" >&2; exit 1; }
done

"$PYTHON_BIN" "$EXP_ROOT/make_original_shard_smoke.py" \
  --original-requests "$FULL_ROOT/manifests/unique_generation_requests.jsonl" \
  --original-predictions "$FULL_ROOT/outputs/how2sign_train_unique_pseudo_gloss_30102.qwen2_5_7b_v4.raw.jsonl" \
  --smoke-requests "$EXP_ROOT/smoke/how2sign_original_shard0_requests_20.jsonl" \
  --smoke-reference "$EXP_ROOT/smoke/how2sign_original_shard0_reference_bf16_20.jsonl" \
  --shard-id 0 --count 20 --batch-size 4

CUDA_VISIBLE_DEVICES=0 HF_HOME="$HF_HOME" "$PYTHON_BIN" \
  "$EXP_ROOT/generate_qwen_v4_pseudo_gloss_portable.py" \
  --requests-jsonl "$EXP_ROOT/smoke/how2sign_original_shard0_requests_20.jsonl" \
  --prompt-file "$PROMPT" \
  --output-jsonl "$EXP_ROOT/smoke/how2sign_original_shard0_rtx5090_bf16_20.jsonl" \
  --preview-tsv "$EXP_ROOT/smoke/how2sign_original_shard0_rtx5090_bf16_20.tsv" \
  --model-id "$MODEL_ID" --revision "$REVISION" --device cuda:0 \
  --batch-size 4 --max-new-tokens 192 --seed 20260720 --dtype bfloat16 \
  > "$EXP_ROOT/logs/smoke_generation.rtx5090.original_shard0.log" 2>&1

"$PYTHON_BIN" "$EXP_ROOT/compare_smoke_reproducibility.py" \
  --reference "$EXP_ROOT/smoke/how2sign_original_shard0_reference_bf16_20.jsonl" \
  --replication "$EXP_ROOT/smoke/how2sign_original_shard0_rtx5090_bf16_20.jsonl" \
  --report-json "$EXP_ROOT/reports/portability_gate_rtx5090_original_shard0.json" \
  --reference-execution "original RTX5090 BF16 batch4, generation shard 0" \
  --replication-execution "RTX5090 BF16 batch4, generation shard 0 batch composition"

for shard_id in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 HF_HOME="$HF_HOME" "$PYTHON_BIN" \
    "$EXP_ROOT/generate_qwen_v4_pseudo_gloss_portable.py" \
    --requests-jsonl "$EXP_ROOT/manifests/shards/epee_requests.shard${shard_id}.jsonl" \
    --prompt-file "$PROMPT" \
    --output-jsonl "$EXP_ROOT/outputs/shards/epee_predictions.shard${shard_id}.jsonl" \
    --preview-tsv "$EXP_ROOT/previews/shards/epee_predictions.shard${shard_id}.tsv" \
    --model-id "$MODEL_ID" --revision "$REVISION" --device cuda:0 \
    --batch-size 4 --max-new-tokens 192 --seed 20260720 --dtype bfloat16 \
    > "$EXP_ROOT/logs/generate.shard${shard_id}.rtx5090.log" 2>&1
done

"$PYTHON_BIN" "$EXP_ROOT/merge_predictions.py" \
  --requests "$EXP_ROOT/manifests/epee_generator_requests_401.jsonl" \
  --shard "$EXP_ROOT/outputs/shards/epee_predictions.shard0.jsonl" \
  --shard "$EXP_ROOT/outputs/shards/epee_predictions.shard1.jsonl" \
  --shard "$EXP_ROOT/outputs/shards/epee_predictions.shard2.jsonl" \
  --shard "$EXP_ROOT/outputs/shards/epee_predictions.shard3.jsonl" \
  --output "$EXP_ROOT/outputs/epee_qwen2_5_v4_predictions_401.jsonl" \
  --report-json "$EXP_ROOT/reports/merge_report.json"

"$PYTHON_BIN" "$METRICS_RUNNER" "$EXP_ROOT/evaluate_generator_against_epee.py" \
  --manifest "$EPEE_ROOT/manifests/epee_sources_401.jsonl" \
  --predictions "$EXP_ROOT/outputs/epee_qwen2_5_v4_predictions_401.jsonl" \
  --output-dir "$EXP_ROOT/reports/direct_human_agreement" \
  --bootstrap-repetitions 10000 --seed 20260911 \
  > "$EXP_ROOT/logs/evaluation.log" 2>&1

printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$EXP_ROOT/reports/EXPERIMENT_COMPLETE"
echo "Epee pseudo-gloss generator validation completed: $EXP_ROOT"
