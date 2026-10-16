#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/123090274/SN_Research}"
PYTHON="${PYTHON:-$ROOT/venvs/wlasl-cu128-env/bin/python}"
EXP_REL="data/how2sign_english_asl_gloss_v1/full_v4_train/semantic_causal_alignment_v2_pilot500"
EXP="$ROOT/$EXP_REL"
CONFIG="$EXP/configs/semantic_causal_alignment_v2_pilot500.json"

cd "$ROOT"
export HF_HOME="${HF_HOME:-$ROOT/huggingface_cache}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$EXP"/{manifests,outputs,reports,previews,reviews,logs,shards}

echo "[semantic-v2] step 1/4: deterministic stratified pilot selection"
"$PYTHON" "$EXP/scripts/prepare_pilot.py" \
  --config "$CONFIG" \
  --output "$EXP/manifests/pilot_parents_500.jsonl" \
  --shard-dir "$EXP/shards" \
  --report "$EXP/reports/pilot_selection_report.json"

echo "[semantic-v2] step 2/4: Fine-tune 1 prefix semantic scoring on two GPUs"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$EXP/scripts/score_prefix_semantics.py" \
  --input "$EXP/shards/pilot_parents_shard_00.jsonl" \
  --config "$CONFIG" \
  --output "$EXP/shards/prefix_semantic_scores_shard_00.jsonl" \
  --report "$EXP/reports/prefix_semantic_scores_shard_00_report.json" \
  --device cuda:0 > "$EXP/logs/score_shard_00.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$EXP/scripts/score_prefix_semantics.py" \
  --input "$EXP/shards/pilot_parents_shard_01.jsonl" \
  --config "$CONFIG" \
  --output "$EXP/shards/prefix_semantic_scores_shard_01.jsonl" \
  --report "$EXP/reports/prefix_semantic_scores_shard_01_report.json" \
  --device cuda:0 > "$EXP/logs/score_shard_01.log" 2>&1 &
PID1=$!

wait "$PID0"
wait "$PID1"

echo "[semantic-v2] step 3/4: monotonic DP fusion and causal trajectory construction"
"$PYTHON" "$EXP/scripts/finalize_semantic_alignment.py" \
  --parents "$EXP/manifests/pilot_parents_500.jsonl" \
  --scores "$EXP/shards/prefix_semantic_scores_shard_00.jsonl" "$EXP/shards/prefix_semantic_scores_shard_01.jsonl" \
  --config "$CONFIG" \
  --output-pairs "$EXP/outputs/semantic_causal_aligned_pairs_v2_pilot500.jsonl" \
  --output-actions "$EXP/manifests/semantic_actions_v2_pilot500.jsonl" \
  --summary "$EXP/reports/semantic_alignment_v2_pilot500_summary.json" \
  --preview "$EXP/previews/v1_vs_v2_alignment_preview_50.tsv" \
  --review "$EXP/reviews/semantic_alignment_manual_review_50.tsv"

echo "[semantic-v2] step 4/4: structural validation"
"$PYTHON" "$EXP/scripts/validate_semantic_alignment.py" \
  --parents "$EXP/manifests/pilot_parents_500.jsonl" \
  --pairs "$EXP/outputs/semantic_causal_aligned_pairs_v2_pilot500.jsonl" \
  --actions "$EXP/manifests/semantic_actions_v2_pilot500.jsonl" \
  --scores "$EXP/shards/prefix_semantic_scores_shard_00.jsonl" "$EXP/shards/prefix_semantic_scores_shard_01.jsonl" \
  --config "$CONFIG" \
  --report "$EXP/reports/validation_report.json"

echo "[semantic-v2] complete"
echo "[semantic-v2] pairs: $EXP/outputs/semantic_causal_aligned_pairs_v2_pilot500.jsonl"
echo "[semantic-v2] actions: $EXP/manifests/semantic_actions_v2_pilot500.jsonl"
echo "[semantic-v2] summary: $EXP/reports/semantic_alignment_v2_pilot500_summary.json"
