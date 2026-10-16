#!/usr/bin/env bash
set -euo pipefail

ROOT=/123090274/SN_Research
EXP="$ROOT/data/how2sign_english_asl_gloss_v1/full_v4_train/transllama_learned_adaptive_20260918"
PY="$ROOT/venvs/wlasl-cu128-env/bin/python"
MANIFEST="$ROOT/data/how2sign_english_asl_gloss_v1/full_v4_train/causal_alignment_v1_corrected_full/manifests/validation_pairs.jsonl"

export HF_HOME="$ROOT/huggingface_cache"
export NLTK_DATA="$EXP/nltk_data"
export PYTHONPATH="$EXP/scripts:$EXP/vendor/simalign_official"
mkdir -p "$EXP/alignments/validation_shards" "$EXP/logs"

for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$shard" "$PY" "$EXP/scripts/build_alignments.py" \
    --input "$MANIFEST" \
    --output "$EXP/alignments/validation_shards/shard_${shard}.jsonl" \
    --report "$EXP/alignments/validation_shards/shard_${shard}.report.json" \
    --expected-split validation \
    --device cuda:0 \
    --shard-index "$shard" \
    --num-shards 4 \
    >"$EXP/logs/align_validation_shard_${shard}.log" 2>&1 &
done

wait
touch "$EXP/alignments/VALIDATION_SHARDS_COMPLETE"
