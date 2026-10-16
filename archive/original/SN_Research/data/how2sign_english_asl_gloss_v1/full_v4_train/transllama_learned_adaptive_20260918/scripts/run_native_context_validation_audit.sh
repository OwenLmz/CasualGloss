#!/usr/bin/env bash
set -euo pipefail

ROOT=/123090274/SN_Research
EXP="$ROOT/data/how2sign_english_asl_gloss_v1/full_v4_train/transllama_learned_adaptive_20260918"
PY="$ROOT/venvs/wlasl-cu128-env/bin/python"
PAIRS="$ROOT/data/how2sign_english_asl_gloss_v1/full_v4_train/causal_alignment_v1_corrected_full/manifests/validation_pairs.jsonl"
OUT="$EXP/runs/lr_screen_dev_native_context_audit"

export HF_HOME="$ROOT/huggingface_cache"
export NLTK_DATA="$EXP/nltk_data"
export PYTHONPATH="$EXP/scripts"
mkdir -p "$OUT" "$EXP/logs"

configs=(screen_lr_1e-5 screen_lr_3e-5 screen_lr_6e-5)
pids=()
for gpu in 0 1 2; do
  name="${configs[$gpu]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$EXP/scripts/evaluate_transllama.py" \
    --config "$EXP/configs/${name}.json" \
    --pairs "$PAIRS" \
    --split validation \
    --adapter-dir "$EXP/runs/lr_screen/$name/adapter_best" \
    --output-dir "$OUT/$name" \
    --stage "$name-native-context-audit" \
    --k-values 1,2,3,4,5,6,7,8 \
    --generation-batch-size 32 \
    >"$EXP/logs/${name}_native_context_audit.log" 2>&1 &
  pids+=("$!")
done

failed=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "Validation audit failed for ${configs[$i]}" >&2
    failed=1
  fi
done

if (( failed != 0 )); then
  touch "$OUT/AUDIT_FAILED"
  exit 1
fi

for name in "${configs[@]}"; do
  test -s "$OUT/$name/sweep_summary.json"
done
touch "$OUT/AUDIT_COMPLETE"
