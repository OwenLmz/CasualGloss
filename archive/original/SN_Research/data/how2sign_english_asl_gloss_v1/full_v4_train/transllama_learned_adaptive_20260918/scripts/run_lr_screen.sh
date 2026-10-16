#!/usr/bin/env bash
set -euo pipefail

ROOT=/123090274/SN_Research
EXP="$ROOT/data/how2sign_english_asl_gloss_v1/full_v4_train/transllama_learned_adaptive_20260918"
PY="$ROOT/venvs/wlasl-cu128-env/bin/python"
SOURCE="$ROOT/data/how2sign_english_asl_gloss_v1/full_v4_train/fine_tuning_v1/runs/qwen2_5_7b_lora_how2sign_full_v1/adapter_best"

export HF_HOME="$ROOT/huggingface_cache"
export PYTHONPATH="$EXP/scripts"
mkdir -p "$EXP/runs/lr_screen" "$EXP/logs"

configs=(screen_lr_1e-5 screen_lr_3e-5 screen_lr_6e-5)
for gpu in 0 1 2; do
  name="${configs[$gpu]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$EXP/scripts/train_transllama.py" \
    --config "$EXP/configs/${name}.json" \
    --train-alignments "$EXP/alignments/train.aligned.jsonl" \
    --validation-alignments "$EXP/alignments/validation.aligned.jsonl" \
    --source-adapter-dir "$SOURCE" \
    --output-dir "$EXP/runs/lr_screen/$name" \
    >"$EXP/logs/${name}.log" 2>&1 &
  echo "$!" >"$EXP/logs/${name}.pid"
done

wait
touch "$EXP/runs/lr_screen/SCREEN_COMPLETE"
