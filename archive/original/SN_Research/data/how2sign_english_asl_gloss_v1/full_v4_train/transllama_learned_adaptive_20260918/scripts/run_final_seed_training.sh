#!/usr/bin/env bash
set -euo pipefail

ROOT=/123090274/SN_Research
EXP="$ROOT/data/how2sign_english_asl_gloss_v1/full_v4_train/transllama_learned_adaptive_20260918"
PY="$ROOT/venvs/wlasl-cu128-env/bin/python"
SOURCE_ADAPTER="$ROOT/data/how2sign_english_asl_gloss_v1/full_v4_train/fine_tuning_v1/runs/qwen2_5_7b_lora_how2sign_full_v1/adapter_best"

export HF_HOME="$ROOT/huggingface_cache"
export NLTK_DATA="$EXP/nltk_data"
export PYTHONPATH="$EXP/scripts"
mkdir -p "$EXP/runs/final_seeds" "$EXP/logs"

"$PY" "$EXP/scripts/prepare_final_seed_configs.py" \
  >"$EXP/logs/prepare_final_seed_configs.log" 2>&1

seeds=(20260829 20260904 20260905)
pids=()
for gpu in 0 1 2; do
  seed="${seeds[$gpu]}"
  name="final_seed_${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$EXP/scripts/train_transllama.py" \
    --config "$EXP/configs/${name}.json" \
    --train-alignments "$EXP/alignments/train.aligned.jsonl" \
    --validation-alignments "$EXP/alignments/validation.aligned.jsonl" \
    --source-adapter-dir "$SOURCE_ADAPTER" \
    --output-dir "$EXP/runs/final_seeds/$name" \
    >"$EXP/logs/${name}.log" 2>&1 &
  pid="$!"
  pids+=("$pid")
  echo "$pid" >"$EXP/logs/${name}.pid"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  touch "$EXP/runs/final_seeds/FINAL_TRAINING_FAILED"
  exit 1
fi
for seed in "${seeds[@]}"; do
  name="final_seed_${seed}"
  test -s "$EXP/runs/final_seeds/$name/training_summary.json"
  test -s "$EXP/runs/final_seeds/$name/adapter_best/adapter_model.safetensors"
done
touch "$EXP/runs/final_seeds/FINAL_TRAINING_COMPLETE"
