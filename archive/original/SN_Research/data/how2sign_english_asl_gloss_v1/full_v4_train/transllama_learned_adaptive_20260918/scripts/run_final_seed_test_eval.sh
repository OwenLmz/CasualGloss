#!/usr/bin/env bash
set -euo pipefail

ROOT=/123090274/SN_Research
EXP="$ROOT/data/how2sign_english_asl_gloss_v1/full_v4_train/transllama_learned_adaptive_20260918"
PY="$ROOT/venvs/wlasl-cu128-env/bin/python"
PAIRS="$ROOT/data/how2sign_english_asl_gloss_v1/full_v4_train/causal_alignment_v1_corrected_full/manifests/test_pairs.jsonl"

export HF_HOME="$ROOT/huggingface_cache"
export NLTK_DATA="$EXP/nltk_data"
export PYTHONPATH="$EXP/scripts"
mkdir -p "$EXP/runs/final_test" "$EXP/logs"

test -f "$EXP/runs/final_seeds/FINAL_TRAINING_COMPLETE"
test -f "$EXP/runs/lr_screen_selection.json"

seeds=(20260829 20260904 20260905)
pids=()
for gpu in 0 1 2; do
  seed="${seeds[$gpu]}"
  name="final_seed_${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$EXP/scripts/evaluate_transllama.py" \
    --config "$EXP/configs/${name}.json" \
    --pairs "$PAIRS" \
    --split test \
    --allow-locked-test \
    --adapter-dir "$EXP/runs/final_seeds/$name/adapter_best" \
    --output-dir "$EXP/runs/final_test/$name" \
    --stage "$name-test" \
    --k-values 1,2,3,4,5,6,7,8 \
    --generation-batch-size 32 \
    >"$EXP/logs/${name}_test_eval.log" 2>&1 &
  pid="$!"
  pids+=("$pid")
  echo "$pid" >"$EXP/logs/${name}_test_eval.pid"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  touch "$EXP/runs/final_test/FINAL_TEST_EVAL_FAILED"
  exit 1
fi
for seed in "${seeds[@]}"; do
  test -s "$EXP/runs/final_test/final_seed_${seed}/sweep_summary.json"
done
touch "$EXP/runs/final_test/FINAL_TEST_EVAL_COMPLETE"
