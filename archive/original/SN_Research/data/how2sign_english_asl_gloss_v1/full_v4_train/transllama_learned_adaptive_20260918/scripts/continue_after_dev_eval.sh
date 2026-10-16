#!/usr/bin/env bash
set -euo pipefail

ROOT=/123090274/SN_Research
EXP="$ROOT/data/how2sign_english_asl_gloss_v1/full_v4_train/transllama_learned_adaptive_20260918"
PY="$ROOT/venvs/wlasl-cu128-env/bin/python"

export HF_HOME="$ROOT/huggingface_cache"
export NLTK_DATA="$EXP/nltk_data"
export PYTHONPATH="$EXP/scripts"

echo "[$(date -Is)] waiting for validation rollout"
while [[ ! -f "$EXP/runs/lr_screen_dev/DEV_EVAL_COMPLETE" ]]; do
  if ! pgrep -f "$EXP/scripts/evaluate_transllama.py.*--split validation" >/dev/null; then
    echo "[$(date -Is)] validation rollout stopped without completion marker" >&2
    touch "$EXP/runs/PIPELINE_FAILED"
    exit 1
  fi
  sleep 60
done

for name in screen_lr_1e-5 screen_lr_3e-5 screen_lr_6e-5; do
  test -s "$EXP/runs/lr_screen_dev/$name/sweep_summary.json"
done

echo "[$(date -Is)] selecting learning rate from validation only"
if ! "$PY" "$EXP/scripts/select_learning_rate.py" \
  >"$EXP/logs/lr_screen_selection.log" 2>&1; then
  touch "$EXP/runs/PIPELINE_FAILED_SELECTION_GATE"
  exit 1
fi

echo "[$(date -Is)] starting final seed training"
if ! bash "$EXP/scripts/run_final_seed_training.sh"; then
  touch "$EXP/runs/PIPELINE_FAILED_FINAL_TRAINING"
  exit 1
fi

echo "[$(date -Is)] starting locked test evaluation"
if ! bash "$EXP/scripts/run_final_seed_test_eval.sh"; then
  touch "$EXP/runs/PIPELINE_FAILED_TEST_EVAL"
  exit 1
fi

touch "$EXP/runs/PIPELINE_COMPLETE"
echo "[$(date -Is)] pipeline complete"
