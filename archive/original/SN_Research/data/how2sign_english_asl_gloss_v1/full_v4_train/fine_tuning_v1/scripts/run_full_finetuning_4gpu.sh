#!/usr/bin/env bash
set -euo pipefail

ROOT=/123090274/SN_Research/data/how2sign_english_asl_gloss_v1/full_v4_train/fine_tuning_v1
PYTHON_BIN=/123090274/SN_Research/venvs/wlasl-cu128-env/bin/python
TORCHRUN=/123090274/SN_Research/venvs/wlasl-cu128-env/bin/torchrun
CONFIG="$ROOT/configs/qwen2_5_7b_lora_full_v1.json"
MANIFESTS="$ROOT/manifests"
RUN="$ROOT/runs/qwen2_5_7b_lora_how2sign_full_v1"

mkdir -p "$RUN" "$ROOT/logs" "$ROOT/results"

"$TORCHRUN" --standalone --nproc_per_node=4 \
  "$ROOT/scripts/train_qwen_lora_full_ddp.py" \
  --config "$CONFIG" \
  --train-manifest "$MANIFESTS/train.jsonl" \
  --validation-manifest "$MANIFESTS/validation.jsonl" \
  --test-manifest "$MANIFESTS/test.jsonl" \
  --output-dir "$RUN"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "$ROOT/scripts/evaluate_full_model.py" \
  --config "$CONFIG" \
  --validation-manifest "$MANIFESTS/validation.jsonl" \
  --test-manifest "$MANIFESTS/test.jsonl" \
  --validation-generation-manifest "$MANIFESTS/validation_generation_eval.jsonl" \
  --test-generation-manifest "$MANIFESTS/test_generation_eval.jsonl" \
  --output-dir "$RUN/evaluation_base" \
  --device cuda:0 \
  --stage base_model \
  > "$ROOT/logs/evaluation_base.log" 2>&1 &
BASE_PID=$!

CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" "$ROOT/scripts/evaluate_full_model.py" \
  --config "$CONFIG" \
  --validation-manifest "$MANIFESTS/validation.jsonl" \
  --test-manifest "$MANIFESTS/test.jsonl" \
  --validation-generation-manifest "$MANIFESTS/validation_generation_eval.jsonl" \
  --test-generation-manifest "$MANIFESTS/test_generation_eval.jsonl" \
  --output-dir "$RUN/evaluation_best" \
  --adapter-dir "$RUN/adapter_best" \
  --device cuda:0 \
  --stage best_lora_adapter \
  > "$ROOT/logs/evaluation_best.log" 2>&1 &
BEST_PID=$!

wait "$BASE_PID"
wait "$BEST_PID"

"$PYTHON_BIN" "$ROOT/scripts/summarize_full_finetuning.py" \
  --training-history "$RUN/training_history.jsonl" \
  --training-summary "$RUN/training_summary.json" \
  --split-report "$ROOT/reports/split_report.json" \
  --base-evaluation "$RUN/evaluation_base/metrics_summary.json" \
  --best-evaluation "$RUN/evaluation_best/metrics_summary.json" \
  --base-test-predictions "$RUN/evaluation_base/test_generation_predictions.jsonl" \
  --best-test-predictions "$RUN/evaluation_best/test_generation_predictions.jsonl" \
  --output-dir "$ROOT/results"

echo "[done] Fine-tune 1 completed."
