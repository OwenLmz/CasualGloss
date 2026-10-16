#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/123090274/SN_Research}"
FULL_ROOT="$ROOT/data/how2sign_english_asl_gloss_v1/full_v4_train"
SCRIPT="$FULL_ROOT/locked500_bleu_metric_v1_20260911/evaluate_locked500_bleu.py"
OUTPUT="$FULL_ROOT/locked500_bleu_metric_v1_20260911/reports/run_20260911"
PYTHON_BIN="$ROOT/venvs/wlasl-cu128-env/bin/python"
METRICS_RUNNER="$FULL_ROOT/epee_pseudogloss_generator_validation_v1_20260911/run_with_metrics_deps.py"

[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite existing output: $OUTPUT" >&2; exit 1; }
"$PYTHON_BIN" "$METRICS_RUNNER" "$SCRIPT" \
  --full-root "$FULL_ROOT" \
  --output-dir "$OUTPUT" \
  --bootstrap-repetitions 10000 \
  --seed 20260911
