#!/usr/bin/env python3
"""Compare Task 5 streaming frontends on paired English and gloss outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap(
    baseline: list[float], candidate: list[float], samples: int, seed: int
) -> dict[str, float]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("Paired metric arrays must be non-empty and equal length")
    rng = random.Random(seed)
    deltas: list[float] = []
    size = len(baseline)
    for _ in range(samples):
        indices = [rng.randrange(size) for _ in range(size)]
        deltas.append(
            mean(candidate[index] - baseline[index] for index in indices)
        )
    observed = mean(candidate) - mean(baseline)
    return {
        "observed_delta": round(observed, 8),
        "ci95_low": round(percentile(deltas, 0.025), 8),
        "ci95_high": round(percentile(deltas, 0.975), 8),
        "bootstrap_samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-frontend-report", type=Path, required=True)
    parser.add_argument("--seamless-frontend-report", type=Path, required=True)
    parser.add_argument("--seamless-conversion-report", type=Path, required=True)
    parser.add_argument("--qwen-e2e-jsonl", type=Path, required=True)
    parser.add_argument("--seamless-e2e-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()

    qwen_frontend = read_json(args.qwen_frontend_report)
    seamless_frontend = read_json(args.seamless_frontend_report)
    seamless_conversion = read_json(args.seamless_conversion_report)
    qwen_rows = read_jsonl(args.qwen_e2e_jsonl)
    seamless_rows = read_jsonl(args.seamless_e2e_jsonl)
    qwen_by_id = {row["id"]: row for row in qwen_rows}
    seamless_by_id = {row["id"]: row for row in seamless_rows}
    if set(qwen_by_id) != set(seamless_by_id):
        raise ValueError("The two end-to-end outputs do not contain identical IDs")
    ids = sorted(qwen_by_id)

    metric_fields = {
        "pseudo_gloss_sequence_similarity": "pseudo_gloss_sequence_similarity",
        "pseudo_gloss_token_f1": "pseudo_gloss_token_f1",
        "first_gloss_write_audio_seconds": "speech_to_gloss_first_write_audio_seconds",
    }
    e2e_metrics: dict[str, Any] = {}
    for label, field in metric_fields.items():
        qwen_values = [float(qwen_by_id[row_id][field]) for row_id in ids]
        seamless_values = [float(seamless_by_id[row_id][field]) for row_id in ids]
        e2e_metrics[label] = {
            "qwen3_cascade": round(mean(qwen_values), 8),
            "seamlessstreaming": round(mean(seamless_values), 8),
            "seamless_minus_qwen_paired_bootstrap": paired_bootstrap(
                qwen_values,
                seamless_values,
                args.bootstrap_samples,
                args.seed,
            ),
        }

    qwen_quality = qwen_frontend["translation_quality"]
    seamless_quality = seamless_frontend["translation_quality"]
    qwen_stream = qwen_frontend["streaming_behavior"]
    seamless_stream = seamless_frontend["streaming_behavior"]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "paired_record_count": len(ids),
        "selection": {
            "selected_frontend": "Meta SeamlessStreaming 2.5B monotonic S2TT",
            "status": "selected_for_task5_v0_on_synthetic_100_sample_pilot",
            "reasons": [
                "Higher English BLEU and chrF against the official How2Sign transcription.",
                "Earlier first committed English token and earlier first downstream gloss write.",
                "Higher downstream pseudo-gloss sequence similarity and token F1.",
                "Native monotonic streaming avoids the separate ASR-plus-MT cascade.",
            ],
        },
        "frontend_metrics": {
            "qwen3_cascade": {
                "sacrebleu": qwen_quality["sacrebleu"],
                "chrf": qwen_quality["chrf"],
                "wer": qwen_quality["corpus_wer"],
                "first_commit_audio_seconds_mean": qwen_stream[
                    "first_commit_audio_seconds"
                ]["mean"],
            },
            "seamlessstreaming": {
                "sacrebleu": seamless_quality["sacrebleu"],
                "chrf": seamless_quality["chrf"],
                "wer": seamless_quality["corpus_wer"],
                "first_commit_audio_seconds_mean": seamless_stream[
                    "first_commit_audio_seconds"
                ]["mean"],
                "average_lagging_ms": seamless_conversion["mean_average_lagging_ms"],
                "length_adaptive_average_lagging_ms": seamless_conversion[
                    "mean_length_adaptive_average_lagging_ms"
                ],
            },
        },
        "end_to_end_metrics": e2e_metrics,
        "engineering_checks": {
            "qwen_completed": sum(
                row["speech_to_gloss_status"] == "completed" for row in qwen_rows
            ),
            "seamless_completed": sum(
                row["speech_to_gloss_status"] == "completed" for row in seamless_rows
            ),
            "qwen_premature_end": sum(
                bool(row["speech_to_gloss_premature_end"]) for row in qwen_rows
            ),
            "seamless_premature_end": sum(
                bool(row["speech_to_gloss_premature_end"]) for row in seamless_rows
            ),
        },
        "scientific_interpretation": [
            "The selected model is best only among the tested Task 5 v0 candidates on this synthetic 100-sample pilot.",
            "Pseudo-gloss metrics measure agreement with Qwen-generated weak labels, not gold ASL correctness.",
            "Real Mandarin speech and qualified ASL evaluation remain required before publication claims.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows = [
        {
            "metric": "English BLEU",
            "qwen3_cascade": qwen_quality["sacrebleu"],
            "seamlessstreaming": seamless_quality["sacrebleu"],
            "delta_seamless_minus_qwen": seamless_quality["sacrebleu"]
            - qwen_quality["sacrebleu"],
        },
        {
            "metric": "English chrF",
            "qwen3_cascade": qwen_quality["chrf"],
            "seamlessstreaming": seamless_quality["chrf"],
            "delta_seamless_minus_qwen": seamless_quality["chrf"]
            - qwen_quality["chrf"],
        },
    ]
    for label, values in e2e_metrics.items():
        rows.append(
            {
                "metric": label,
                "qwen3_cascade": values["qwen3_cascade"],
                "seamlessstreaming": values["seamlessstreaming"],
                "delta_seamless_minus_qwen": values[
                    "seamless_minus_qwen_paired_bootstrap"
                ]["observed_delta"],
            }
        )
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
