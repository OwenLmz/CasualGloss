#!/usr/bin/env python3
"""Convert SimulEval SeamlessStreaming output to the common Task 5 schema."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nltk.tokenize import TreebankWordDetokenizer, TreebankWordTokenizer


TOKENIZER = TreebankWordTokenizer()
DETOKENIZER = TreebankWordDetokenizer()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_metrics(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle, delimiter="\t")
        ]


def decoded_spm_prefixes(pieces: list[str]) -> list[str]:
    current = ""
    prefixes: list[str] = []
    for piece in pieces:
        current += piece.replace("▁", " ")
        prefixes.append(current.lstrip())
    return prefixes


def token_commit_times(
    prediction: str, pieces: list[str], delays_ms: list[float]
) -> tuple[list[str], list[float], list[str]]:
    tokens = TOKENIZER.tokenize(prediction)
    flags: list[str] = []
    if not tokens:
        return [], [], flags
    if len(pieces) != len(delays_ms) or not pieces:
        flags.append("spm_delay_length_mismatch")
        fallback = float(delays_ms[-1]) / 1000.0 if delays_ms else 0.0
        return tokens, [fallback] * len(tokens), flags

    prefixes = decoded_spm_prefixes(pieces)
    decoded = prefixes[-1]
    spans = list(TOKENIZER.span_tokenize(prediction))
    if decoded != prediction:
        flags.append("spm_text_prediction_mismatch")
        times = [
            float(delays_ms[min(len(delays_ms) - 1, (i * len(delays_ms)) // len(tokens))])
            / 1000.0
            for i in range(1, len(tokens) + 1)
        ]
        return tokens, times, flags

    times: list[float] = []
    piece_index = 0
    for _, token_end in spans:
        while piece_index < len(prefixes) - 1 and len(prefixes[piece_index]) < token_end:
            piece_index += 1
        times.append(float(delays_ms[piece_index]) / 1000.0)
    return tokens, times, flags


def build_events(
    prediction: str, pieces: list[str], delays_ms: list[float]
) -> tuple[list[dict[str, Any]], list[str]]:
    tokens, commit_times, flags = token_commit_times(prediction, pieces, delays_ms)
    grouped: dict[float, list[str]] = defaultdict(list)
    for token, commit_time in zip(tokens, commit_times):
        grouped[round(commit_time, 6)].append(token)

    events: list[dict[str, Any]] = []
    committed: list[str] = []
    ordered_times = sorted(grouped)
    for event_index, commit_time in enumerate(ordered_times):
        emitted = grouped[commit_time]
        committed.extend(emitted)
        events.append(
            {
                "audio_read_seconds": commit_time,
                "source_status": (
                    "COMPLETE" if event_index == len(ordered_times) - 1 else "INCOMPLETE"
                ),
                "candidate_translation": prediction,
                "emitted_english_tokens": emitted,
                "committed_english_prefix": DETOKENIZER.detokenize(committed),
                "prefix_conflict": False,
            }
        )
    return events, flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--instances-log", type=Path, required=True)
    parser.add_argument("--metrics-tsv", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    args = parser.parse_args()

    manifest = read_jsonl(args.input_manifest)
    instances = read_jsonl(args.instances_log)
    metrics = read_metrics(args.metrics_tsv)
    if len(instances) != len(metrics):
        raise ValueError("instances.log and metrics.tsv have different row counts")

    output_rows: list[dict[str, Any]] = []
    flag_counts: dict[str, int] = defaultdict(int)
    for position, instance in enumerate(instances):
        index = int(instance.get("index", position))
        if index >= len(manifest):
            raise IndexError(f"SimulEval index {index} exceeds input manifest")
        row = dict(manifest[index])
        prediction = str(instance.get("prediction", "")).strip()
        pieces = [str(piece) for piece in instance.get("prediction_spm", [])]
        delays_ms = [float(value) for value in instance.get("delays", [])]
        events, flags = build_events(prediction, pieces, delays_ms)
        if not prediction:
            flags.append("empty_prediction")
        for flag in set(flags):
            flag_counts[flag] += 1

        first_commit = events[0]["audio_read_seconds"] if events else None
        last_commit = events[-1]["audio_read_seconds"] if events else None
        row.update(
            {
                "audio_duration_seconds": float(
                    row.get("tts_duration_seconds", instance.get("source_length", 0) / 1000.0)
                ),
                "streaming_frontend_status": "completed" if prediction else "failed",
                "streaming_frontend_error": None if prediction else "empty_prediction",
                "streaming_frontend_method": "Meta SeamlessStreaming 2.5B monotonic S2TT",
                "streaming_frontend_model_revision": args.model_revision,
                "streaming_frontend_hypothesis_en": prediction,
                "streaming_frontend_commit_events": events,
                "streaming_frontend_quality_flags": sorted(set(flags)),
                "streaming_frontend_first_commit_audio_seconds": first_commit,
                "streaming_frontend_last_commit_audio_seconds": last_commit,
                "streaming_frontend_commit_event_count": len(events),
                "streaming_frontend_average_lagging_ms": metrics[position]["AL"],
                "streaming_frontend_length_adaptive_average_lagging_ms": metrics[position][
                    "LAAL"
                ],
                "streaming_frontend_generated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        output_rows.append(row)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    completed = sum(row["streaming_frontend_status"] == "completed" for row in output_rows)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": len(output_rows),
        "completed_count": completed,
        "failed_count": len(output_rows) - completed,
        "quality_flag_counts": dict(sorted(flag_counts.items())),
        "mean_average_lagging_ms": round(
            sum(row["streaming_frontend_average_lagging_ms"] for row in output_rows)
            / len(output_rows),
            6,
        ),
        "mean_length_adaptive_average_lagging_ms": round(
            sum(
                row["streaming_frontend_length_adaptive_average_lagging_ms"]
                for row in output_rows
            )
            / len(output_rows),
            6,
        ),
        "output_jsonl": str(args.output_jsonl),
        "ok": completed == len(output_rows),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
