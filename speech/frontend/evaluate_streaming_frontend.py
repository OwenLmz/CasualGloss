#!/usr/bin/env python3
"""Evaluate final translation quality and streaming behavior."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, reference_token in enumerate(reference, 1):
        current = [i]
        for j, hypothesis_token in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (reference_token != hypothesis_token),
                )
            )
        previous = current
    return previous[-1]


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def aggregate(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 6) if values else None,
        "median": round(statistics.median(values), 6) if values else None,
        "p90": round(percentile(values, 0.90), 6) if values else None,
        "min": round(min(values), 6) if values else None,
        "max": round(max(values), 6) if values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-preview", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--status-field", default="streaming_asr_status")
    parser.add_argument("--hypothesis-field", default="streaming_asr_hypothesis_en")
    parser.add_argument("--first-commit-field", default="first_commit_audio_seconds")
    parser.add_argument("--last-commit-field", default="last_commit_audio_seconds")
    parser.add_argument("--rtf-field", default="streaming_asr_real_time_factor")
    parser.add_argument("--event-count-field", default="streaming_asr_commit_event_count")
    parser.add_argument("--method-label", default="Meta SeamlessStreaming 2.5B monotonic S2TT")
    parser.add_argument(
        "--commitment-definition",
        default="Append-only English emitted by SeamlessStreaming at recorded source-read timestamps.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.predictions)
    status_counts = Counter(row.get(args.status_field, "missing") for row in rows)
    completed = [row for row in rows if row.get(args.status_field) == "completed"]

    total_edits = 0
    total_reference_tokens = 0
    exact = 0
    references: list[str] = []
    hypotheses: list[str] = []
    first_commit: list[float] = []
    last_commit: list[float] = []
    rtfs: list[float] = []
    event_counts: list[float] = []
    per_row: list[dict[str, Any]] = []
    for row in completed:
        reference_tokens = normalize(row["reference_text_en"])
        hypothesis_tokens = normalize(row[args.hypothesis_field])
        distance = edit_distance(reference_tokens, hypothesis_tokens)
        total_edits += distance
        total_reference_tokens += len(reference_tokens)
        is_exact = reference_tokens == hypothesis_tokens
        exact += int(is_exact)
        references.append(row["reference_text_en"])
        hypotheses.append(row[args.hypothesis_field])
        if row.get(args.first_commit_field) is not None:
            first_commit.append(float(row[args.first_commit_field]))
        if row.get(args.last_commit_field) is not None:
            last_commit.append(float(row[args.last_commit_field]))
        if row.get(args.rtf_field) is not None:
            rtfs.append(float(row[args.rtf_field]))
        event_counts.append(float(row.get(args.event_count_field, 0)))
        per_row.append(
            {
                "id": row["id"],
                "speaker": row.get("tts_speaker"),
                "reference": row["reference_text_en"],
                "hypothesis": row[args.hypothesis_field],
                "wer": round(distance / max(len(reference_tokens), 1), 6),
                "exact_match": is_exact,
                "first_commit_audio_seconds": row.get(args.first_commit_field),
                "last_commit_audio_seconds": row.get(args.last_commit_field),
                "real_time_factor": row.get(args.rtf_field),
                "event_count": row.get(args.event_count_field),
            }
        )

    bleu = None
    chrf = None
    try:
        import sacrebleu

        bleu = round(sacrebleu.corpus_bleu(hypotheses, [references]).score, 6)
        chrf = round(sacrebleu.corpus_chrf(hypotheses, [references]).score, 6)
    except ImportError:
        pass

    duplicate_ids = [key for key, count in Counter(row["id"] for row in rows).items() if count > 1]
    expected_ok = args.expected_count is None or len(rows) == args.expected_count
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "predictions": str(args.predictions),
        "method": args.method_label,
        "record_count": len(rows),
        "expected_count": args.expected_count,
        "status_counts": dict(status_counts),
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_ids_preview": duplicate_ids[:20],
        "translation_quality": {
            "corpus_wer": round(total_edits / max(total_reference_tokens, 1), 6),
            "exact_match_rate": round(exact / max(len(completed), 1), 6),
            "sacrebleu": bleu,
            "chrf": chrf,
            "important_scope": "English translation quality against official How2Sign transcription; not ASL accuracy.",
        },
        "streaming_behavior": {
            "first_commit_audio_seconds": aggregate(first_commit),
            "last_commit_audio_seconds": aggregate(last_commit),
            "real_time_factor": aggregate(rtfs),
            "commit_event_count": aggregate(event_counts),
            "commitment_definition": args.commitment_definition,
        },
        "scientific_limitations": [
            "The Mandarin input is synthetic multi-speaker TTS, not real recorded speech.",
            "Latency is measured in an offline chunked simulation and reported separately from compute time.",
            "This report evaluates only the Chinese-speech-to-English frontend.",
        ],
        "ok": expected_ok and not duplicate_ids and len(completed) == len(rows),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_preview.parent.mkdir(parents=True, exist_ok=True)
    with args.output_preview.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(per_row, key=lambda item: item["wer"], reverse=True):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
