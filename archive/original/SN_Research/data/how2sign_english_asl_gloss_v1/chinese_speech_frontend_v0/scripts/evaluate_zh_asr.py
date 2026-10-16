#!/usr/bin/env python3
"""Evaluate streaming Mandarin ASR on synthetic TTS speech."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(text: str) -> list[str]:
    return list("".join(re.findall(r"[\u3400-\u9fffA-Za-z0-9]+", text)).casefold())


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


def summarize(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 6) if values else None,
        "median": round(statistics.median(values), 6) if values else None,
        "min": round(min(values), 6) if values else None,
        "max": round(max(values), 6) if values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-preview", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl)
    status_counts = Counter(row.get("zh_asr_status", "missing") for row in rows)
    completed = [row for row in rows if row.get("zh_asr_status") == "completed"]
    total_edits = 0
    total_chars = 0
    exact = 0
    per_row: list[dict] = []
    speaker_edits = defaultdict(int)
    speaker_chars = defaultdict(int)
    speaker_exact = defaultdict(int)
    speaker_count = Counter()
    rtfs: list[float] = []
    for row in completed:
        reference = normalize(row["source_text_zh"])
        hypothesis = normalize(row["zh_asr_transcript"])
        edits = edit_distance(reference, hypothesis)
        is_exact = reference == hypothesis
        total_edits += edits
        total_chars += len(reference)
        exact += int(is_exact)
        speaker = row["tts_speaker"]
        speaker_edits[speaker] += edits
        speaker_chars[speaker] += len(reference)
        speaker_exact[speaker] += int(is_exact)
        speaker_count[speaker] += 1
        rtfs.append(float(row["zh_asr_real_time_factor"]))
        per_row.append(
            {
                "id": row["id"],
                "speaker": speaker,
                "reference_zh": row["source_text_zh"],
                "hypothesis_zh": row["zh_asr_transcript"],
                "character_error_rate": round(edits / max(len(reference), 1), 6),
                "exact_match": is_exact,
                "real_time_factor": row["zh_asr_real_time_factor"],
            }
        )
    duplicate_ids = [key for key, count in Counter(row["id"] for row in rows).items() if count > 1]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_jsonl": str(args.input_jsonl),
        "record_count": len(rows),
        "expected_count": args.expected_count,
        "status_counts": dict(status_counts),
        "duplicate_id_count": len(duplicate_ids),
        "corpus_character_error_rate": round(total_edits / max(total_chars, 1), 6),
        "sentence_exact_match_rate": round(exact / max(len(completed), 1), 6),
        "real_time_factor": summarize(rtfs),
        "by_speaker": {
            speaker: {
                "count": speaker_count[speaker],
                "character_error_rate": round(
                    speaker_edits[speaker] / max(speaker_chars[speaker], 1), 6
                ),
                "sentence_exact_match_rate": round(
                    speaker_exact[speaker] / max(speaker_count[speaker], 1), 6
                ),
            }
            for speaker in sorted(speaker_count)
        },
        "scientific_scope": "Synthetic multi-speaker Mandarin TTS; these metrics do not estimate real-speaker robustness.",
        "ok": (
            len(rows) == args.expected_count
            and len(completed) == len(rows)
            and not duplicate_ids
        ),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_preview.parent.mkdir(parents=True, exist_ok=True)
    with args.output_preview.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(per_row, key=lambda item: item["character_error_rate"], reverse=True):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
