#!/usr/bin/env python3
"""Validate generated 16 kHz mono WAV files and summarize acoustic proxies."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

import numpy as np
import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.open(encoding="utf-8-sig") if line.strip()]
    status = Counter()
    durations: list[float] = []
    rms_values: list[float] = []
    peak_values: list[float] = []
    silence_values: list[float] = []
    issues: list[dict[str, str]] = []
    for row in rows:
        path = args.root / row["tts_audio_16k_mono"]
        if not path.exists():
            status["missing"] += 1
            issues.append({"id": row["id"], "issue": "missing_file"})
            continue
        try:
            info = sf.info(path)
            audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
            mono = audio[:, 0]
            duration = len(mono) / float(sample_rate)
            peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
            rms = float(np.sqrt(np.mean(np.square(mono)))) if len(mono) else 0.0
            silence = float(np.mean(np.abs(mono) < 1e-3)) if len(mono) else 1.0
            row_issues = []
            if sample_rate != 16000:
                row_issues.append("wrong_sample_rate")
            if info.channels != 1:
                row_issues.append("not_mono")
            if not np.isfinite(mono).all():
                row_issues.append("non_finite_samples")
            if duration < 0.5 or duration > 45.0:
                row_issues.append("duration_out_of_range")
            if peak >= 0.999:
                row_issues.append("possible_clipping")
            if rms < 0.003:
                row_issues.append("very_low_rms")
            if silence > 0.8:
                row_issues.append("high_silence_ratio")
            if row_issues:
                status["flagged"] += 1
                issues.append({"id": row["id"], "issue": ",".join(row_issues)})
            else:
                status["valid"] += 1
            durations.append(duration)
            rms_values.append(rms)
            peak_values.append(peak)
            silence_values.append(silence)
        except Exception as exc:
            status["unreadable"] += 1
            issues.append({"id": row["id"], "issue": repr(exc)})

    def summary(values: list[float]):
        return {
            "mean": round(mean(values), 8) if values else None,
            "median": round(median(values), 8) if values else None,
            "min": round(min(values), 8) if values else None,
            "max": round(max(values), 8) if values else None,
        }

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": len(rows),
        "expected_count": args.expected_count,
        "status_counts": dict(status),
        "duration_seconds": summary(durations),
        "rms": summary(rms_values),
        "absolute_peak": summary(peak_values),
        "silence_ratio_abs_lt_1e-3": summary(silence_values),
        "issues": issues,
        "ok": len(rows) == args.expected_count and not (
            status["missing"] or status["unreadable"]
        ),
        "note": "Acoustic proxies verify engineering integrity, not human-perceived naturalness.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

