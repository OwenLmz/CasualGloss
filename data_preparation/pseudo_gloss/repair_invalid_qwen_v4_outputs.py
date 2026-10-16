#!/usr/bin/env python3
"""Create a final v4 JSONL with transparent rule fallback for invalid rows."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GLOSS_RE = re.compile(r"^[A-Z0-9][A-Z0-9'/-]*(?: [A-Z0-9][A-Z0-9'/-]*)*$")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
DROP_WORDS = {
    "a",
    "an",
    "the",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "and",
}
CONTRACTIONS = {
    "can't": ["CAN", "NOT"],
    "cannot": ["CAN", "NOT"],
    "won't": ["WILL", "NOT"],
    "don't": ["DO", "NOT"],
    "doesn't": ["DO", "NOT"],
    "didn't": ["DO", "NOT"],
    "isn't": ["NOT"],
    "aren't": ["NOT"],
    "wasn't": ["NOT"],
    "weren't": ["NOT"],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def transparent_fallback(text: str) -> str:
    tokens: list[str] = []
    for raw_word in WORD_RE.findall(text.replace("’", "'")):
        word = raw_word.lower()
        if word in CONTRACTIONS:
            tokens.extend(CONTRACTIONS[word])
            continue
        if word in DROP_WORDS:
            continue
        if word.endswith("'s") and len(word) > 2:
            word = word[:-2]
        token = re.sub(r"[^A-Za-z0-9']", "", word).upper()
        if token:
            tokens.append(token)
    return " ".join(tokens) or "UNKNOWN-SOURCE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl)
    repaired_rows: list[dict[str, Any]] = []
    repaired_ids: list[str] = []
    reason_counts: Counter[str] = Counter()
    for original in rows:
        row = dict(original)
        gloss = str(row.get("target_pseudo_asl_gloss", "")).strip()
        reasons: list[str] = []
        if row.get("generation_status") == "parse_failed":
            reasons.append("primary_json_parse_failed")
        if not gloss:
            reasons.append("primary_empty_gloss")
        elif not GLOSS_RE.fullmatch(gloss):
            reasons.append("primary_invalid_gloss_format")
        if reasons:
            fallback = transparent_fallback(str(row["source_text_en"]))
            if not GLOSS_RE.fullmatch(fallback):
                raise SystemExit(
                    f"Fallback produced invalid gloss for {row.get('request_id')}"
                )
            row["primary_generation_attempt"] = {
                "generation_status": row.get("generation_status"),
                "target_pseudo_asl_gloss": gloss,
                "quality_flags": row.get("quality_flags", []),
                "raw_model_output": row.get("raw_model_output", ""),
            }
            row["target_pseudo_asl_gloss"] = fallback
            row["target_pseudo_asl_gloss_tokens"] = fallback.split()
            row["model_gloss_before_normalization"] = row.get(
                "model_gloss_before_normalization", ""
            )
            row["generation_status"] = "completed_with_rule_based_fallback"
            row["quality_flags"] = sorted(
                set(row.get("quality_flags", []))
                | {
                    "deterministic_transparent_english_rule_fallback",
                    "manual_linguistic_review_required",
                }
            )
            row["fallback"] = {
                "method": "deterministic_transparent_english_content_token_fallback",
                "version": "v1",
                "reasons": reasons,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "important_warning": "This fallback is a transparent pseudo label, not an ASL annotation.",
            }
            repaired_ids.append(str(row.get("request_id")))
            reason_counts.update(reasons)
        repaired_rows.append(row)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in repaired_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "record_count": len(repaired_rows),
        "repaired_count": len(repaired_ids),
        "repaired_ids": repaired_ids,
        "repair_reason_counts": dict(reason_counts),
        "repair_method": "deterministic_transparent_english_content_token_fallback_v1",
        "scientific_status": "fallback_rows_are_pseudo_labels_not_gold_asl",
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
