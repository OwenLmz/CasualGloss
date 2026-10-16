#!/usr/bin/env python3
"""Map one pseudo gloss per unique English text back to all How2Sign records."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GENERATION_FIELDS = {
    "target_pseudo_asl_gloss",
    "target_pseudo_asl_gloss_tokens",
    "model_gloss_before_normalization",
    "pseudo_gloss_is_gold",
    "generator_uncertain",
    "generator_note",
    "source_issue",
    "generator_potential_omissions",
    "generation_status",
    "quality_flags",
    "manual_review_status",
    "generation",
    "raw_model_output",
    "primary_generation_attempt",
    "fallback",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-records-jsonl", type=Path, required=True)
    parser.add_argument("--unique-predictions-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()

    all_records = read_jsonl(args.all_records_jsonl)
    unique_predictions = read_jsonl(args.unique_predictions_jsonl)
    prediction_by_request = {
        str(row["request_id"]): row for row in unique_predictions
    }
    if len(prediction_by_request) != len(unique_predictions):
        raise SystemExit("Duplicate request IDs in unique predictions")

    missing: list[str] = []
    expanded: list[dict[str, Any]] = []
    for record in all_records:
        request_id = str(record["request_id"])
        prediction = prediction_by_request.get(request_id)
        if prediction is None:
            missing.append(request_id)
            continue
        generation_values = {
            field: prediction[field]
            for field in GENERATION_FIELDS
            if field in prediction
        }
        expanded.append(
            {
                **record,
                **generation_values,
                "unique_generation_request_id": request_id,
                "unique_generation_request_index": prediction["request_index"],
                "pseudo_gloss_mapping_method": (
                    "exact_normalized_english_key_to_unique_generation_request"
                ),
            }
        )

    if missing:
        raise SystemExit(f"Missing predictions for request IDs: {sorted(set(missing))[:20]}")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in expanded:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_records_jsonl": str(args.all_records_jsonl),
        "all_records_count": len(all_records),
        "unique_predictions_jsonl": str(args.unique_predictions_jsonl),
        "unique_prediction_count": len(unique_predictions),
        "expanded_output_jsonl": str(args.output_jsonl),
        "expanded_output_count": len(expanded),
        "expanded_unique_id_count": len({row["id"] for row in expanded}),
        "mapped_unique_request_id_count": len(
            {row["unique_generation_request_id"] for row in expanded}
        ),
        "duplicate_english_reuse_record_count": sum(
            bool(row["gloss_generation_reused_for_duplicate_english"])
            for row in expanded
        ),
        "output_sha256": file_sha256(args.output_jsonl),
        "pipeline_ok": (
            len(expanded) == len(all_records)
            and len({row["id"] for row in expanded}) == len(all_records)
            and len({row["unique_generation_request_id"] for row in expanded})
            == len(unique_predictions)
        ),
        "scientific_status": "complete_text_derived_pseudo_labels_not_gold_asl",
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pipeline_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
