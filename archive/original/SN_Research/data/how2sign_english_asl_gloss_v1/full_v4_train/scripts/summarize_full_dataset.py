#!/usr/bin/env python3
"""Write human-readable full-dataset and professor-update summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation-report", type=Path, required=True)
    parser.add_argument("--repair-report", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--results-md", type=Path, required=True)
    parser.add_argument("--professor-update-md", type=Path, required=True)
    args = parser.parse_args()

    preparation = read_json(args.preparation_report)
    repair = read_json(args.repair_report)
    validation = read_json(args.validation_report)
    status_counts = validation["generation_status_counts"]
    review_count = validation["unique_rows_with_automatic_review_flags"]
    unique_count = validation["unique_prediction_count"]
    review_fraction = review_count / max(1, unique_count)

    results = f"""# Full How2Sign train pseudo-gloss v4 results

## Dataset construction

- Official source: How2Sign manually realigned English train transcription.
- Source SHA-256: `{preparation['source_csv_sha256']}`.
- Official valid sentence records retained: **{preparation['all_records_count']:,}**.
- Source videos represented: **{preparation['source_video_count']:,}**.
- Unique normalized English generation requests: **{preparation['unique_generation_request_count']:,}**.
- Redundant model calls avoided through exact-English deduplication: **{preparation['duplicate_rows_beyond_unique_count']:,}**.

## Frozen generation protocol

- Model: `Qwen/Qwen2.5-7B-Instruct`.
- Revision: `a09a35458c702b33eeacc393d103063234e8bc28`.
- Prompt: `asl_gloss_prompt_v4_deterministic_text_only`.
- Greedy decoding, no sampling, seed `20260720`, batch size `4`.
- Four request shards processed by four RTX 5090 GPUs.
- Output is resumable by stable `request_id`.

## Engineering validation

- Final unique predictions: **{validation['unique_prediction_count']:,}**.
- Final expanded official records: **{validation['expanded_prediction_count']:,}**.
- Empty glosses after repair: **{validation['empty_gloss_count']}**.
- Invalid gloss formats after repair: **{validation['invalid_gloss_format_count']}**.
- Mapping errors: **{validation['expanded_mapping_error_count']}**.
- Transparent deterministic fallback rows: **{validation['deterministic_rule_fallback_count']:,}**.
- Final pipeline status: **{validation['pipeline_ok']}**.
- Quality-gate status: `{validation['quality_gate_status']}`.

Generation status counts:

```json
{json.dumps(status_counts, ensure_ascii=False, indent=2)}
```

## Automatic review triage

- Unique rows with at least one automatic review flag: **{review_count:,} / {unique_count:,} ({review_fraction:.1%})**.
- Review queue: `reviews/full_unique_pseudo_gloss_review_queue_200.tsv`.
- Full automatic audit: `reviews/full_unique_pseudo_gloss_automatic_audit.jsonl`.

These flags are lexical and structural triage signals. They are not ASL
correctness scores.

## Main outputs

- All official records: `outputs/how2sign_train_all_records_31165.pseudo_gloss.qwen2_5_7b_v4.final.jsonl`.
- Unique English records: `outputs/how2sign_train_unique_pseudo_gloss_30102.qwen2_5_7b_v4.final.jsonl`.
- Validation report: `reports/full_dataset_validation_report.json`.

## Scientific limitation

The targets are deterministic text-derived pseudo ASL gloss candidates, not
gold ASL annotations. The successful engineering checks establish provenance,
format, configuration consistency, and exact record mapping. They do not
establish linguistic correctness or sign-video alignment. Qualified ASL review
is required before publication-grade annotation claims.
"""

    professor_update = f"""# Progress update: full How2Sign pseudo-gloss construction

- I scaled the frozen v4 English-to-pseudo-ASL-gloss pipeline to the complete How2Sign manually realigned train transcription.
- All **{preparation['all_records_count']:,}** valid official sentence records from **{preparation['source_video_count']:,}** videos were retained.
- Exact normalized-English deduplication produced **{preparation['unique_generation_request_count']:,}** unique generation requests and avoided **{preparation['duplicate_rows_beyond_unique_count']:,}** redundant model calls.
- Generation used fixed `Qwen2.5-7B-Instruct` revision `a09a354...`, the frozen v4 prompt, greedy decoding, batch size 4, and four RTX 5090 GPUs.
- Final validation found **{validation['empty_gloss_count']}** empty glosses, **{validation['invalid_gloss_format_count']}** invalid formats, and **{validation['expanded_mapping_error_count']}** record-mapping errors after transparent repair.
- **{repair['repaired_count']:,}** invalid primary outputs were retained for audit and replaced with an explicitly marked deterministic fallback.
- The final engineering pipeline status is **{validation['pipeline_ok']}**.
- The final dataset contains pseudo labels, not gold ASL annotations. Automatic QC and the 200-row review queue support triage, but qualified ASL evaluation remains necessary.
"""

    args.results_md.parent.mkdir(parents=True, exist_ok=True)
    args.results_md.write_text(results, encoding="utf-8")
    args.professor_update_md.parent.mkdir(parents=True, exist_ok=True)
    args.professor_update_md.write_text(professor_update, encoding="utf-8")


if __name__ == "__main__":
    main()
