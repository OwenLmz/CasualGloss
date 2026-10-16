#!/usr/bin/env python3
"""Write concise research summaries for the full causal-alignment build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation-report", type=Path, required=True)
    parser.add_argument("--finalization-report", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--professor-update", type=Path, required=True)
    args = parser.parse_args()

    preparation = json.loads(args.preparation_report.read_text(encoding="utf-8-sig"))
    final = json.loads(args.finalization_report.read_text(encoding="utf-8-sig"))
    splits = final["parent_split_counts"]
    actions = final["action_split_counts"]
    results = f"""# Full corrected causal-alignment results

## Status

- Engineering construction status: `{'passed' if final['ok'] else 'failed'}`
- Scientific status: `{final['scientific_status']}`
- Timeline policy: `{final['timeline_policy']}`
- Qualified ASL linguistic review: pending

## Scale

- Fine-tune 1 source parents: {preparation['input_record_count']}
- Candidates after fixed prefilter: {preparation['eligible_after_prefilter_count']}
- Structurally eligible after alignment: {final['eligible_after_alignment_filter_count']}
- Final selected parents: {final['final_parent_count']}
- Train/validation/test parents: {splits.get('train', 0)}/{splits.get('validation', 0)}/{splits.get('test', 0)}
- Train/validation/test actions: {actions.get('train', 0)}/{actions.get('validation', 0)}/{actions.get('test', 0)}
- Held-out-video quarantine parents: {final['heldout_video_quarantine_parent_count']}
- Structurally excluded parents: {final['structurally_excluded_parent_count']}

## Locked evaluation integrity

- Locked validation parent set exact match: {final['locked_parent_id_set_exact_match']['validation']}
- Locked test parent set exact match: {final['locked_parent_id_set_exact_match']['test']}
- Train-validation source-video overlap: {final['source_video_overlap_counts']['train_validation']}
- Train-test source-video overlap: {final['source_video_overlap_counts']['train_test']}
- Validation-test source-video overlap: {final['source_video_overlap_counts']['validation_test']}

## Alignment profile

- Mean lexical alignment ratio: {final['mean_lexical_alignment_ratio']}
- Mean SimAlign fallback ratio: {final['mean_simalign_fallback_ratio']}
- Mean unaligned gloss-token ratio: {final['mean_unaligned_gloss_token_ratio']}
- Mean WAIT-action ratio: {final['mean_wait_action_ratio']}

## Interpretation boundary

This dataset is suitable for the next controlled full-scale causal-policy
fine-tuning experiment. It does not provide gold ASL gloss or gold sign timing.
The fixed validation and test parents must remain evaluation-only.
"""
    professor = f"""# Professor update

- Built the full corrected causal-alignment dataset from the fixed Qwen pseudo-gloss corpus.
- Preserved the controlled 5K experiment's 500 validation and 500 test parents exactly.
- Retained {splits.get('train', 0)} training parents and {actions.get('train', 0)} training actions.
- Quarantined {final['heldout_video_quarantine_parent_count']} extra parents sharing held-out How2Sign videos.
- Verified zero source-video overlap across train, validation, and test.
- Targets and timing remain pseudo labels; qualified ASL review is still pending.
"""
    args.results.write_text(results, encoding="utf-8", newline="\n")
    args.professor_update.write_text(professor, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
