# Full How2Sign train English-to-pseudo-ASL-gloss v4

This directory scales the frozen and tested v4 text-only pseudo-gloss pipeline
to the complete official How2Sign manually realigned train transcription.

## Data design

- Retain every valid official train sentence record: expected 31,165 rows.
- Normalize English only for exact deduplication using NFKC, whitespace
  normalization, and case folding.
- Generate one pseudo gloss for each distinct normalized English sentence:
  expected 30,102 unique requests.
- Map the generated result back to all official records through `request_id`.
- Preserve official video, sentence, timing, and source-row metadata.

This avoids 1,063 redundant model calls while preserving all sentence-level
How2Sign records. It does not merge semantically similar but non-identical text.

## Frozen generation protocol

- Model: `Qwen/Qwen2.5-7B-Instruct`.
- Revision: `a09a35458c702b33eeacc393d103063234e8bc28`.
- Prompt: `asl_gloss_prompt_v4_deterministic_text_only`.
- Greedy decoding with no sampling, seed `20260720`.
- Batch size `4`, matching the validated v4-1000 protocol.
- Four request shards, one model process per RTX 5090.
- JSONL output is flushed per row and resumes by `request_id`.

A smoke comparison found that larger bf16 batches could change a few greedy
outputs because of numerical differences. Therefore batch size is treated as
part of the frozen generation protocol rather than only a speed parameter.

## Scientific status

The outputs are deterministic text-derived pseudo ASL gloss candidates. They
are not gold ASL annotations and do not prove alignment with the corresponding
sign video. Automatic checks support engineering integrity and review triage;
qualified ASL evaluation remains required.

## Main final output

`outputs/how2sign_train_all_records_31165.pseudo_gloss.qwen2_5_7b_v4.final.jsonl`

The unique-text version is retained at:

`outputs/how2sign_train_unique_pseudo_gloss_30102.qwen2_5_7b_v4.final.jsonl`

## Completion status

- Full generation completed on four RTX 5090 GPUs.
- Engineering validation passed: 30,102 unique outputs and 31,165 mapped
  official records, with no empty final glosses, invalid final formats, or
  record-mapping errors.
- Ninety-five invalid primary model responses were retained transparently with
  deterministic rule-based fallback labels and explicit status metadata.
- Automatic review flags were assigned to 5,481 unique rows for triage; these
  flags are not ASL correctness judgments.
- Qualified ASL review is still required before treating any pseudo label as a
  publication-grade linguistic annotation.

See `RESULTS.md` for the complete result summary,
`PROFESSOR_UPDATE.md` for a concise progress report, and
`reports/final_core_sha256.txt` for core output checksums.
