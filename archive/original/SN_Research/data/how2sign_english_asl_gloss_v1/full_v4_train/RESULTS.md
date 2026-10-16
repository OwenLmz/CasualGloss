# Full How2Sign train pseudo-gloss v4 results

## Dataset construction

- Official source: How2Sign manually realigned English train transcription.
- Source SHA-256: `67a8f4fb6dc066f38006b89ac84c27a4b05d7fa07d5e6952f997042724d7a847`.
- Official valid sentence records retained: **31,165**.
- Source videos represented: **2,192**.
- Unique normalized English generation requests: **30,102**.
- Redundant model calls avoided through exact-English deduplication: **1,063**.

## Frozen generation protocol

- Model: `Qwen/Qwen2.5-7B-Instruct`.
- Revision: `a09a35458c702b33eeacc393d103063234e8bc28`.
- Prompt: `asl_gloss_prompt_v4_deterministic_text_only`.
- Greedy decoding, no sampling, seed `20260720`, batch size `4`.
- Four request shards processed by four RTX 5090 GPUs.
- Output is resumable by stable `request_id`.

## Engineering validation

- Final unique predictions: **30,102**.
- Final expanded official records: **31,165**.
- Empty glosses after repair: **0**.
- Invalid gloss formats after repair: **0**.
- Mapping errors: **0**.
- Transparent deterministic fallback rows: **95**.
- Final pipeline status: **True**.
- Quality-gate status: `engineering_complete_pending_qualified_asl_review`.

Generation status counts:

```json
{
  "completed": 30007,
  "completed_with_rule_based_fallback": 95
}
```

## Automatic review triage

- Unique rows with at least one automatic review flag: **5,481 / 30,102 (18.2%)**.
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
