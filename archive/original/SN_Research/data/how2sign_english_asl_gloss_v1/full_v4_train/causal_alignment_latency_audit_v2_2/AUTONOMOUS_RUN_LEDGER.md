# Autonomous Causal-Alignment Run Ledger

Date started: 2026-08-17 (Asia/Shanghai)

## Execution order

1. Primary v2.2.1 full Fine-tune 2
   - Full Treebank-repaired v2.2 actions.
   - One predeclared epoch; validation every 1,000 updates.
   - Both `adapter_best` and `adapter_final` are retained.
2. Primary validation and locked test
   - Best/final selected on validation only.
   - Locked 500-parent test is confirmatory.
3. Recovery v2.2.2
   - Continue from validation-selected v2.2.1.
   - WAIT-balanced on-policy states plus delay-recovery states.
   - Validation early stopping; separately reported ablation.
4. v2.2 model decision
   - Validation selects v2.2.1 or v2.2.2.
   - Locked test requires statistically shorter LAAL and quality,
     completion, invalid-output and output-length non-inferiority.
5. Chinese-speech end-to-end confirmation
   - Correct transition semantics: WRITE holds the source prefix and WAIT reads.
   - Reports both audio-time and serial compute-adjusted latency.
6. Exploratory v2.3 local agreement
   - Runs only after the prior GPU tasks release the device.
   - Selects a stability policy on all 500 validation parents.
   - Locked test remains untouched if no validation policy passes constraints.

## Completion artifacts

- `reports/locked500_predeclared_model_decision.json`
- `reports/FINAL_V2_2_RESEARCH_SUMMARY.json`
- `reports/FINAL_V2_2_RESEARCH_SUMMARY.md`
- `reports/FINAL_V2_2_RESULTS.png`
- `reports/FINAL_V2_2_RESULTS.pdf`
- `reports/FINAL_V2_2_REPRODUCIBILITY_MANIFEST.json`
- `chinese_speech_frontend_v0/reports/task5_pilot_100.causal_e2e_selection.json`
- `local_agreement_causal_v2_3/reports/FINAL_V2_3_RESEARCH_SUMMARY.json`
- `local_agreement_causal_v2_3/reports/FINAL_V2_3_RESEARCH_SUMMARY.md`

## Failure handling

- Never overwrite v1, v2.1, v2.2.1 or v2.2.2 results.
- A failed optional branch does not relabel a previous model as successful.
- Inspect logs and resume only idempotent stages with missing outputs.
- Do not tune on the locked test after observing its results.
- Keep every linguistic claim scoped to pseudo gloss until qualified ASL review.
