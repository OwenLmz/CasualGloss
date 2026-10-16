# Fine-tune 1: full How2Sign English to pseudo-ASL-gloss LoRA

This experiment adapts the frozen `Qwen/Qwen2.5-7B-Instruct` checkpoint to
map official How2Sign English transcriptions directly to the deterministic v4
pseudo-ASL-gloss targets.

## Scientific scope

- The experiment measures agreement with pseudo labels, not linguistic ASL
  correctness.
- No sign video is provided to the LLM during this fine-tuning stage.
- Splits are disjoint by connected How2Sign source-video group to reduce
  leakage from neighboring sentences and duplicated English text.
- The original model revision, data hashes, split seed, LoRA configuration,
  training history, checkpoints, and predictions are retained.

## Experiment stages

1. Build deterministic train/validation/test manifests.
2. Run a four-GPU DDP LoRA smoke test.
3. Train for one full epoch on the training split.
4. Select the checkpoint with the lowest validation pseudo-label loss.
5. Compare the base model and best LoRA checkpoint on fixed held-out data.
6. Export loss/accuracy curves, metrics, examples, and a professor update.

## Completed result

- Split sizes: 27,079 train / 1,516 validation / 1,507 test.
- Four RTX 5090 GPUs, one epoch, effective batch size 16, 1,693 optimizer
  updates.
- Best checkpoint: step 1,693, selected only by validation pseudo-label loss.
- Held-out test pseudo-label loss: 0.7778 (base) to 0.2464 (LoRA).
- Held-out teacher-forced token accuracy: 0.8249 to 0.9140.
- Fixed 500-row test greedy exact match: 0.0540 to 0.2540.
- Fixed 500-row test sequence similarity: 0.5941 to 0.7210.
- No train or validation example was truncated at the 512-token limit.
- Final engineering validation: `pipeline_ok = true`.

The main human-readable outputs are `results/RESULTS.md`,
`results/PROFESSOR_UPDATE.md`, and `results/full_finetuning_curves.png`.
The best reusable LoRA adapter is under
`runs/qwen2_5_7b_lora_how2sign_full_v1/adapter_best/`.
