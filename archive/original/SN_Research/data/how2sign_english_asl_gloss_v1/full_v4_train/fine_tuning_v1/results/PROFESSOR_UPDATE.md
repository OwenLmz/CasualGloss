# Professor update: Fine-tune 1

- Constructed leakage-controlled splits from 30,102 unique How2Sign English/pseudo-gloss pairs: 27079 train, 1516 validation, and 1507 test.
- Fine-tuned the fixed Qwen2.5-7B-Instruct checkpoint with rank-16 LoRA using four RTX 5090 GPUs.
- Selected checkpoint step 1693 using validation pseudo-label loss only.
- On the unseen test split, loss changed from 0.7778 to 0.2464, token accuracy from 0.8249 to 0.9140, and greedy sequence similarity from 0.5941 to 0.7210.
- These are pseudo-label agreement metrics, not gold ASL accuracy. Causal-alignment fine-tuning remains the next stage.
