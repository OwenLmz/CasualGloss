# Professor Update: Full-scale Fine-tune 2

- Scaled causal supervision from 71,882 to 397700 actions.
- Restarted from the same Fine-tune 1 adapter used by the 5K control.
- Evaluated both models on the exact same locked 500-parent test set.
- Sequence similarity: 0.6525 -> 0.6840.
- Token F1: 0.6632 -> 0.6949.
- Decision: full_scale_improvement_supported_proceed_to_replication_and_speech_frontend.
- Metrics use pseudo gloss and pseudo timing, not gold ASL labels.
