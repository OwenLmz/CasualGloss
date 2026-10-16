# Full-scale Fine-tune 2 Controlled Experiment

## Training

- Training actions: 397700
- Optimizer updates: 12429
- Baseline -> best validation loss: 0.7803 -> 0.1122

## Locked 500-parent free-running test: 5K -> full

- END completion: 0.9960 -> 1.0000
- Sequence similarity: 0.6525 -> 0.6840
- Token F1: 0.6632 -> 0.6949
- Invalid-action ratio: 0.0030 -> 0.0002
- Sequence paired bootstrap: {'paired_parent_count': 500, 'delta_full_minus_5k': 0.03156576, 'paired_bootstrap_95_ci': [0.01728601, 0.04630266], 'resamples': 5000}
- Token-F1 paired bootstrap: {'paired_parent_count': 500, 'delta_full_minus_5k': 0.03168314, 'paired_bootstrap_95_ci': [0.01747572, 0.04641359], 'resamples': 5000}

## Decision

- Gates: {'preflight_exact_locked_evaluation_match': True, 'validation_loss_improved_over_finetune1': True, 'full_end_completion_at_least_0_98': True, 'full_timeout_at_most_0_02': True, 'full_invalid_action_at_most_0_01': True, 'sequence_similarity_delta_positive': True, 'sequence_similarity_ci_lower_above_zero': True, 'token_f1_delta_positive': True, 'token_f1_ci_lower_above_zero': True, 'normalized_lag_increase_at_most_0_05': True}
- Recommendation: `full_scale_improvement_supported_proceed_to_replication_and_speech_frontend`

All quality targets and timing are pseudo labels, not gold ASL annotations.
