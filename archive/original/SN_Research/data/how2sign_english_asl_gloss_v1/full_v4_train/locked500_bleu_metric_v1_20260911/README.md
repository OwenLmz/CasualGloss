# Locked-500 BLEU metric audit

This analysis adds BLEU-1 through BLEU-4 to already frozen outputs. It does not
run inference, change model selection, or tune on the locked set. All systems
must contain the same 500 parent IDs, English sources, and pseudo-ASL references.

The complete modified Wait-k sweep (`k=1..8`) and all three CausalGloss seeds
are scored. CausalGloss-versus-`k=1` BLEU differences use 10,000 paired
parent-level bootstrap resamples. The report labels BLEU as a post-hoc added
metric and the references as LLM-generated pseudo-ASL rather than expert gold.
