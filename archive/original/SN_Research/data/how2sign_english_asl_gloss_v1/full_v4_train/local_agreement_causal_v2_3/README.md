# Local-Agreement Causal Alignment v2.3

This is a validation-gated fallback for the v2.2 causal-action experiments.
It does not replace v2.2 unless validation evidence supports that decision.

The frozen Fine-tune 1 model translates every incrementally observed English
prefix. The online policy commits only the longest gloss prefix that is
unchanged across a fixed number of consecutive prefix translations. At the end
of the source, the remaining final Fine-tune 1 output is committed and `<END>`
is emitted.

The design is inspired by retranslation/local-agreement and confident
translation-length approaches. It is an adaptation to English-to-pseudo-ASL
gloss and is not claimed as a reproduction of a published system.

Scientific boundaries:

- all references are Qwen-generated pseudo glosses, not expert gold ASL;
- streaming commits are irreversible and later conflicts are reported;
- stability parameters are selected on validation only;
- the locked test set is evaluated once after policy selection.
