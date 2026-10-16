# TransLLaMA-Style Learned Adaptive Baseline Protocol

Status: frozen before implementation and before any new locked-test evaluation

Official implementation anchor: `RomanKoshkin/transllama` commit
`344659a79a328f29bd48615d3a3c6edd0ef1ea54`.

## Scientific objective

Build a method-faithful TransLLaMA-style learned adaptive baseline for the
same English-prefix-to-pseudo-ASL-gloss task as CausalGloss. This is an
adaptation to a shared backbone and dataset, not a claim that the original
LLaMA2 English-German/Russian experiments were reproduced.

## Inputs and split discipline

- Read only `id`, `split`, `source_text_en`, `source_tokens`,
  `target_pseudo_asl_gloss`, and `target_pseudo_asl_gloss_tokens` from the
  existing pair manifests.
- Keep the existing parent-level train, validation, and locked-test IDs.
- Do not read existing alignment fields, timelines, action labels, or prior
  learned-baseline outputs when constructing training supervision.
- Use train for fitting and validation for learning-rate and checkpoint
  selection. Do not use the locked 500-parent test split for any selection.
- Evaluate the locked test only after the implementation tests, smoke run,
  learning-rate selection, and three final seed configurations are frozen.

## Method-faithful components

- Tokenize English and gloss with NLTK word tokenization.
- Generate alignments independently with SimAlign using
  `bert-base-multilingual-cased`, BPE token type, `mai` matching, and the
  `itermax` alignment, following the official alignment algorithm.
- Apply the official one-to-one alignment cleanup that keeps the latest edge.
- Insert a WAIT symbol into the target until every retained target alignment
  is causal. Pad the source with filler positions only when required to match
  a longer aligned target.
- During SFT, randomly sample an aligned prefix. Remove earlier WAIT symbols
  and retain only a final WAIT when the truncated target ends in WAIT. At the
  complete source, train on the complete target and EOS.
- Mask system, user, and source prompt tokens from the loss. Compute loss only
  on the target continuation, including WAIT and EOS.
- At inference, a generated WAIT performs READ, a normal gloss token performs
  WRITE without advancing the source, and EOS ends the sentence. The modified
  Wait-k parameter is only a minimum initial-read constraint.

## Shared comparison conditions

- Backbone: `Qwen/Qwen2.5-7B-Instruct` revision
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- Common policy-branch initialization: the existing fine-tuning-1 adapter,
  before CausalGloss-specific timing, semantic, burst, and recovery training.
- Preserve its LoRA structure: rank 16, alpha 32, dropout 0.05, `q_proj` and
  `v_proj` targets.
- Use bfloat16 and the same tokenizer revision and maximum context length.
- Run a development-only learning-rate screen at 1e-5, 3e-5, and 6e-5 with
  one fixed screening seed. Select by target-only validation loss, with
  free-running development metrics used as a sanity gate, not the locked test.
- Use three training epochs, matching the official TransLLaMA SFT schedule.
- After learning-rate selection, train seeds 20260829, 20260904, and 20260905.

## Evaluation

- Evaluate the full predeclared inference sweep k = 1,...,8 for every seed.
- Report Token F1, sequence similarity, BLEU-4, LAAL, completion rate, invalid
  action rate, length ratio, and over-generation diagnostics.
- Report mean and standard deviation over the three seeds.
- Compare with CausalGloss using paired parent-level and aggregate
  seed-and-parent bootstrap 95% confidence intervals.
- Report the complete quality-latency curve. Do not choose k using the locked
  test.
- If interface-compatible, run the selected baseline on the existing CLERC
  human-gloss and Mandarin-speech cascade evaluations as inference-only
  secondary checks.

## Reuse policy

Allowed after verification:

- raw pair text, target glosses, parent split IDs, Qwen cache, fine-tuning-1
  adapter, generic Qwen/PEFT loading code, metric formulas, rollout record
  schema, and bootstrap utilities.

Forbidden:

- prior `learned_baseline` adapters and results, prior v1/v2 action manifests,
  hybrid lexical/semantic timing labels, CausalGloss-specific supervision,
  old TransLLaMA-style inference results as new evidence, and any test-driven
  hyperparameter or checkpoint choice.

## Paper naming

Use `TransLLaMA-style learned adaptive baseline adapted to the shared
backbone, data, and evaluation setting`. Do not call this an exact reproduction
of the original task or model configuration.

