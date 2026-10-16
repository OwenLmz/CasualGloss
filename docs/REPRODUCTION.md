# Reproducing the submitted paper

This repository targets the submitted PDF whose SHA-256 is
`5f7793358d54f9d754aa2fa3c8f1d5fbb723ed4506105c21ca886771e9856e1a`.
It is not a collection of every historical experiment. Source locations and
original checksums are recorded in `provenance/source_manifest.json`.

## 1. Start without a GPU

Run from the repository root:

```bash
python scripts/check_release.py
python -m unittest discover -s tests -v
python -m unittest discover -s tests/verification -v
python scripts/paper.py list
python scripts/paper.py run --dry-run rollout -- --help
```

These checks validate source integrity, syntax and dispatcher isolation. They
do not run a language model and do not establish a training reproduction.
`check_git_snapshot.py` additionally checks the *staged Git blobs* after files
are staged; it is a maintainer check, not a required inference step.

## 2. Data and checkpoints are separate from source code

Obtain the exact recorded manifests and selected adapters according to
`docs/DATA_AND_MODELS.md` and `provenance/artifact_manifest.json`. A remote
path in an inventory does not mean its content has been uploaded to GitHub.
Check licenses/access terms before redistributing source transcripts, CLERC
records, audio, video or model weights. Keep raw data and checkpoints outside
the tracked source tree.

Do not regenerate splits opportunistically. The locked 500-parent test set,
development/confirmation partition, source-video identities, reference
normalization and checkpoint identities are part of the experiment.

## 3. Read and run the real entrypoints

`scripts/paper.py` forwards arguments unchanged and sets only task-specific
first-party import paths. It does not rewrite a configuration, model prompt,
data file, seed, decoding rule or selected checkpoint. Options before the entry
name belong to the dispatcher; options after `--` belong to the selected file.

```bash
python scripts/paper.py run rollout -- --help
python scripts/paper.py run train-policy -- --help
python scripts/paper.py run build-burst -- --help
python scripts/paper.py run build-recovery -- --help
python scripts/paper.py run train-learned -- --help
python scripts/paper.py run eval-epee-action -- --help
python scripts/paper.py run speech-cascade -- --help
```

These underlying scripts import their ML dependencies even for `--help`.
Use `--dry-run` to inspect a command without loading those dependencies.

Configure data/model paths for the machine where the job will run. Historical
configuration paths identify provenance, not portable defaults. Record any
path-only changes in a separate run configuration. Do not substitute model
revisions or reference files silently.

## 4. Main model: preserve the two-stage dependency

The required model chain is:

1. Upstream weak-gloss LoRA, trained on 27,079 paired examples.
2. Stage A confidence/burst Treebank action training: 534,877 actions from
   18,632 parents, initialized from the upstream LoRA.
3. Stage B state-recovery training: 439,103 actions comprising 367,267 balanced
   aligned actions and 71,836 recovery actions, initialized from the selected
   Stage A adapter.

The three final CausalGloss seeds are 20260829, 20260904 and 20260905. Their
paper adapters are all `adapter_final`. In particular, the first seed's
`adapter_best` was its unchanged initialization at step zero and is **not** the
paper's selected model. Preserve selection metadata, not just a filename.

Example of the *real argument interface*, with explicit external paths:

```bash
python scripts/paper.py run --nproc 2 train-policy -- \
  --config /work/configs/recovery_seed20260829.json \
  --train-manifest /work/data/train_actions_full.jsonl \
  --validation-manifest /work/data/development_actions.jsonl \
  --test-manifest /work/data/confirmation_actions.jsonl \
  --source-adapter-dir /work/checkpoints/stage_a/adapter_best \
  --output-dir /work/runs/recovery_seed20260829

python scripts/paper.py run rollout -- \
  --config /work/configs/recovery_seed20260829.json \
  --parent-splits /work/data/parent_splits_full.jsonl \
  --causal-pairs /work/data/causal_aligned_pairs_full.jsonl \
  --adapter-dir /work/checkpoints/recovery_seed20260829/adapter_final \
  --output-dir /work/results/locked500/seed20260829 \
  --stage causalgloss_seed20260829_locked500 \
  --split test --device cuda:0 --generation-batch-size 32
```

The two trainer invocations for Stage A and Stage B use the same implementation
but different recorded configurations, action manifests and source adapters.
Do not train Stage B directly from the upstream weak-gloss model.

## 5. Baselines, ablations and external references

- Fixed Wait-k uses schedule-specific training at k=1,3,5,7, not an
  inference-only constraint on the CausalGloss model. The paper has one seed
  for k=1,3 and three seeds for k=5,7. k=3 uses its recovered finite run.
- The final learned adaptive baseline is the independently trained
  **20260918 TransLLaMA-style SimAlign workflow**. It is not the earlier
  professor-feedback learned-baseline attempt. Decode k=1,...,8 using each
  selected final seed. `train-learned` is single-process; the dispatcher rejects
  `--nproc 2` because the implementation does not implement DDP.
- `ablations/` keeps its own trainer/common/evaluator. The dispatcher isolates
  these same-named modules from `method/causalgloss/`. Do not combine their
  `PYTHONPATH`s. The six Table 2 contrasts include recovery and the five
  component removals, using the frozen diagnostic set.
- CLERC Epee uses the pinned v0.3 revision, best-reference normalization and
  the final FP16 evaluators in `evaluation/clerc/`. The completed-only analysis
  is post-hoc and conditional on baseline completion, not a replacement for
  the primary all-prompts comparison.

## 6. Speech cascade and complete corpus

`speech/frontend/` contains the recorded SeamlessStreaming-to-action cascade,
not alternative unreported speech frontends. Reuse its recorded **320-ms input
read increment** and synthetic locked500 audio. The paper's "1.2-s chunks"
description is an unresolved reporting discrepancy, not a justified replacement
parameter; see `docs/VERIFICATION.md`. Do not change 320 to 1200 merely to match
that wording. Adaptive text release and an audio read increment are distinct.
Report both raw and computation-aware latency; hardware and batching affect
the latter. The shell wrapper requires explicit `TASK_ROOT`, `PYTHON_ENV`,
`INPUT_TSV`, `OUTPUT_DIR` and `END_INDEX`, so it cannot silently use old pilot
inputs or a one-item default.

The speech lexical control is the historical `fine_tuning_v2_causal_full_seed2`
adapter. It must not be conflated with other lexical controls. Evaluate all
three selected CausalGloss seeds for the multi-seed speech summary.

`speech/corpus/` builds the full synthetic ChineseSpeech-ASLGloss-v1 corpus.
The paper corpus is the **cleaned 19,924-item** version (18,924/500/500,
36.54 hours), not the uncleaned 19,947-item intermediate version.

## 7. What has actually been validated

Read `docs/VERIFICATION.md` for the achieved verification level and known
differences. Historical saved-prediction reaggregation, a checkpoint inference
rerun and training from scratch are different claims. The release does not
claim that full multi-seed training was rerun from scratch.

Additional audit runs that are not results in the submitted paper are not
bundled in this paper-only repository. A passed dispatcher/source check alone
says nothing about scientific accuracy or from-scratch reproducibility.
