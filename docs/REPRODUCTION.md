# Reproducing the submitted paper

**Status: validation in progress, not complete reproduction.** Read
[VERIFICATION.md](VERIFICATION.md) and the current
[status record](../results/reproduction_status.json). Frozen-output scoring,
checkpoint inference and new training are different verification levels.

Paper SHA-256: `5f7793358d54f9d754aa2fa3c8f1d5fbb723ed4506105c21ca886771e9856e1a`.
Run from the repository root. Use absolute PRIVATE paths outside the checkout;
all new output directories below must not already exist. The installed asset
root corresponds to old `SN_Research`, not its `data/` child.

## 1. Inspect without executing a model

```bash
python scripts/check_release.py
python -m unittest discover -s tests -v
python -m unittest discover -s tests/verification -v
python scripts/reproduction/run_checkpoint_suite.py --list
python scripts/reproduction/run_training_plan.py --list
```

Tests require NumPy (`requirements/analysis.txt`). Passing source checks and
tests does not establish model execution, metric equality or training success.

## 2. Install frozen private assets

Git clone downloads source and the catalog, not large data/weights. The private
`OwenLmz/CasualGloss` Release `paper-reproduction-20261016` requires repository
access and upstream data terms; see [DATA_AND_MODELS.md](DATA_AND_MODELS.md) and
[THIRD_PARTY.md](../THIRD_PARTY.md). Release validation is still in progress;
catalog/attachment availability must be checked before claiming installation.

```bash
python scripts/reproduction/release_assets.py list
python scripts/reproduction/release_assets.py install \
  --destination /private/paper_assets --accept-data-terms
python scripts/reproduction/release_assets.py verify \
  --destination /private/paper_assets
```

Online installation uses authenticated `gh` or the existing Git credential
helper without printing credentials. For predownloaded attachments, add
`--asset-dir /private/release_downloads` to `install`; that mode is offline.
Assets/cache and destination must be separate external trees. Matching files
can resume; mismatches, unrelated content and link/path escapes are rejected.

Default groups are `data,adapters,audio`. `--groups data,adapters` suffices for
the CPU/checkpoint-policy routes below, not full waveform reconstruction; pass
the same groups to `verify` when only a subset is installed. Original JSONL
bytes, historical paths, splits and results remain unchanged. The installer
does not make old absolute audio paths valid on a different machine.

### Rebuild and verify CPU-only data products

After installing all groups, use an environment with NumPy and soundfile for
waveform QC/finalization. The recovery builder can use the policy interpreter.
No model is loaded and no audio is synthesized by this command:

```bash
python -m pip install -r requirements/data-builders.txt
python scripts/reproduction/validate_data_builders.py \
  --data-root /private/paper_assets --output-root /private/data_builder_run1
```

The small CPU requirements include NLTK 3.9.2 for the original recovery
builder's Treebank detokenizer. Both producer interpreters default to the
current Python; `--tts-python` / `--policy-python` can select separate runtimes.

The default runs original waveform QC, corpus finalization and recovery-manifest
production. Repeated `--check waveform-qc`, `--check finalizer`, `--check recovery`
select subsets. Recovery-only needs no audio group. Checks verify producer/input
hashes before execution, compare stable QC/summary fields, and require byte/order
equality for rebuilt cleaned and recovery manifests. The 71,836-row recovery
intermediate is regenerated and checked by its recorded SHA, not stored twice.
Read `cpu_validation_report.json`; new timestamps/path metadata are not claimed
to be byte-identical to historical reports.

## 3. Independently rescore saved predictions on CPU

Materialize exactly 83 inputs: 82 historical predictions (35,535 records) and
one CLERC reference manifest. Both output directories must be new.

```bash
python scripts/reproduction/materialize_verification_inputs.py \
  --data-root /private/paper_assets --output-dir /private/verification_inputs
python scripts/verification/verify_paper_results.py \
  --input-root /private/verification_inputs --output-dir /private/frozen_scores_run1
```

Materialization copies checked bytes and writes an offline receipt; it does
not invent a new remote-server checksum observation. Read the generated
`verification_report.json`: it independently scores predictions, checks input
counts, displayed values and selected bootstrap CIs. Keep default 20,000 draws
for the reported 20,000-draw checks. This route loads no checkpoint and produces
no new prediction; historical numerical agreement is not fresh inference.

## 4. Obtain the pinned model and check the environment

Official backbone: `Qwen/Qwen2.5-7B-Instruct`, revision
`a09a35458c702b33eeacc393d103063234e8bc28`. This is separate from private adapters.
In an environment with `huggingface_hub`:

```bash
python scripts/reproduction/download_policy_model.py \
  --models-root /private/hf_home --accept-upstream-terms
python scripts/reproduction/setup_nltk.py --destination /private/nltk_data
python scripts/reproduction/preflight.py \
  --data-root /private/paper_assets --models-root /private/hf_home \
  --nltk-data /private/nltk_data --environment policy --verify-data-hashes
```

Add `--local-only` to the downloader to check an existing cache offline. It
requests the fixed snapshot and checks required config/tokenizer files and
four weight shards. `--models-root` accepts HF_HOME containing `hub/`, a
compatible model cache, or the exact pinned snapshot.

[Environment guidance](../requirements/README.md) separates policy, TTS and
Seamless runtimes. Locks record the **current inspected server**, not a proven
historical environment or tested clean installation. Do not merge incompatible
CUDA/PyTorch environments. Preflight installs nothing; it checks packages,
CUDA, model/input identities and NLTK resources. Use `--nltk-data /private/nltk_data`
if resources are elsewhere; `--require-recorded-versions` makes policy version
differences errors instead of warnings. A pass is not numerical reproduction.
`setup_nltk.py` verifies the original official `punkt_tab` ZIP and all four
English model files by SHA/size; `--verify-only` checks without network access.
Keep NLTK resources outside the installed paper-data tree. Legacy Punkt and
WordNet are not required by the inspected retained policy runtime.

## 5. Rerun selected checkpoints: 82 outputs, 40 invocations

[experiments.json](../configs/reproduction/experiments.json) contains 35
locked500, 35 CLERC401, eight diagnostic250 and four speech500 outputs. Default
coalescing retains six learned k=1,...,8 sweeps: **40 model invocations** produce
**82 outputs**. Inspect the no-write plan first:

```bash
python scripts/reproduction/run_checkpoint_suite.py --dry-run \
  --data-root /private/paper_assets --models-root /private/hf_home \
  --output-root /private/checkpoint_rerun1
```

To execute, remove `--dry-run`: unlike the training runner, this runner then
executes after input checks. `--family locked500`, `clerc401`, `diagnostic250`
or `speech500` selects a subset; `--id` selects a listed entry. `--python`
chooses the policy interpreter; `CUDA_VISIBLE_DEVICES` selects the physical GPU.
Do not use `--separate-k` as an exact historical grouped-sweep substitute.

The runner verifies inputs/adapters, preserves historical precision, batches,
order and caps, and changes only the local `model_id` path in new runtime
configs. It writes logs, `receipt.json` and `checkpoint_output_index_*.json`
without replacing archived predictions. A completed receipt proves exit status
and record count, **not metric equality**. Independently rescore fresh outputs
and record actual exact/approximate differences. Cross-hardware low-precision
execution need not be bit-identical.

The 2026-10-17 audit exercised all 82 outputs and independently checked their
inputs, receipts and metrics. It found actual numerical differences, not exact
paper-number reproduction; see
[checkpoint_reinference_20261017.json](../results/checkpoint_reinference_20261017.json).
No tolerance or expected value was changed to turn that result into a pass.

To rescore a newly completed full suite rather than archived predictions:

```bash
python scripts/reproduction/materialize_checkpoint_outputs.py \
  --checkpoint-root /private/checkpoint_rerun1 \
  --data-root /private/paper_assets --output-dir /private/fresh_verifier_inputs
python scripts/verification/verify_paper_results.py \
  --input-root /private/fresh_verifier_inputs --output-dir /private/fresh_scores_run1
```

Repeat `--checkpoint-root` if the 82 outputs are split across GPU directories.
The materializer requires all 82 unique matrix entries with valid completed
runner receipts, output hashes/counts and the registered 401-prompt reference.
It copies independently, never links files or claims new outputs match old
prediction hashes. The original numerical verifier may fail on real differences;
that failure must be reported, not suppressed or relaxed.

The four speech500 jobs rerun **only the policy on recorded shared frontend
events**. They do not resynthesize TTS, rerun Seamless, or reproduce historical
frontend wall time. Full waveform/QC/frontend regeneration is a separate level.
The retained Seamless launcher reads 320-ms increments; do not change it to
1200 to fit the paper wording.

## 6. Inspect training before explicitly executing it

[training.json](../configs/reproduction/training.json) records 30 invocations:
upstream (1), main policy (4), fixed Wait-k (8), speech lexical control (1),
independent learned baseline (6, including validation-only LR screens), and
five two-stage component ablations (10).

```bash
python scripts/reproduction/run_training_plan.py --dry-run \
  --data-root /private/paper_assets --models-root /private/hf_home \
  --output-root /private/new_training1
```

Dry-run is also the default: no model load or output writes. Only after
reviewing compute cost, replace `--dry-run` with **`--execute`** to train.
`--id`/`--family` restrict selection; `--with-dependencies` includes required
new upstream jobs. Default `--initialization-source new` follows new adapters
and validation selection; `archived` retrains individual stages from archived
initializers and is not whole-chain retraining. Single-GPU DDP still uses torchrun.

Main dependency: weak-gloss LoRA → Stage A (534,877 actions) → Stage B (439,103
actions). Published Stage B seeds select `adapter_final`; S1's unchanged
step-zero `adapter_best` is not the paper checkpoint. New validation-only
selection stops if its chosen LR differs rather than silently changing paper
settings. Read each `training_receipt.json`.

Training uses frozen weak labels, alignments and action manifests; it does not
automatically regenerate these or train Qwen from random weights. After real
new training, checkpoint inference can use `--training-root /private/new_training1`
to validate new receipts/hashes instead of archived adapters. Full new training
has not been demonstrated; runnable plans alone are not reproduced results.

## Lower-level study and interpretation

`python scripts/paper.py list` exposes historical implementations;
`run --dry-run ENTRY -- --help` inspects dispatch without ML imports. Actual
implementation `--help` may require ML dependencies. The dispatcher isolates
main/ablation modules and forwards arguments unchanged. See
[PAPER_TO_CODE.md](PAPER_TO_CODE.md).

Keep every caveat in [VERIFICATION.md](VERIFICATION.md), particularly upstream
test exposure, weak-target lineage, reference-length generation caps,
320-ms/1.2-s wording and native no-recovery completion. Byte checks and matching
scores do not resolve those scientific/reporting limitations.
