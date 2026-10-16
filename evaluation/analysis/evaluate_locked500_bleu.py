#!/usr/bin/env python3
"""Add reproducible corpus BLEU metrics to the frozen locked-500 sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sacrebleu.metrics import BLEU


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokens(row: dict[str, Any], list_key: str, text_key: str) -> list[str]:
    value = row.get(list_key)
    if isinstance(value, list):
        return [str(token) for token in value]
    return str(row.get(text_key, "")).split()


def ngram_counts(sequence: list[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(sequence[i : i + order]) for i in range(len(sequence) - order + 1))


def sentence_statistics(
    hypotheses: list[list[str]], references: list[list[str]], max_order: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(hypotheses)
    correct = np.zeros((count, max_order), dtype=np.int64)
    total = np.zeros((count, max_order), dtype=np.int64)
    hyp_lengths = np.zeros(count, dtype=np.int64)
    ref_lengths = np.zeros(count, dtype=np.int64)

    for i, (hypothesis, reference) in enumerate(zip(hypotheses, references)):
        hyp_lengths[i] = len(hypothesis)
        ref_lengths[i] = len(reference)
        for order in range(1, max_order + 1):
            hyp_ngrams = ngram_counts(hypothesis, order)
            ref_ngrams = ngram_counts(reference, order)
            correct[i, order - 1] = sum(
                min(value, ref_ngrams[ngram]) for ngram, value in hyp_ngrams.items()
            )
            total[i, order - 1] = max(len(hypothesis) - order + 1, 0)
    return correct, total, hyp_lengths, ref_lengths


def bleu_from_aggregates(
    correct: np.ndarray,
    total: np.ndarray,
    hyp_length: np.ndarray,
    ref_length: np.ndarray,
    order: int,
) -> np.ndarray:
    correct = np.asarray(correct, dtype=np.float64)[..., :order]
    total = np.asarray(total, dtype=np.float64)[..., :order]
    hyp_length = np.asarray(hyp_length, dtype=np.float64)
    ref_length = np.asarray(ref_length, dtype=np.float64)

    valid = np.all((correct > 0) & (total > 0), axis=-1) & (hyp_length > 0)
    scores = np.zeros(np.broadcast_shapes(valid.shape, hyp_length.shape), dtype=np.float64)
    if not np.any(valid):
        return scores
    precision_log_mean = np.mean(np.log(correct[valid] / total[valid]), axis=-1)
    brevity_penalty = np.minimum(1.0, np.exp(1.0 - ref_length[valid] / hyp_length[valid]))
    scores[valid] = 100.0 * brevity_penalty * np.exp(precision_log_mean)
    return scores


def corpus_scores(stats: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> dict[str, float]:
    correct, total, hyp_lengths, ref_lengths = stats
    aggregate = (
        correct.sum(axis=0),
        total.sum(axis=0),
        np.asarray(hyp_lengths.sum()),
        np.asarray(ref_lengths.sum()),
    )
    result: dict[str, float] = {}
    for order in range(1, 5):
        result[f"bleu_{order}"] = float(bleu_from_aggregates(*aggregate, order=order))
    result["hypothesis_length"] = int(hyp_lengths.sum())
    result["reference_length"] = int(ref_lengths.sum())
    result["length_ratio"] = float(hyp_lengths.sum() / ref_lengths.sum())
    result["brevity_penalty"] = float(
        min(1.0, math.exp(1.0 - ref_lengths.sum() / hyp_lengths.sum()))
    )
    return result


def paired_bootstrap(
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    candidate: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sample_count = baseline[0].shape[0]
    indices = rng.integers(0, sample_count, size=(repetitions, sample_count), dtype=np.int32)
    output: dict[str, Any] = {}

    def aggregate(stats: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        correct, total, hyp_lengths, ref_lengths = stats
        return (
            correct[indices].sum(axis=1),
            total[indices].sum(axis=1),
            hyp_lengths[indices].sum(axis=1),
            ref_lengths[indices].sum(axis=1),
        )

    baseline_samples = aggregate(baseline)
    candidate_samples = aggregate(candidate)
    for order in range(1, 5):
        difference = bleu_from_aggregates(*candidate_samples, order=order) - bleu_from_aggregates(
            *baseline_samples, order=order
        )
        output[f"bleu_{order}"] = {
            "difference_direction": "candidate_minus_baseline",
            "observed_difference": corpus_scores(candidate)[f"bleu_{order}"]
            - corpus_scores(baseline)[f"bleu_{order}"],
            "bootstrap_mean_difference": float(difference.mean()),
            "ci95_lower": float(np.quantile(difference, 0.025)),
            "ci95_upper": float(np.quantile(difference, 0.975)),
        }
    return output


def validate_and_load(
    name: str,
    path: Path,
    canonical: dict[str, tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[list[str]], list[list[str]]]:
    rows = read_jsonl(path)
    if len(rows) != 500:
        raise SystemExit(f"{name}: expected 500 rows, found {len(rows)}")
    by_id = {str(row["id"]): row for row in rows}
    if len(by_id) != 500:
        raise SystemExit(f"{name}: duplicate IDs")
    if set(by_id) != set(canonical):
        raise SystemExit(f"{name}: locked parent IDs differ")

    ordered = [by_id[parent_id] for parent_id in sorted(canonical)]
    references: list[list[str]] = []
    hypotheses: list[list[str]] = []
    for row in ordered:
        parent_id = str(row["id"])
        source = str(row.get("source_text_en", ""))
        reference = str(row.get("target_pseudo_asl_gloss", ""))
        if (source, reference) != canonical[parent_id]:
            raise SystemExit(f"{name}: source/reference mismatch for {parent_id}")
        references.append(tokens(row, "target_pseudo_asl_gloss_tokens", "target_pseudo_asl_gloss"))
        hypotheses.append(tokens(row, "predicted_pseudo_asl_gloss_tokens", "predicted_pseudo_asl_gloss"))
    return ordered, hypotheses, references


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()

    systems: dict[str, Path] = {
        **{
            f"transllama_wait_k{k}": args.full_root
            / f"modified_wait_k_inference_v1/locked500/v1/k{k}/sentence_rollouts.jsonl"
            for k in range(1, 9)
        },
        "causalgloss_seed20260829": args.full_root
        / "state_recovery_v2_3_full/final_evaluation_v2_3/locked500/v2_3/sentence_rollouts.jsonl",
        "causalgloss_seed20260904": args.full_root
        / "state_recovery_v2_3_multiseed_20260904/final_evaluation_seed20260904/locked500/v2_3/sentence_rollouts.jsonl",
        "causalgloss_seed20260905": args.full_root
        / "state_recovery_v2_3_multiseed_20260905/final_evaluation_seed20260905/locked500/v2_3/sentence_rollouts.jsonl",
    }
    for name, path in systems.items():
        if not path.is_file():
            raise SystemExit(f"Missing {name}: {path}")

    first_rows = read_jsonl(systems["transllama_wait_k1"])
    canonical = {
        str(row["id"]): (
            str(row.get("source_text_en", "")),
            str(row.get("target_pseudo_asl_gloss", "")),
        )
        for row in first_rows
    }
    if len(canonical) != 500:
        raise SystemExit("Canonical k=1 file does not contain 500 unique IDs")

    all_scores: dict[str, dict[str, Any]] = {}
    all_stats: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    input_hashes: dict[str, dict[str, str]] = {}
    for name, path in systems.items():
        rows, hypotheses, references = validate_and_load(name, path, canonical)
        stats = sentence_statistics(hypotheses, references)
        scores = corpus_scores(stats)

        # Independent implementation check against SacreBLEU for standard BLEU-4.
        hypothesis_text = [" ".join(value) for value in hypotheses]
        reference_text = [" ".join(value) for value in references]
        sacrebleu_score = BLEU(tokenize="none", smooth_method="none", effective_order=False).corpus_score(
            hypothesis_text, [reference_text]
        ).score
        if not math.isclose(scores["bleu_4"], sacrebleu_score, abs_tol=1e-10):
            raise SystemExit(
                f"{name}: BLEU implementation mismatch {scores['bleu_4']} vs {sacrebleu_score}"
            )
        scores["completed_with_end_rate"] = float(
            np.mean([bool(row.get("completed_with_end", False)) for row in rows])
        )
        all_scores[name] = scores
        all_stats[name] = stats
        input_hashes[name] = {"path": str(path), "sha256": sha256_file(path)}

    paired: dict[str, Any] = {}
    for seed_name in (
        "causalgloss_seed20260829",
        "causalgloss_seed20260904",
        "causalgloss_seed20260905",
    ):
        paired[seed_name] = paired_bootstrap(
            all_stats["transllama_wait_k1"],
            all_stats[seed_name],
            repetitions=args.bootstrap_repetitions,
            seed=args.seed,
        )

    causal_names = [name for name in all_scores if name.startswith("causalgloss_seed")]
    aggregate: dict[str, Any] = {}
    for metric in ("bleu_1", "bleu_2", "bleu_3", "bleu_4"):
        values = np.asarray([all_scores[name][metric] for name in causal_names])
        aggregate[metric] = {
            "mean": float(values.mean()),
            "sample_standard_deviation": float(values.std(ddof=1)),
            "seed_count": len(values),
        }

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "post_hoc_additional_metric_on_frozen_locked500_outputs",
        "reference_status": "LLM_generated_pseudo_ASL_gloss_not_expert_gold",
        "selection_used_locked_test": False,
        "normalization": "existing whitespace-delimited target/prediction token fields; case preserved",
        "bleu_definition": {
            "implementation": "SacreBLEU-compatible corpus BLEU",
            "tokenizer": "none",
            "smoothing": "none",
            "single_reference": True,
            "reported_scale": "0_to_100",
        },
        "locked_parent_count": 500,
        "same_parent_ids_source_and_reference": True,
        "systems": all_scores,
        "causalgloss_mean_and_sample_sd": aggregate,
        "paired_vs_predeclared_transllama_wait_k1": paired,
        "bootstrap": {
            "method": "paired_parent_level_percentile_bootstrap_of_corpus_BLEU",
            "repetitions": args.bootstrap_repetitions,
            "seed": args.seed,
        },
        "inputs": input_hashes,
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "locked500_bleu_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "locked500_bleu_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["system", "bleu_1", "bleu_2", "bleu_3", "bleu_4", "length_ratio", "completion_rate"]
        )
        for name, score in all_scores.items():
            writer.writerow(
                [
                    name,
                    *[f"{score[f'bleu_{order}']:.6f}" for order in range(1, 5)],
                    f"{score['length_ratio']:.6f}",
                    f"{score['completed_with_end_rate']:.6f}",
                ]
            )

    lines = [
        "Locked-500 BLEU audit",
        "=====================",
        "",
        "Scores use frozen outputs and pseudo-ASL references; BLEU is an added post-hoc metric.",
        "",
        "system\tBLEU-1\tBLEU-2\tBLEU-3\tBLEU-4\tlength ratio",
    ]
    for name, score in all_scores.items():
        lines.append(
            f"{name}\t{score['bleu_1']:.3f}\t{score['bleu_2']:.3f}\t"
            f"{score['bleu_3']:.3f}\t{score['bleu_4']:.3f}\t{score['length_ratio']:.3f}"
        )
    lines.extend(["", "CausalGloss BLEU-4 mean +/- sample SD:"])
    lines.append(f"{aggregate['bleu_4']['mean']:.3f} +/- {aggregate['bleu_4']['sample_standard_deviation']:.3f}")
    (args.output_dir / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
