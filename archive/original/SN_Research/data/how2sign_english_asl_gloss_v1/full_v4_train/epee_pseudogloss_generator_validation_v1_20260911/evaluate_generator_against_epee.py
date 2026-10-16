#!/usr/bin/env python3
"""Evaluate frozen pseudo-gloss generator outputs against human Epee glosses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from sacrebleu.metrics import BLEU


VARIANT_SUFFIX_RE = re.compile(r"_(\d+)$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collapse_exact_period(tokens: list[str]) -> list[str]:
    for period in range(1, len(tokens) // 2 + 1):
        if len(tokens) % period == 0 and tokens == tokens[:period] * (len(tokens) // period):
            return tokens[:period]
    return tokens


def normalize_tokens(tokens: list[str], track: str) -> list[str]:
    strict = [" ".join(str(token).upper().replace("’", "'").split()) for token in tokens]
    strict = [token for token in strict if token]
    if track == "strict":
        return collapse_exact_period(strict)
    content = []
    for token in strict:
        token = token.rstrip("?")
        if token == "QUESTION" or not token:
            continue
        content.append(VARIANT_SUFFIX_RE.sub("", token))
    return collapse_exact_period(content)


def overlap_counts(reference: list[str], prediction: list[str]) -> tuple[int, int, int]:
    overlap = sum((Counter(reference) & Counter(prediction)).values())
    return overlap, len(reference), len(prediction)


def token_prf(reference: list[str], prediction: list[str]) -> tuple[float, float, float]:
    overlap, ref_count, pred_count = overlap_counts(reference, prediction)
    precision = overlap / max(1, pred_count)
    recall = overlap / max(1, ref_count)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return precision, recall, f1


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, 1):
        current = [i]
        for j, right_token in enumerate(right, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left_token != right_token))
            )
        previous = current
    return previous[-1]


def score_against_references(
    prediction: list[str], references: list[list[str]]
) -> dict[str, float | int]:
    candidates = []
    for reference in references:
        precision, recall, f1 = token_prf(reference, prediction)
        candidates.append(
            {
                "precision": precision,
                "recall": recall,
                "token_f1": f1,
                "sequence_similarity": SequenceMatcher(
                    None, reference, prediction, autojunk=False
                ).ratio(),
                "gloss_error_rate": edit_distance(reference, prediction) / max(1, len(reference)),
                "exact_match": float(reference == prediction),
                "length_ratio": len(prediction) / max(1, len(reference)),
                "absolute_relative_length_error": abs(len(prediction) - len(reference))
                / max(1, len(reference)),
                "reference_length": len(reference),
            }
        )
    best_index = max(range(len(candidates)), key=lambda index: candidates[index]["token_f1"])
    best = candidates[best_index]
    return {
        "best_reference_index": best_index,
        "best_precision": best["precision"],
        "best_recall": best["recall"],
        "best_token_f1": best["token_f1"],
        "best_sequence_similarity": max(row["sequence_similarity"] for row in candidates),
        "minimum_gloss_error_rate": min(row["gloss_error_rate"] for row in candidates),
        "any_exact_match": max(row["exact_match"] for row in candidates),
        "best_f1_length_ratio": best["length_ratio"],
        "best_f1_absolute_relative_length_error": best[
            "absolute_relative_length_error"
        ],
        "best_f1_reference_length": best["reference_length"],
    }


def bootstrap_mean(values: np.ndarray, repetitions: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = rng.integers(0, len(values), len(values))
        estimates[index] = values[sample].mean()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"mean": float(values.mean()), "ci95": [float(low), float(high)]}


def variable_reference_streams(rows: list[dict[str, Any]], track: str) -> list[list[str]]:
    references_per_row = [row[f"{track}_references"] for row in rows]
    maximum = max(len(references) for references in references_per_row)
    streams: list[list[str]] = [[] for _ in range(maximum)]
    for references in references_per_row:
        rendered = [" ".join(reference) for reference in references]
        rendered.extend([rendered[0]] * (maximum - len(rendered)))
        for index, reference in enumerate(rendered):
            streams[index].append(reference)
    return streams


def corpus_bleu_scores(rows: list[dict[str, Any]], track: str) -> dict[str, float]:
    hypotheses = [" ".join(row[f"{track}_prediction"]) for row in rows]
    references = variable_reference_streams(rows, track)
    scores = {}
    for order in range(1, 5):
        metric = BLEU(
            tokenize="none", lowercase=False, effective_order=True, max_ngram_order=order
        )
        scores[f"bleu_{order}"] = float(metric.corpus_score(hypotheses, references).score)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()

    sources = read_jsonl(args.manifest)
    predictions = {str(row["request_id"]): row for row in read_jsonl(args.predictions)}
    source_ids = [str(row["id"]) for row in sources]
    if len(source_ids) != 401 or set(source_ids) != set(predictions):
        raise SystemExit("Expected exact agreement between 401 source IDs and predictions")

    rows = []
    for source in sources:
        prediction_row = predictions[str(source["id"])]
        raw_prediction = prediction_row.get("target_pseudo_asl_gloss_tokens", [])
        valid = bool(raw_prediction) and prediction_row.get("generation_status") in {
            "completed",
            "completed_with_quality_flags",
        }
        row: dict[str, Any] = {
            "id": source["id"],
            "phrase_id": source["phrase_id"],
            "source_text_en": source["source_text_en"],
            "reference_count": source["reference_count"],
            "generation_status": prediction_row.get("generation_status"),
            "prediction": prediction_row.get("target_pseudo_asl_gloss", ""),
            "invalid_output": float(not valid),
            "generator_uncertain": float(bool(prediction_row.get("generator_uncertain", False))),
        }
        for track, key in (("strict", "strict_tokens"), ("content", "content_tokens")):
            references = [
                normalize_tokens(reference[key], track)
                for reference in source["references"]
                if reference[key]
            ]
            prediction = normalize_tokens(raw_prediction, track)
            row[f"{track}_references"] = references
            row[f"{track}_prediction"] = prediction
            for metric, value in score_against_references(prediction, references).items():
                row[f"{track}_{metric}"] = value
        rows.append(row)

    metric_names = [
        "best_precision",
        "best_recall",
        "best_token_f1",
        "best_sequence_similarity",
        "minimum_gloss_error_rate",
        "any_exact_match",
        "best_f1_length_ratio",
        "best_f1_absolute_relative_length_error",
    ]
    report: dict[str, Any] = {
        "scientific_status": "direct_frozen_generator_vs_external_human_gloss_validation",
        "source_count": len(rows),
        "primary_endpoint": "content best-of-reference Token F1",
        "tracks": {},
        "output_quality": {
            "invalid_output_rate": mean(row["invalid_output"] for row in rows),
            "generator_uncertain_rate": mean(row["generator_uncertain"] for row in rows),
        },
        "bootstrap": {
            "method": "source-level percentile bootstrap of per-source means",
            "repetitions": args.bootstrap_repetitions,
            "seed": args.seed,
        },
        "input_sha256": {
            "manifest": sha256(args.manifest),
            "predictions": sha256(args.predictions),
        },
        "limitations": [
            "Epee is an external clip-level corpus, not a purpose-built text-to-gloss benchmark.",
            "Best-reference scoring accepts signer variation and can be optimistic.",
            "Token overlap does not validate ASL grammar, classifiers, or non-manual markers.",
            "This experiment supports weak-label credibility but is not expert certification of the full corpus.",
        ],
    }
    summary_rows = []
    for track_index, track in enumerate(("strict", "content")):
        track_report: dict[str, Any] = {"corpus_bleu": corpus_bleu_scores(rows, track)}
        for metric_index, metric in enumerate(metric_names):
            values = np.array([float(row[f"{track}_{metric}"]) for row in rows], dtype=float)
            result = bootstrap_mean(
                values,
                args.bootstrap_repetitions,
                args.seed + track_index * 100 + metric_index,
            )
            track_report[metric] = result
            summary_rows.append(
                {
                    "track": track,
                    "metric": metric,
                    "mean": result["mean"],
                    "ci95_low": result["ci95"][0],
                    "ci95_high": result["ci95"][1],
                }
            )
        report["tracks"][track] = track_report

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "generator_human_agreement.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "summary.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)
    with (args.output_dir / "per_source_scores.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
