#!/usr/bin/env python3
"""Validate full v4 provenance, unique generation, expansion, and QC triage."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GLOSS_RE = re.compile(r"^[A-Z0-9][A-Z0-9'/-]*(?: [A-Z0-9][A-Z0-9'/-]*)*$")
ENGLISH_WORD_RE = re.compile(r"[A-Za-z0-9']+")
STOPWORDS = {
    "a", "an", "the", "am", "is", "are", "was", "were", "be", "been",
    "being", "do", "does", "did", "have", "has", "had", "to", "of", "for",
    "and", "or", "but", "that", "this", "these", "those", "it", "its", "as",
    "at", "by", "from", "with", "on", "in", "into", "out", "up", "down", "so",
    "then", "just", "very", "i", "you", "he", "she", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "our", "their", "what", "which",
    "who", "there", "here",
}
NEGATION_WORDS = {
    "not", "no", "never", "without", "cannot", "can't", "won't", "don't",
    "doesn't", "didn't",
}
NEGATION_GLOSS = {
    "NOT", "NO", "NEVER", "WITHOUT", "CANNOT", "CAN'T", "WON'T", "DON'T",
}
MODALITY = {
    "must": {"MUST", "REQUIRE", "NEED"},
    "should": {"SHOULD", "OUGHT"},
    "can": {"CAN", "ABLE"},
    "could": {"COULD", "CAN", "ABLE", "POSSIBLE"},
    "may": {"MAY", "POSSIBLE"},
    "might": {"MIGHT", "POSSIBLE"},
    "want": {"WANT"},
    "need": {"NEED", "MUST", "REQUIRE"},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stem(word: str) -> str:
    word = word.lower()
    if len(word) > 5 and word.endswith("ies"):
        return word[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def lexical_proxy(english: str, gloss: str) -> tuple[float, set[str], set[str]]:
    english_words = {
        stem(word)
        for word in ENGLISH_WORD_RE.findall(english)
        if word.lower() not in STOPWORDS
    }
    gloss_atoms = {
        stem(atom)
        for token in gloss.split()
        for atom in re.split(r"[-/]", token)
        if atom
    }
    if not english_words:
        return 1.0, english_words, gloss_atoms
    return len(english_words & gloss_atoms) / len(english_words), english_words, gloss_atoms


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def automatic_audit(request: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    english = str(request["source_text_en"])
    gloss = str(prediction.get("target_pseudo_asl_gloss", "")).strip()
    flags: list[str] = []
    if not gloss:
        flags.append("empty_gloss")
    elif not GLOSS_RE.fullmatch(gloss):
        flags.append("invalid_gloss_format")
    if prediction.get("generator_uncertain"):
        flags.append("generator_uncertain")
    if str(prediction.get("source_issue", "")).strip():
        flags.append("source_issue_reported")
    if prediction.get("generator_potential_omissions"):
        flags.append("generator_reported_potential_omission")

    score, english_words, _ = lexical_proxy(english, gloss)
    token_count = len(gloss.split())
    english_token_count = len(ENGLISH_WORD_RE.findall(english))
    length_ratio = token_count / max(1, english_token_count)
    if score < 0.25 and len(english_words) >= 4:
        flags.append("low_transparent_lexical_coverage_proxy")
    if length_ratio < 0.2 and english_token_count >= 8:
        flags.append("very_short_gloss_relative_to_source")
    if length_ratio > 1.5:
        flags.append("very_long_gloss_relative_to_source")

    lower_words = {word.lower() for word in ENGLISH_WORD_RE.findall(english)}
    upper_gloss_atoms = {
        atom.upper()
        for token in gloss.split()
        for atom in re.split(r"[-/]", token)
        if atom
    }
    if lower_words & NEGATION_WORDS and not upper_gloss_atoms & NEGATION_GLOSS:
        flags.append("negation_not_transparently_preserved")
    for modal, accepted in MODALITY.items():
        if modal in lower_words and not upper_gloss_atoms & accepted:
            flags.append(f"modality_not_transparently_preserved:{modal}")
    source_numbers = {word for word in lower_words if word.isdigit()}
    if source_numbers and not source_numbers.issubset(
        {atom for atom in upper_gloss_atoms if atom.isdigit()}
    ):
        flags.append("numeric_digit_not_transparently_preserved")

    severity = (
        100 * int("empty_gloss" in flags or "invalid_gloss_format" in flags)
        + 20 * int("generator_reported_potential_omission" in flags)
        + 15 * int("generator_uncertain" in flags)
        + 10 * int(any(flag.startswith("negation_") for flag in flags))
        + 8 * sum(flag.startswith("modality_") for flag in flags)
        + 5 * int("low_transparent_lexical_coverage_proxy" in flags)
        + 2 * len(flags)
        + (1 - score)
    )
    return {
        "request_id": request["request_id"],
        "request_index": request["request_index"],
        "source_text_en": english,
        "target_pseudo_asl_gloss": gloss,
        "automatic_review_flags": flags,
        "automatic_review_priority": round(severity, 6),
        "transparent_lexical_coverage_proxy": round(score, 6),
        "gloss_to_english_token_ratio": round(length_ratio, 6),
        "gloss_token_count": token_count,
        "proxy_warning": "Lexical overlap is a triage heuristic, not an ASL quality metric.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-records-jsonl", type=Path, required=True)
    parser.add_argument("--unique-requests-jsonl", type=Path, required=True)
    parser.add_argument("--unique-predictions-jsonl", type=Path, required=True)
    parser.add_argument("--expanded-predictions-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--audit-jsonl", type=Path, required=True)
    parser.add_argument("--review-queue-tsv", type=Path, required=True)
    parser.add_argument("--expected-all-count", type=int, required=True)
    parser.add_argument("--expected-unique-count", type=int, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--review-queue-size", type=int, default=200)
    args = parser.parse_args()

    all_records = read_jsonl(args.all_records_jsonl)
    requests = read_jsonl(args.unique_requests_jsonl)
    predictions = read_jsonl(args.unique_predictions_jsonl)
    expanded = read_jsonl(args.expanded_predictions_jsonl)
    request_ids = [str(row["request_id"]) for row in requests]
    prediction_ids = [str(row.get("request_id", "")) for row in predictions]
    all_ids = [str(row["id"]) for row in all_records]
    expanded_ids = [str(row.get("id", "")) for row in expanded]
    request_by_id = {str(row["request_id"]): row for row in requests}
    prediction_by_id = {str(row.get("request_id", "")): row for row in predictions}

    audit_rows = [
        automatic_audit(request, prediction_by_id[str(request["request_id"])])
        for request in requests
        if str(request["request_id"]) in prediction_by_id
    ]
    flag_counts: Counter[str] = Counter(
        flag for row in audit_rows for flag in row["automatic_review_flags"]
    )
    review_rows = sorted(
        audit_rows,
        key=lambda row: (
            -float(row["automatic_review_priority"]),
            int(row["request_index"]),
        ),
    )[: args.review_queue_size]
    args.audit_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.audit_jsonl.open("w", encoding="utf-8") as handle:
        for row in audit_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.review_queue_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.review_queue_tsv.open("w", encoding="utf-8") as handle:
        handle.write(
            "rank\trequest_index\trequest_id\tsource_text_en\ttarget_pseudo_asl_gloss\t"
            "automatic_review_flags\tlexical_coverage_proxy\n"
        )
        for rank, row in enumerate(review_rows, start=1):
            values = [
                rank,
                row["request_index"],
                row["request_id"],
                row["source_text_en"],
                row["target_pseudo_asl_gloss"],
                json.dumps(row["automatic_review_flags"], ensure_ascii=False),
                row["transparent_lexical_coverage_proxy"],
            ]
            handle.write(
                "\t".join(
                    str(value).replace("\t", " ").replace("\n", " ")
                    for value in values
                )
                + "\n"
            )

    status_counts = Counter(str(row.get("generation_status", "missing")) for row in predictions)
    revision_counts = Counter(
        str(row.get("generation", {}).get("resolved_revision", "missing"))
        for row in predictions
    )
    prompt_hash_counts = Counter(
        str(row.get("generation", {}).get("prompt_sha256", "missing"))
        for row in predictions
    )
    empty_ids = [
        str(row.get("request_id"))
        for row in predictions
        if not str(row.get("target_pseudo_asl_gloss", "")).strip()
    ]
    invalid_ids = [
        str(row.get("request_id"))
        for row in predictions
        if str(row.get("target_pseudo_asl_gloss", "")).strip()
        and not GLOSS_RE.fullmatch(str(row["target_pseudo_asl_gloss"]).strip())
    ]
    gold_error_ids = [
        str(row.get("request_id"))
        for row in predictions
        if row.get("pseudo_gloss_is_gold") is not False
    ]
    expanded_mapping_errors = 0
    for row in expanded:
        prediction = prediction_by_id.get(str(row.get("unique_generation_request_id", "")))
        if prediction is None or row.get("target_pseudo_asl_gloss") != prediction.get(
            "target_pseudo_asl_gloss"
        ):
            expanded_mapping_errors += 1

    accepted_status_count = sum(
        status_counts.get(status, 0)
        for status in (
            "completed",
            "completed_with_quality_flags",
            "completed_with_rule_based_fallback",
        )
    )
    pipeline_ok = all(
        [
            len(all_records) == args.expected_all_count,
            len(requests) == args.expected_unique_count,
            len(predictions) == args.expected_unique_count,
            len(expanded) == args.expected_all_count,
            len(set(all_ids)) == args.expected_all_count,
            len(set(request_ids)) == args.expected_unique_count,
            len(set(prediction_ids)) == args.expected_unique_count,
            len(set(expanded_ids)) == args.expected_all_count,
            request_ids == prediction_ids,
            all_ids == expanded_ids,
            len({row["normalized_english_key"] for row in requests})
            == args.expected_unique_count,
            set(row["request_id"] for row in all_records) == set(request_ids),
            all(
                row.get("source_csv_sha256") == args.expected_source_sha256
                for row in all_records + requests
            ),
            not empty_ids,
            not invalid_ids,
            not gold_error_ids,
            accepted_status_count == args.expected_unique_count,
            revision_counts == Counter({args.expected_revision: args.expected_unique_count}),
            len(prompt_hash_counts) == 1,
            expanded_mapping_errors == 0,
            len(audit_rows) == args.expected_unique_count,
        ]
    )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "full_text_derived_pseudo_labels_not_gold_asl",
        "all_records_jsonl": str(args.all_records_jsonl),
        "all_records_sha256": file_sha256(args.all_records_jsonl),
        "unique_requests_jsonl": str(args.unique_requests_jsonl),
        "unique_requests_sha256": file_sha256(args.unique_requests_jsonl),
        "unique_predictions_jsonl": str(args.unique_predictions_jsonl),
        "unique_predictions_sha256": file_sha256(args.unique_predictions_jsonl),
        "expanded_predictions_jsonl": str(args.expanded_predictions_jsonl),
        "expanded_predictions_sha256": file_sha256(args.expanded_predictions_jsonl),
        "all_record_count": len(all_records),
        "unique_request_count": len(requests),
        "unique_prediction_count": len(predictions),
        "expanded_prediction_count": len(expanded),
        "generation_status_counts": dict(status_counts),
        "resolved_revision_counts": dict(revision_counts),
        "prompt_sha256_counts": dict(prompt_hash_counts),
        "empty_gloss_count": len(empty_ids),
        "invalid_gloss_format_count": len(invalid_ids),
        "pseudo_gloss_is_not_false_count": len(gold_error_ids),
        "expanded_mapping_error_count": expanded_mapping_errors,
        "deterministic_rule_fallback_count": status_counts.get(
            "completed_with_rule_based_fallback", 0
        ),
        "generator_uncertain_count": sum(
            bool(row.get("generator_uncertain")) for row in predictions
        ),
        "generator_potential_omission_count": sum(
            bool(row.get("generator_potential_omissions")) for row in predictions
        ),
        "automatic_review_flag_counts": dict(flag_counts),
        "unique_rows_with_automatic_review_flags": sum(
            bool(row["automatic_review_flags"]) for row in audit_rows
        ),
        "gloss_token_count_unique": summarize(
            [float(row["gloss_token_count"]) for row in audit_rows]
        ),
        "transparent_lexical_coverage_proxy_unique": summarize(
            [float(row["transparent_lexical_coverage_proxy"]) for row in audit_rows]
        ),
        "review_queue_size": len(review_rows),
        "review_queue_tsv": str(args.review_queue_tsv),
        "audit_jsonl": str(args.audit_jsonl),
        "pipeline_ok": pipeline_ok,
        "quality_gate_status": "engineering_complete_pending_qualified_asl_review",
        "interpretation": (
            "pipeline_ok verifies official-source provenance, counts, deterministic "
            "configuration, unique-text generation, exact mapping, and gloss format. "
            "Automatic checks are triage heuristics and do not establish ASL correctness."
        ),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not pipeline_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
