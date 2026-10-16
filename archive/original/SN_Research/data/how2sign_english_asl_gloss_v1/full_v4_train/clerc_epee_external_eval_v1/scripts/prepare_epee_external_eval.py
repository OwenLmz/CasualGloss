#!/usr/bin/env python3
"""Build a source-level, multi-reference CLERC Epee evaluation manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from nltk.tokenize import TreebankWordTokenizer


TOKENIZER = TreebankWordTokenizer()
VARIANT_SUFFIX_RE = re.compile(r"_(\d+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'")
    return " ".join(value.split())


def normalize_gloss_piece(piece: str) -> str:
    return normalize_text(piece).upper()


def split_gloss_sequence(value: str) -> list[str]:
    tokens: list[str] = []
    for segment in value.split("|"):
        segment = normalize_gloss_piece(segment.strip())
        if segment:
            tokens.extend(segment.split())
    return tokens


def collapse_exact_period(tokens: list[str]) -> tuple[list[str], int]:
    size = len(tokens)
    for period in range(1, size // 2 + 1):
        if size % period == 0 and tokens == tokens[:period] * (size // period):
            return tokens[:period], size // period
    return list(tokens), 1


def content_normalize(tokens: list[str]) -> tuple[list[str], int]:
    normalized = []
    for token in tokens:
        token = token.rstrip("?")
        if token == "QUESTION" or not token:
            continue
        normalized.append(VARIANT_SUFFIX_RE.sub("", token))
    return collapse_exact_period(normalized)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-commit", required=True)
    args = parser.parse_args()

    with args.metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        clips = list(csv.DictReader(handle))
    required = {
        "clip_id", "signer_id", "phrase_id", "parallel", "sign_language",
        "text_en", "fps", "n_frames", "gloss_sequence",
    }
    if not clips or not required.issubset(clips[0]):
        raise ValueError(f"Missing metadata columns: {sorted(required - set(clips[0]))}")

    by_phrase: dict[int, list[dict[str, str]]] = defaultdict(list)
    clip_rows: list[dict[str, Any]] = []
    for clip in clips:
        phrase_id = int(clip["phrase_id"])
        strict_raw = split_gloss_sequence(clip["gloss_sequence"])
        strict, strict_repeat = collapse_exact_period(strict_raw)
        content, content_repeat = content_normalize(strict_raw)
        row = {
            "clip_id": clip["clip_id"],
            "signer_id": clip["signer_id"],
            "phrase_id": phrase_id,
            "parallel": clip["parallel"].strip().lower() == "yes",
            "sign_language": clip["sign_language"],
            "text_en": normalize_text(clip["text_en"]),
            "fps": float(clip["fps"]),
            "n_frames": int(clip["n_frames"]),
            "gloss_sequence_raw": clip["gloss_sequence"],
            "strict_tokens_raw": strict_raw,
            "strict_tokens": strict,
            "content_tokens": content,
            "strict_whole_sequence_repeat_factor": strict_repeat,
            "content_whole_sequence_repeat_factor": content_repeat,
        }
        clip_rows.append(row)
        by_phrase[phrase_id].append(row)

    sources: list[dict[str, Any]] = []
    inconsistent_text = []
    for phrase_id, references in sorted(by_phrase.items()):
        normalized_texts = sorted({normalize_text(row["text_en"]) for row in references})
        if len(normalized_texts) != 1:
            inconsistent_text.append({"phrase_id": phrase_id, "texts": normalized_texts})
            continue
        text_en = normalized_texts[0]
        reference_rows = [
            {
                "clip_id": row["clip_id"],
                "signer_id": row["signer_id"],
                "strict_tokens_raw": row["strict_tokens_raw"],
                "strict_tokens": row["strict_tokens"],
                "content_tokens": row["content_tokens"],
                "strict_whole_sequence_repeat_factor": row["strict_whole_sequence_repeat_factor"],
                "content_whole_sequence_repeat_factor": row["content_whole_sequence_repeat_factor"],
            }
            for row in sorted(references, key=lambda item: (item["signer_id"], item["clip_id"]))
        ]
        reference_lengths = [len(row["content_tokens"]) for row in reference_rows if row["content_tokens"]]
        median_reference_length = int(median(reference_lengths)) if reference_lengths else 1
        canonical = min(
            reference_rows,
            key=lambda row: (
                abs(len(row["content_tokens"]) - median_reference_length),
                row["signer_id"],
            ),
        )
        all_parallel = all(row["parallel"] for row in references)
        if all_parallel and len(reference_rows) >= 4:
            subset = "parallel_multi_reference"
        elif all_parallel:
            subset = "parallel_single_reference"
        else:
            subset = "second_register_single_reference"
        sources.append(
            {
                "id": f"clerc_epee_phrase_{phrase_id:04d}",
                "phrase_id": phrase_id,
                "evaluation_subset": subset,
                "source_text_en": text_en,
                "source_tokens": TOKENIZER.tokenize(text_en),
                "reference_count": len(reference_rows),
                "unique_strict_reference_count": len({tuple(row["strict_tokens"]) for row in reference_rows}),
                "unique_content_reference_count": len({tuple(row["content_tokens"]) for row in reference_rows}),
                "median_content_reference_length": median_reference_length,
                "canonical_reference_tokens_for_runtime_guard_only": canonical["content_tokens"],
                "references": reference_rows,
                "dataset_commit": args.dataset_commit,
            }
        )

    normalized_text_groups: dict[str, list[int]] = defaultdict(list)
    for row in sources:
        normalized_text_groups[normalize_text(row["source_text_en"]).casefold()].append(row["phrase_id"])
    duplicate_texts = {
        text: ids for text, ids in normalized_text_groups.items() if len(ids) > 1
    }

    subset_counts = Counter(row["evaluation_subset"] for row in sources)
    signer_counts = Counter(row["signer_id"] for row in clip_rows)
    reference_count_distribution = Counter(row["reference_count"] for row in sources)
    audit = {
        "scientific_status": "external_human_asl_gloss_reference_audit",
        "dataset": "CLERC-DATA/epee",
        "dataset_version": "v0.3",
        "dataset_commit": args.dataset_commit,
        "license": "CC BY-NC-SA 4.0",
        "metadata_path": str(args.metadata),
        "metadata_sha256": sha256(args.metadata),
        "clip_count": len(clip_rows),
        "source_phrase_count": len(sources),
        "signer_count": len(signer_counts),
        "signer_clip_counts": dict(sorted(signer_counts.items())),
        "subset_source_counts": dict(sorted(subset_counts.items())),
        "reference_count_distribution": {
            str(key): value for key, value in sorted(reference_count_distribution.items())
        },
        "human_reference_count": sum(row["reference_count"] for row in sources),
        "unique_strict_reference_sequence_count": len(
            {tuple(row["strict_tokens"]) for source in sources for row in source["references"]}
        ),
        "unique_content_reference_sequence_count": len(
            {tuple(row["content_tokens"]) for source in sources for row in source["references"]}
        ),
        "clips_with_strict_exact_whole_sequence_repetition": sum(
            row["strict_whole_sequence_repeat_factor"] > 1 for row in clip_rows
        ),
        "clips_with_content_exact_whole_sequence_repetition": sum(
            row["content_whole_sequence_repeat_factor"] > 1 for row in clip_rows
        ),
        "clips_with_question_marker": sum(
            "QUESTION" in row["strict_tokens_raw"] for row in clip_rows
        ),
        "tokens_with_variant_suffix": sum(
            bool(VARIANT_SUFFIX_RE.search(token))
            for row in clip_rows
            for token in row["strict_tokens_raw"]
        ),
        "duplicate_normalized_english_text_across_phrase_ids": duplicate_texts,
        "inconsistent_text_within_phrase_id": inconsistent_text,
        "normalization_policy_frozen_before_inference": {
            "strict": "uppercase; split pipe-delimited segments and embedded whitespace; collapse only exact whole-sequence repetition",
            "content": "strict plus remove QUESTION, strip trailing question mark, map trailing _N variants to base, collapse exact whole-sequence repetition",
            "multi_reference_reduction": "report best reference and mean over signer references",
        },
        "important_limitations": [
            "Epee is a clip-level ASL corpus, not a purpose-built text-to-gloss translation benchmark.",
            "Parallel signer renderings are human ASL references but not independent adjudicated translations.",
            "QUESTION and BASE_N conventions differ from the How2Sign pseudo-gloss training convention.",
            "The second-register FOXTROT subset has one reference per source.",
        ],
    }

    if inconsistent_text:
        raise ValueError(f"Inconsistent English within phrase_id: {inconsistent_text[:3]}")
    write_jsonl(args.output_dir / "manifests" / "epee_sources_401.jsonl", sources)
    write_jsonl(args.output_dir / "manifests" / "epee_clips_1200.jsonl", clip_rows)
    write_json(args.output_dir / "reports" / "dataset_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

