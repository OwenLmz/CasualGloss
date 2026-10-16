#!/usr/bin/env python3
"""Build one shard of corrected English-to-pseudo-ASL causal alignments."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordDetokenizer, TreebankWordTokenizer
from simalign import SentenceAligner


TOKENIZER = TreebankWordTokenizer()
DETOKENIZER = TreebankWordDetokenizer()
STEMMER = PorterStemmer()
PUNCTUATION_RE = re.compile(r"^[^A-Za-z0-9]+$")
GLOSS_PIECE_RE = re.compile(r"[A-Za-z0-9]+")

ALIASES = {
    "ca": "can",
    "wo": "will",
    "n't": "not",
    "nt": "not",
    "'re": "be",
    "re": "be",
    "'m": "be",
    "m": "be",
    "'ve": "have",
    "ve": "have",
    "'ll": "will",
    "ll": "will",
    "'d": "would",
    "d": "would",
    "'s": "be",
    "s": "be",
    "gonna": "go",
    "wanna": "want",
    "gotta": "get",
}

NUMBER_ALIASES = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_mean(values: list[float]) -> float | None:
    return round(mean(values), 6) if values else None


def normalize_lexeme(token: str) -> str:
    value = token.lower().replace("\u2019", "'").replace("\u2018", "'")
    value = re.sub(r"[^a-z0-9']+", "", value)
    value = ALIASES.get(value, value)
    return NUMBER_ALIASES.get(value, value)


def lexical_signature(token: str) -> tuple[str, str]:
    normalized = normalize_lexeme(token)
    return normalized, STEMMER.stem(normalized) if normalized else ""


def gloss_pieces(gloss_tokens: list[str]) -> tuple[list[str], list[int]]:
    pieces: list[str] = []
    owners: list[int] = []
    for gloss_index, token in enumerate(gloss_tokens):
        token_pieces = GLOSS_PIECE_RE.findall(token) or [token]
        for piece in token_pieces:
            pieces.append(piece)
            owners.append(gloss_index)
    return pieces, owners


def relative_position_choice(
    candidates: list[int], piece_index: int, piece_count: int, source_count: int
) -> int:
    if len(candidates) == 1:
        return candidates[0]
    expected = 0.0 if piece_count <= 1 else piece_index * (source_count - 1) / (piece_count - 1)
    return min(candidates, key=lambda index: (abs(index - expected), -index))


def build_hybrid_alignment(
    source_tokens: list[str], target_tokens: list[str], aligner: SentenceAligner
) -> dict[str, Any]:
    pieces, owners = gloss_pieces(target_tokens)
    source_signatures = [lexical_signature(token) for token in source_tokens]
    piece_signatures = [lexical_signature(piece) for piece in pieces]
    piece_alignments: list[dict[str, Any] | None] = [None] * len(pieces)

    for piece_index, (normalized, stemmed) in enumerate(piece_signatures):
        if not normalized:
            continue
        exact = [
            index
            for index, signature in enumerate(source_signatures)
            if signature[0] and signature[0] == normalized
        ]
        if exact:
            piece_alignments[piece_index] = {
                "source_index": relative_position_choice(
                    exact, piece_index, len(pieces), len(source_tokens)
                ),
                "method": "lexical_exact",
                "confidence": 1.0,
            }
            continue
        stem = [
            index
            for index, signature in enumerate(source_signatures)
            if stemmed and signature[1] == stemmed
        ]
        if stem:
            piece_alignments[piece_index] = {
                "source_index": relative_position_choice(
                    stem, piece_index, len(pieces), len(source_tokens)
                ),
                "method": "lexical_stem",
                "confidence": 0.9,
            }

    unresolved = [index for index, value in enumerate(piece_alignments) if value is None]
    simalign_error = None
    simalign_pairs: list[tuple[int, int]] = []
    if unresolved and source_tokens and pieces:
        try:
            result = aligner.get_word_aligns(source_tokens, pieces)
            simalign_pairs = sorted(tuple(pair) for pair in result.get("itermax", []))
        except Exception as error:
            simalign_error = f"{type(error).__name__}: {error}"

    target_to_sources: dict[int, list[int]] = {}
    for source_index, piece_index in simalign_pairs:
        if source_index >= len(source_tokens) or piece_index >= len(pieces):
            continue
        if PUNCTUATION_RE.fullmatch(source_tokens[source_index]):
            continue
        target_to_sources.setdefault(piece_index, []).append(source_index)
    for piece_index in unresolved:
        candidates = target_to_sources.get(piece_index, [])
        if candidates:
            piece_alignments[piece_index] = {
                "source_index": max(candidates),
                "method": "simalign_itermax",
                "confidence": 0.6,
            }

    piece_records: list[dict[str, Any]] = []
    for piece_index, piece in enumerate(pieces):
        alignment = piece_alignments[piece_index]
        record = {
            "piece_index": piece_index,
            "gloss_index": owners[piece_index],
            "gloss_piece": piece,
            "source_index": None,
            "source_token": None,
            "method": "unaligned",
            "confidence": 0.0,
        }
        if alignment is not None:
            source_index = int(alignment["source_index"])
            record.update(
                {
                    "source_index": source_index,
                    "source_token": source_tokens[source_index],
                    "method": alignment["method"],
                    "confidence": alignment["confidence"],
                }
            )
        piece_records.append(record)

    source_count = len(source_tokens)
    gloss_alignment: list[dict[str, Any]] = []
    evidence_positions: list[int] = []
    for gloss_index, gloss_token in enumerate(target_tokens):
        owned = [item for item in piece_records if item["gloss_index"] == gloss_index]
        if not owned or any(item["source_index"] is None for item in owned):
            evidence_position = source_count
            method = "unaligned_delayed_to_full_source"
            confidence = 0.0
        else:
            evidence_position = max(int(item["source_index"]) + 1 for item in owned)
            methods = {item["method"] for item in owned}
            if "simalign_itermax" in methods:
                method = "simalign_itermax"
            elif "lexical_stem" in methods:
                method = "lexical_stem"
            else:
                method = "lexical_exact"
            confidence = min(float(item["confidence"]) for item in owned)
        evidence_positions.append(evidence_position)
        gloss_alignment.append(
            {
                "gloss_index": gloss_index,
                "gloss_token": gloss_token,
                "evidence_source_position": evidence_position,
                "alignment_method": method,
                "alignment_confidence": confidence,
                "pieces": owned,
            }
        )

    emit_positions: list[int] = []
    running_max = 1
    for evidence_position in evidence_positions:
        running_max = max(running_max, evidence_position)
        emit_positions.append(running_max)
    for record, emit_position in zip(gloss_alignment, emit_positions):
        record["earliest_ordered_emit_source_position"] = emit_position

    return {
        "gloss_piece_alignment": piece_records,
        "gloss_token_alignment": gloss_alignment,
        "raw_evidence_positions": evidence_positions,
        "ordered_emit_positions": emit_positions,
        "simalign_pair_count": len(simalign_pairs),
        "simalign_error": simalign_error,
    }


def average_lagging(
    emit_positions: list[int], source_count: int, target_count: int
) -> tuple[float | None, float | None]:
    if not emit_positions or source_count <= 0 or target_count <= 0:
        return None, None
    ratio = target_count / source_count
    tau = next(
        (index for index, position in enumerate(emit_positions, 1) if position >= source_count),
        target_count,
    )
    lag_values = [
        emit_positions[target_index - 1] - (target_index - 1) / ratio
        for target_index in range(1, tau + 1)
    ]
    value = mean(lag_values)
    return round(value, 6), round(value / source_count, 6)


def build_timeline(
    source_tokens: list[str],
    gloss_tokens: list[str],
    evidence_positions: list[int],
    emit_positions: list[int],
    wait_token: str,
    end_token: str,
    parent_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    actions: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    target_index = 0
    source_count = len(source_tokens)

    def add_action(source_prefix_length: int, action: str, action_type: str) -> None:
        committed = [item["action"] for item in actions if item["action_type"].startswith("WRITE")]
        target_gloss_index = target_index if action_type.startswith("WRITE") else None
        required_position = (
            evidence_positions[target_index]
            if target_gloss_index is not None and target_index < len(evidence_positions)
            else None
        )
        action_index = len(actions)
        actions.append(
            {
                "action_index": action_index,
                "source_prefix_length": source_prefix_length,
                "action": action,
                "action_type": action_type,
                "target_gloss_index": target_gloss_index,
                "required_source_position": required_position,
            }
        )
        examples.append(
            {
                "example_id": f"{parent_id}__action_{action_index:03d}",
                "parent_id": parent_id,
                "action_index": action_index,
                "partial_source_text_en": DETOKENIZER.detokenize(
                    source_tokens[:source_prefix_length]
                ),
                "partial_source_tokens": source_tokens[:source_prefix_length],
                "committed_pseudo_asl_gloss": " ".join(committed),
                "committed_pseudo_asl_gloss_tokens": committed,
                "target_action": action,
                "action_type": action_type,
                "source_prefix_length": source_prefix_length,
                "source_token_count": source_count,
                "source_complete": source_prefix_length >= source_count,
                "target_pseudo_gloss_is_gold": False,
                "required_source_position": required_position,
            }
        )

    for source_prefix_length in range(1, source_count + 1):
        if target_index < len(gloss_tokens) and emit_positions[target_index] <= source_prefix_length:
            add_action(source_prefix_length, gloss_tokens[target_index], "WRITE_GLOSS")
            target_index += 1
        elif source_prefix_length == source_count and target_index == len(gloss_tokens):
            add_action(source_prefix_length, end_token, "END")
        else:
            add_action(source_prefix_length, wait_token, "READ_WAIT")

    while target_index < len(gloss_tokens):
        add_action(source_count, gloss_tokens[target_index], "WRITE_AFTER_SOURCE_END")
        target_index += 1
    if not actions or actions[-1]["action_type"] != "END":
        add_action(source_count, end_token, "END")
    return actions, examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-pairs", type=Path, required=True)
    parser.add_argument("--output-actions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    rows = read_jsonl(args.input)
    alignment_device = str(config.get("alignment_device", "cpu"))
    aligner = SentenceAligner(
        model="bert",
        token_type="bpe",
        matching_methods="i",
        device=alignment_device,
    )
    pair_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []

    for row_index, source_row in enumerate(rows, 1):
        parent_id = str(source_row.get("id") or source_row.get("request_id"))
        source_tokens = TOKENIZER.tokenize(source_row["source_text_en"])
        gloss_tokens = str(source_row["target_pseudo_asl_gloss"]).split()
        alignment = build_hybrid_alignment(source_tokens, gloss_tokens, aligner)
        actions, examples = build_timeline(
            source_tokens,
            gloss_tokens,
            alignment["raw_evidence_positions"],
            alignment["ordered_emit_positions"],
            config["wait_token"],
            config["end_token"],
            parent_id,
        )
        action_rows.extend(examples)

        method_counts = Counter(
            record["alignment_method"] for record in alignment["gloss_token_alignment"]
        )
        gloss_count = len(gloss_tokens)
        lexical_count = method_counts["lexical_exact"] + method_counts["lexical_stem"]
        simalign_count = method_counts["simalign_itermax"]
        unaligned_count = method_counts["unaligned_delayed_to_full_source"]
        wait_count = sum(action["action_type"] == "READ_WAIT" for action in actions)
        non_end_actions = max(1, len(actions) - 1)
        average_lagging_tokens, normalized_average_lagging = average_lagging(
            alignment["ordered_emit_positions"], len(source_tokens), gloss_count
        )
        metrics = {
            "source_token_count": len(source_tokens),
            "gloss_token_count": gloss_count,
            "action_count_including_end": len(actions),
            "wait_action_count": wait_count,
            "wait_action_ratio": round(wait_count / non_end_actions, 6),
            "lexically_aligned_gloss_token_count": lexical_count,
            "lexical_alignment_ratio": round(lexical_count / max(1, gloss_count), 6),
            "simalign_fallback_gloss_token_count": simalign_count,
            "simalign_fallback_ratio": round(simalign_count / max(1, gloss_count), 6),
            "unaligned_gloss_token_count": unaligned_count,
            "unaligned_gloss_token_ratio": round(unaligned_count / max(1, gloss_count), 6),
            "mean_ordered_emit_source_proportion": safe_mean(
                [position / max(1, len(source_tokens)) for position in alignment["ordered_emit_positions"]]
            ),
            "word_position_average_lagging_source_tokens": average_lagging_tokens,
            "word_position_normalized_average_lagging": normalized_average_lagging,
            "gloss_tokens_emitted_at_full_source_count": sum(
                position == len(source_tokens) for position in alignment["ordered_emit_positions"]
            ),
        }
        flags: list[str] = []
        if metrics["unaligned_gloss_token_ratio"] > 0.30:
            flags.append("high_unaligned_gloss_ratio")
        if metrics["lexical_alignment_ratio"] < 0.50:
            flags.append("low_lexical_alignment_ratio")
        if metrics["wait_action_ratio"] > 0.75:
            flags.append("excessive_wait_ratio")
        if metrics["gloss_tokens_emitted_at_full_source_count"] / max(1, gloss_count) > 0.50:
            flags.append("mostly_delayed_to_full_source")
        if len(actions) > 64:
            flags.append("very_long_action_sequence")
        if source_row.get("automatic_review_flagged"):
            flags.append("input_automatic_review_flagged")
        if alignment["simalign_error"]:
            flags.append("simalign_runtime_error")

        pair_rows.append(
            {
                "id": parent_id,
                "candidate_index": source_row["candidate_index"],
                "candidate_selection_bucket": source_row["candidate_selection_bucket"],
                "candidate_source_length_bin": source_row["candidate_source_length_bin"],
                "candidate_video_group_key": source_row["candidate_video_group_key"],
                "source_text_en": source_row["source_text_en"],
                "source_tokens": source_tokens,
                "target_pseudo_asl_gloss": source_row["target_pseudo_asl_gloss"],
                "target_pseudo_asl_gloss_tokens": gloss_tokens,
                "target_pseudo_gloss_is_gold": False,
                "source_dataset": source_row.get("source_dataset"),
                "how2sign_video_ids": source_row.get("how2sign_video_ids", []),
                "input_generation_status": source_row.get("generation_status"),
                "input_automatic_review_flags": source_row.get("automatic_review_flags", []),
                "alignment_method": config["alignment_method"],
                "alignment_model": config["alignment_model"],
                "timeline_policy": config["timeline_policy"],
                "gloss_piece_alignment": alignment["gloss_piece_alignment"],
                "gloss_token_alignment": alignment["gloss_token_alignment"],
                "causal_action_sequence": [action["action"] for action in actions],
                "causal_action_source_prefix_lengths": [
                    action["source_prefix_length"] for action in actions
                ],
                "causal_timeline": actions,
                "metrics": metrics,
                "causal_alignment_quality_flags": flags,
                "manual_review_status": "pending",
                "simalign_pair_count": alignment["simalign_pair_count"],
                "simalign_error": alignment["simalign_error"],
            }
        )
        if row_index % 50 == 0 or row_index == len(rows):
            print(f"[causal-align] {row_index}/{len(rows)}", flush=True)

    write_jsonl(args.output_pairs, pair_rows)
    write_jsonl(args.output_actions, action_rows)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": config["scientific_status"],
        "input_manifest": str(args.input),
        "output_pairs": str(args.output_pairs),
        "output_actions": str(args.output_actions),
        "record_count": len(pair_rows),
        "prefix_action_example_count": len(action_rows),
        "alignment_method": config["alignment_method"],
        "alignment_model": config["alignment_model"],
        "alignment_device": alignment_device,
        "timeline_policy": config["timeline_policy"],
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("simalign", "nltk", "transformers", "torch")
        },
        "selection_bucket_counts": dict(
            Counter(row["candidate_selection_bucket"] for row in pair_rows)
        ),
        "alignment_quality_flag_counts": dict(
            Counter(flag for row in pair_rows for flag in row["causal_alignment_quality_flags"])
        ),
        "alignment_method_token_counts": dict(
            Counter(
                token["alignment_method"]
                for row in pair_rows
                for token in row["gloss_token_alignment"]
            )
        ),
        "mean_lexical_alignment_ratio": safe_mean(
            [row["metrics"]["lexical_alignment_ratio"] for row in pair_rows]
        ),
        "mean_simalign_fallback_ratio": safe_mean(
            [row["metrics"]["simalign_fallback_ratio"] for row in pair_rows]
        ),
        "mean_unaligned_gloss_token_ratio": safe_mean(
            [row["metrics"]["unaligned_gloss_token_ratio"] for row in pair_rows]
        ),
        "mean_wait_action_ratio": safe_mean(
            [row["metrics"]["wait_action_ratio"] for row in pair_rows]
        ),
        "simalign_runtime_error_count": sum(bool(row["simalign_error"]) for row in pair_rows),
        "all_targets_marked_non_gold": all(
            row["target_pseudo_gloss_is_gold"] is False for row in pair_rows
        ),
        "ok": len(pair_rows) == len(rows) and not any(row["simalign_error"] for row in pair_rows),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
