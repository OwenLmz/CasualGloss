from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import nltk
from nltk.tokenize import TreebankWordDetokenizer


WAIT_SYMBOL = "@@"
DISPLAY_WAIT_SYMBOL = "<WAIT>"
OFFICIAL_TRANSLLAMA_COMMIT = "344659a79a328f29bd48615d3a3c6edd0ef1ea54"
OFFICIAL_SIMALIGN_COMMIT = "249a7f331814d18bef7f8ff69f8474a91568c2c2"
_DETOKENIZER = TreebankWordDetokenizer()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenize_words(text: str) -> list[str]:
    return nltk.word_tokenize(text.strip())


def detokenize_source(words: list[str]) -> str:
    text = _DETOKENIZER.detokenize([word for word in words if word != WAIT_SYMBOL])
    return re.sub(r"\s([,.!?;](?:\s|$))", r"\1", text).strip()


def detokenize_target(words: list[str]) -> str:
    return " ".join(words).strip()


def fix_alignment(alignment: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Official TransLLaMA one-to-one cleanup, preserving the latest edges."""
    seen_target: set[int] = set()
    reversed_unique_target: list[tuple[int, int]] = []
    for edge in reversed(list(alignment)):
        if edge[1] not in seen_target:
            reversed_unique_target.append(edge)
            seen_target.add(edge[1])
    unique_target = list(reversed(reversed_unique_target))

    seen_source: set[int] = set()
    reversed_unique_source: list[tuple[int, int]] = []
    for edge in reversed(unique_target):
        if edge[0] not in seen_source:
            reversed_unique_source.append(edge)
            seen_source.add(edge[0])
    return list(reversed(reversed_unique_source))


def align_pair(
    *,
    parent_id: str,
    split: str,
    source_text: str,
    target_text: str,
    aligner: Any,
    input_line_index: int,
    source_words: list[str] | None = None,
    target_words: list[str] | None = None,
) -> dict[str, Any]:
    source_words = list(source_words) if source_words is not None else tokenize_words(source_text)
    target_words = list(target_words) if target_words is not None else tokenize_words(target_text)
    if not source_words or not target_words:
        raise ValueError("empty source or target")

    alignment_output = aligner.get_word_aligns(source_words, target_words)
    raw_itermax = [tuple(map(int, edge)) for edge in alignment_output["itermax"]]
    fixed_edges = fix_alignment(raw_itermax)
    mutable_edges = [list(edge) for edge in fixed_edges]
    aligned_target = list(target_words)

    inserted_wait_count = 0
    for edge_index in range(len(mutable_edges)):
        while mutable_edges[edge_index][1] < mutable_edges[edge_index][0]:
            target_index = mutable_edges[edge_index][1]
            aligned_target.insert(target_index, WAIT_SYMBOL)
            inserted_wait_count += 1
            for later_index in range(edge_index, len(mutable_edges)):
                mutable_edges[later_index][1] += 1

    aligned_source = list(source_words)
    source_tail_filler_count = 0
    target_tail_wait_count = 0
    if len(aligned_target) > len(aligned_source):
        source_tail_filler_count = len(aligned_target) - len(aligned_source)
        aligned_source.extend([WAIT_SYMBOL] * source_tail_filler_count)
    elif len(aligned_target) < len(aligned_source):
        # Gloss targets often omit final punctuation. The official MT data
        # normally aligns sentence-final punctuation, so this adaptation makes
        # the implicit final READ positions explicit without adding glosses.
        target_tail_wait_count = len(aligned_source) - len(aligned_target)
        aligned_target.extend([WAIT_SYMBOL] * target_tail_wait_count)

    if len(aligned_source) != len(aligned_target):
        raise AssertionError("aligned source and target lengths differ")
    if any(word == WAIT_SYMBOL for word in source_words + target_words):
        raise ValueError("raw pair contains reserved WAIT symbol")

    return {
        "id": parent_id,
        "split": split,
        "input_line_index": input_line_index,
        "source_text_en": source_text,
        "target_pseudo_asl_gloss": target_text,
        "source_words": source_words,
        "target_words": target_words,
        "aligned_source_words": aligned_source,
        "aligned_target_words": aligned_target,
        "raw_itermax_alignment": [list(edge) for edge in raw_itermax],
        "fixed_alignment_before_wait_insertion": [list(edge) for edge in fixed_edges],
        "fixed_alignment_after_wait_insertion": mutable_edges,
        "alignment_method": "official_transllama_simalign_itermax_adapted_to_gloss",
        "alignment_model": "bert-base-multilingual-cased",
        "tokenization": "frozen_shared_manifest_tokens",
        "wait_symbol_internal": WAIT_SYMBOL,
        "inserted_wait_count": inserted_wait_count,
        "target_tail_wait_count": target_tail_wait_count,
        "source_tail_filler_count": source_tail_filler_count,
        "real_source_word_count": len(source_words),
        "target_gloss_token_count": len(target_words),
        "aligned_length": len(aligned_source),
    }


def make_prefix_example(row: dict[str, Any], source_prefix_length: int) -> dict[str, Any]:
    source_words = list(row["source_words"])
    target_words = list(row["target_words"])
    aligned_source = list(row["aligned_source_words"])
    aligned_target = list(row["aligned_target_words"])
    real_source_length = len(source_words)
    if source_prefix_length < 1 or source_prefix_length > real_source_length:
        raise ValueError("source_prefix_length outside real source")

    source_complete = source_prefix_length == real_source_length
    source_prefix_words = [
        word for word in aligned_source[:source_prefix_length] if word != WAIT_SYMBOL
    ]
    if source_complete:
        assistant_words = target_words
    else:
        aligned_prefix = aligned_target[:source_prefix_length]
        ends_in_wait = bool(aligned_prefix) and aligned_prefix[-1] == WAIT_SYMBOL
        assistant_words = [word for word in aligned_prefix if word != WAIT_SYMBOL]
        if ends_in_wait:
            assistant_words.append(WAIT_SYMBOL)
    if not assistant_words:
        assistant_words = [WAIT_SYMBOL]

    return {
        "id": f"{row['id']}__prefix_{source_prefix_length:04d}",
        "parent_id": row["id"],
        "split": row["split"],
        "source_prefix_length": source_prefix_length,
        "source_word_count": real_source_length,
        "source_complete": source_complete,
        "source_prefix_text": detokenize_source(source_prefix_words),
        "assistant_target_text": detokenize_target(assistant_words),
        "assistant_target_words": assistant_words,
        "full_target_words": target_words,
    }
