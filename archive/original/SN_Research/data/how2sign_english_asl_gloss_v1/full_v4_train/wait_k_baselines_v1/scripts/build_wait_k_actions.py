#!/usr/bin/env python3
"""Build fixed Wait-k causal-action supervision from locked parent splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from nltk.tokenize.treebank import TreebankWordDetokenizer


DETOKENIZER = TreebankWordDetokenizer()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_prompt(source_tokens: list[str], prefix_length: int, committed: list[str]) -> str:
    source = DETOKENIZER.detokenize(source_tokens[:prefix_length])
    gloss = " ".join(committed) if committed else "<EMPTY>"
    status = "COMPLETE" if prefix_length >= len(source_tokens) else "INCOMPLETE"
    return (
        f"Observed English prefix:\n{source}\n\n"
        f"Already committed pseudo ASL gloss:\n{gloss}\n\n"
        f"Source status: {status}\n"
        "Return exactly one next action."
    )


def build_schedule(source_count: int, gloss_tokens: list[str], wait_k: int) -> list[dict[str, Any]]:
    if source_count < 1:
        raise ValueError("Wait-k requires at least one source token")
    prefix_length = 1
    schedule: list[dict[str, Any]] = []
    for gloss_index, gloss_token in enumerate(gloss_tokens):
        required = min(source_count, wait_k + gloss_index)
        while prefix_length < required:
            schedule.append(
                {
                    "source_prefix_length": prefix_length,
                    "action": "<WAIT>",
                    "action_type": "READ_WAIT",
                    "target_gloss_index": None,
                    "required_source_position": None,
                }
            )
            prefix_length += 1
        schedule.append(
            {
                "source_prefix_length": prefix_length,
                "action": gloss_token,
                "action_type": "WRITE_GLOSS",
                "target_gloss_index": gloss_index,
                "required_source_position": required,
            }
        )
    while prefix_length < source_count:
        schedule.append(
            {
                "source_prefix_length": prefix_length,
                "action": "<WAIT>",
                "action_type": "READ_WAIT",
                "target_gloss_index": None,
                "required_source_position": None,
            }
        )
        prefix_length += 1
    schedule.append(
        {
            "source_prefix_length": source_count,
            "action": "<END>",
            "action_type": "END",
            "target_gloss_index": None,
            "required_source_position": None,
        }
    )
    return schedule


def build_action_rows(parent: dict[str, Any], split: str, wait_k: int) -> list[dict[str, Any]]:
    source_tokens = list(parent["source_tokens"])
    gloss_tokens = list(parent["target_pseudo_asl_gloss_tokens"])
    schedule = build_schedule(len(source_tokens), gloss_tokens, wait_k)
    committed: list[str] = []
    rows: list[dict[str, Any]] = []
    for action_index, event in enumerate(schedule):
        prefix_length = int(event["source_prefix_length"])
        rows.append(
            {
                "id": f"{parent['id']}__wait_k{wait_k}_action_{action_index:03d}",
                "parent_id": parent["id"],
                "split": split,
                "state_prompt": state_prompt(source_tokens, prefix_length, committed),
                "target_action": event["action"],
                "target_action_type": event["action_type"],
                "source_prefix_length": prefix_length,
                "source_token_count": len(source_tokens),
                "source_complete": prefix_length >= len(source_tokens),
                "required_source_position": event["required_source_position"],
                "partial_source_text_en": DETOKENIZER.detokenize(source_tokens[:prefix_length]),
                "committed_pseudo_asl_gloss": " ".join(committed),
                "parent_target_pseudo_asl_gloss": parent["target_pseudo_asl_gloss"],
                "wait_k": wait_k,
                "policy_family": "fixed_wait_k_prefix_to_prefix",
                "alignment_version": f"wait_k{wait_k}_actions_v1",
                "transition_semantics": "WRITE_keeps_prefix_WAIT_advances_prefix",
                "prompt_detokenization": "nltk_treebank_word_detokenizer",
                "target_is_gold": False,
                "alignment_is_gold": False,
            }
        )
        if event["action_type"] == "WRITE_GLOSS":
            committed.append(str(event["action"]))
    return rows


def selected_rows(path: Path, limit: int) -> Iterable[dict[str, Any]]:
    for index, row in enumerate(read_jsonl(path)):
        if limit >= 0 and index >= limit:
            break
        yield row


def build_split(
    input_path: Path,
    output_path: Path,
    split: str,
    wait_k: int,
    parent_limit: int,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parent_count = 0
    action_count = 0
    action_types: Counter[str] = Counter()
    source_lengths: list[int] = []
    gloss_lengths: list[int] = []
    write_positions: list[int] = []
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for parent in selected_rows(input_path, parent_limit):
            rows = build_action_rows(parent, split, wait_k)
            parent_count += 1
            action_count += len(rows)
            source_lengths.append(len(parent["source_tokens"]))
            gloss_lengths.append(len(parent["target_pseudo_asl_gloss_tokens"]))
            for row in rows:
                action_types[row["target_action_type"]] += 1
                if row["target_action_type"] == "WRITE_GLOSS":
                    write_positions.append(int(row["source_prefix_length"]))
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "input": str(input_path),
        "output": str(output_path),
        "parent_limit": parent_limit,
        "parent_count": parent_count,
        "action_count": action_count,
        "action_type_counts": dict(sorted(action_types.items())),
        "mean_source_tokens": round(sum(source_lengths) / max(1, len(source_lengths)), 6),
        "mean_gloss_tokens": round(sum(gloss_lengths) / max(1, len(gloss_lengths)), 6),
        "mean_write_source_position": round(sum(write_positions) / max(1, len(write_positions)), 6),
        "sha256": sha256(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--wait-k", type=int, required=True)
    parser.add_argument("--scale-name", required=True)
    parser.add_argument("--train-limit", type=int, default=-1)
    parser.add_argument("--validation-limit", type=int, default=-1)
    parser.add_argument("--test-limit", type=int, default=-1)
    args = parser.parse_args()
    if args.wait_k < 1:
        raise SystemExit("--wait-k must be >= 1")

    destination = args.output_root / f"k{args.wait_k}" / args.scale_name
    limits = {
        "train": args.train_limit,
        "validation": args.validation_limit,
        "test": args.test_limit,
    }
    split_reports = {}
    for split, limit in limits.items():
        split_reports[split] = build_split(
            args.pairs_root / f"{split}_pairs.jsonl",
            destination / f"{split}_actions.jsonl",
            split,
            args.wait_k,
            limit,
        )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "fixed_wait_k_pseudo_gloss_action_supervision_not_gold_asl",
        "policy": {
            "name": "wait-k",
            "wait_k": args.wait_k,
            "definition": "g(t)=min(k+t-1, source_length), with one-indexed target t",
            "initial_source_prefix_tokens": 1,
            "transition_semantics": "WRITE_keeps_prefix_WAIT_advances_prefix",
            "end_condition": "source_complete_and_reference_gloss_complete",
        },
        "scale_name": args.scale_name,
        "splits": split_reports,
    }
    report_path = destination / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
