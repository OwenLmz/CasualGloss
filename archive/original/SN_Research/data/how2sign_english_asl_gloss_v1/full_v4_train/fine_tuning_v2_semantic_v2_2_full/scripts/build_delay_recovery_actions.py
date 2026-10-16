#!/usr/bin/env python3
"""Create off-policy WRITE states that teach recovery from accumulated delay.

For selected gloss positions, the source prefix is advanced beyond the v2.2
scheduled emission point while the committed history is deliberately left
behind. The correct action remains the overdue gloss token. These examples teach
the model to catch up with WRITE bursts after an unnecessary WAIT.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from nltk.tokenize import TreebankWordDetokenizer


DETOKENIZER = TreebankWordDetokenizer()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def detokenize(tokens: list[str]) -> str:
    return DETOKENIZER.detokenize(tokens)


def selected_target_indices(target_count: int, states_per_parent: int) -> list[int]:
    if target_count <= 0:
        return []
    if states_per_parent >= target_count:
        return list(range(target_count))
    if states_per_parent == 1:
        return [target_count // 2]
    return sorted(
        {
            min(target_count - 1, round(index * (target_count - 1) / (states_per_parent - 1)))
            for index in range(states_per_parent)
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-actions", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--states-per-parent", type=int, default=3)
    parser.add_argument("--delay-offsets", type=int, nargs="+", default=[1, 3])
    args = parser.parse_args()

    output: list[dict[str, Any]] = []
    delay_counts = Counter()
    skipped_at_source_end = 0
    selected_parent_count = 0
    for pair in read_jsonl(args.pairs):
        if pair.get("split", "train") != args.split:
            continue
        selected_parent_count += 1
        source_tokens = pair["source_tokens"]
        target_tokens = pair["target_pseudo_asl_gloss_tokens"]
        positions = [
            int(item.get("v2_2_emit_source_position", item["scheduled_emit_source_position"]))
            for item in pair["semantic_gloss_token_alignment"]
        ]
        for rank, target_index in enumerate(
            selected_target_indices(len(target_tokens), args.states_per_parent)
        ):
            scheduled_position = positions[target_index]
            delay = args.delay_offsets[rank % len(args.delay_offsets)]
            recovery_position = min(len(source_tokens), scheduled_position + delay)
            if recovery_position <= scheduled_position:
                skipped_at_source_end += 1
                continue
            committed = target_tokens[:target_index]
            partial_source = detokenize(source_tokens[:recovery_position])
            output.append(
                {
                    "id": (
                        f"{pair['id']}__semantic_v2_2_delay_recovery_"
                        f"{target_index:03d}_d{delay}"
                    ),
                    "parent_id": pair["id"],
                    "split": pair.get("split", "train"),
                    "state_prompt": (
                        "Observed English prefix:\n"
                        + partial_source
                        + "\n\nAlready committed pseudo ASL gloss:\n"
                        + (" ".join(committed) if committed else "<EMPTY>")
                        + "\n\nSource status: "
                        + ("COMPLETE" if recovery_position >= len(source_tokens) else "INCOMPLETE")
                        + "\nReturn exactly one next action."
                    ),
                    "target_action": target_tokens[target_index],
                    "target_action_type": "WRITE_DELAY_RECOVERY",
                    "source_prefix_length": recovery_position,
                    "source_token_count": len(source_tokens),
                    "source_complete": recovery_position >= len(source_tokens),
                    "partial_source_text_en": partial_source,
                    "committed_pseudo_asl_gloss": " ".join(committed),
                    "parent_target_pseudo_asl_gloss": pair["target_pseudo_asl_gloss"],
                    "target_is_gold": False,
                    "alignment_is_gold": False,
                    "alignment_version": "semantic_causal_alignment_v2_2_delay_recovery_v1",
                    "transition_semantics": "WRITE_keeps_prefix_WAIT_advances_prefix",
                    "augmentation_type": "off_policy_delay_recovery_write",
                    "target_gloss_index": target_index,
                    "original_scheduled_emit_source_position": scheduled_position,
                    "recovery_source_prefix_length": recovery_position,
                    "extra_read_delay_tokens": recovery_position - scheduled_position,
                    "method_note": (
                        "The source is deliberately advanced while committed gloss remains behind; "
                        "the model must WRITE the overdue next gloss token."
                    ),
                }
            )
            delay_counts[recovery_position - scheduled_position] += 1

    write_jsonl(args.output_actions, output)
    report = {
        "scientific_status": "off_policy_delay_recovery_over_pseudo_gloss_not_gold_asl",
        "source_pairs": str(args.pairs),
        "split": args.split,
        "selected_parent_count": selected_parent_count,
        "recovery_action_count": len(output),
        "states_per_parent": args.states_per_parent,
        "requested_delay_offsets": args.delay_offsets,
        "realized_extra_delay_counts": dict(delay_counts),
        "skipped_because_original_emit_at_source_end": skipped_at_source_end,
        "target_action_type": "WRITE_DELAY_RECOVERY",
        "important_note": (
            "This is optional continuation-training augmentation inspired by delay-recovery SFT. "
            "It is not included in the primary v2.2 experiment and must be evaluated separately."
        ),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
