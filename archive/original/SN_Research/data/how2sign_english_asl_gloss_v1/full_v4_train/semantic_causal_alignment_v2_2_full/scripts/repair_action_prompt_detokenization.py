#!/usr/bin/env python3
"""Make v2.2 action prompts use the same Treebank detokenizer as evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nltk.tokenize import TreebankWordDetokenizer


DETOKENIZER = TreebankWordDetokenizer()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def state_prompt(partial_source: str, committed: str, complete: bool) -> str:
    return (
        "Observed English prefix:\n"
        + partial_source
        + "\n\nAlready committed pseudo ASL gloss:\n"
        + (committed if committed else "<EMPTY>")
        + "\n\nSource status: "
        + ("COMPLETE" if complete else "INCOMPLETE")
        + "\nReturn exactly one next action."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--output-actions", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    pair_tokens: dict[str, list[str]] = {
        row["id"]: row["source_tokens"] for row in read_jsonl(args.pairs)
    }
    args.output_actions.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    changed_partial_count = 0
    changed_prompt_count = 0
    missing_parent_ids = []
    changed_examples = []
    with args.output_actions.open("w", encoding="utf-8", newline="\n") as output:
        for row in read_jsonl(args.actions):
            row_count += 1
            source_tokens = pair_tokens.get(row["parent_id"])
            if source_tokens is None:
                missing_parent_ids.append(row["parent_id"])
                continue
            prefix_length = int(row["source_prefix_length"])
            partial_source = DETOKENIZER.detokenize(source_tokens[:prefix_length])
            repaired_prompt = state_prompt(
                partial_source,
                row.get("committed_pseudo_asl_gloss", ""),
                bool(row["source_complete"]),
            )
            partial_changed = partial_source != row.get("partial_source_text_en", "")
            prompt_changed = repaired_prompt != row.get("state_prompt", "")
            changed_partial_count += partial_changed
            changed_prompt_count += prompt_changed
            if (partial_changed or prompt_changed) and len(changed_examples) < 20:
                changed_examples.append(
                    {
                        "id": row["id"],
                        "before_partial_source": row.get("partial_source_text_en", ""),
                        "after_partial_source": partial_source,
                    }
                )
            repaired = dict(row)
            repaired["partial_source_text_en"] = partial_source
            repaired["state_prompt"] = repaired_prompt
            repaired["prompt_detokenization"] = "nltk_treebank_word_detokenizer"
            repaired["alignment_version"] = (
                "semantic_causal_alignment_v2_2_1_treebank_prompt_fix"
            )
            output.write(json.dumps(repaired, ensure_ascii=False) + "\n")

    report: dict[str, Any] = {
        "scientific_status": "deterministic_prompt_format_repair",
        "pairs": str(args.pairs),
        "actions": str(args.actions),
        "output_actions": str(args.output_actions),
        "input_row_count": row_count,
        "output_row_count": row_count - len(missing_parent_ids),
        "changed_partial_source_count": changed_partial_count,
        "changed_state_prompt_count": changed_prompt_count,
        "missing_parent_id_count": len(missing_parent_ids),
        "missing_parent_id_preview": missing_parent_ids[:20],
        "changed_examples": changed_examples,
        "detokenizer": "nltk.tokenize.TreebankWordDetokenizer",
        "ok": not missing_parent_ids,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
