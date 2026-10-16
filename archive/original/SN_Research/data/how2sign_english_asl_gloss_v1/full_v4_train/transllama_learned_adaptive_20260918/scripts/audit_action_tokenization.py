#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from transllama_data import read_jsonl, write_json


GLOSS_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9'/_-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-max-new-tokens", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    config = json.loads(args.config.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        local_files_only=True,
        trust_remote_code=False,
    )

    rows = read_jsonl(args.pairs)
    maximum = 0
    maximum_examples: list[dict[str, object]] = []
    over_limit: list[dict[str, object]] = []
    invalid_glosses: list[dict[str, str]] = []
    token_count = 0
    for row in rows:
        for gloss in row["target_pseudo_asl_gloss_tokens"]:
            token_count += 1
            if not GLOSS_PATTERN.fullmatch(gloss):
                invalid_glosses.append({"id": str(row["id"]), "gloss": gloss})
            lengths = {
                "initial": len(tokenizer.encode(gloss, add_special_tokens=False)),
                "continuation": len(tokenizer.encode(" " + gloss, add_special_tokens=False)),
            }
            length = max(lengths.values())
            example = {
                "id": str(row["id"]),
                "gloss": gloss,
                "token_lengths": lengths,
            }
            if length > maximum:
                maximum = length
                maximum_examples = [example]
            elif length == maximum and len(maximum_examples) < 20:
                maximum_examples.append(example)
            if length > args.action_max_new_tokens:
                over_limit.append(example)

    report = {
        "pair_count": len(rows),
        "gloss_token_count": token_count,
        "action_max_new_tokens": args.action_max_new_tokens,
        "maximum_qwen_subtokens_per_gloss": maximum,
        "maximum_examples": maximum_examples,
        "over_limit_count": len(over_limit),
        "over_limit_examples": over_limit[:20],
        "invalid_gloss_count": len(invalid_glosses),
        "invalid_gloss_examples": invalid_glosses[:20],
        "passed": not over_limit and not invalid_glosses,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit("action-tokenization audit failed")


if __name__ == "__main__":
    main()
