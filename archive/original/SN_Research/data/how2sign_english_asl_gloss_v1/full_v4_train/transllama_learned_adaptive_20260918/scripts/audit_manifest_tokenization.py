#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transllama_data import tokenize_words


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--show", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = 0
    source_mismatches = 0
    target_mismatches = 0
    examples: list[dict] = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if args.limit is not None and total >= args.limit:
                break
            row = json.loads(line)
            source_nltk = tokenize_words(row["source_text_en"])
            target_nltk = tokenize_words(row["target_pseudo_asl_gloss"])
            source_manifest = list(row["source_tokens"])
            target_manifest = list(row["target_pseudo_asl_gloss_tokens"])
            source_differs = source_nltk != source_manifest
            target_differs = target_nltk != target_manifest
            source_mismatches += int(source_differs)
            target_mismatches += int(target_differs)
            if (source_differs or target_differs) and len(examples) < args.show:
                examples.append(
                    {
                        "id": row["id"],
                        "source_nltk": source_nltk,
                        "source_manifest": source_manifest,
                        "target_nltk": target_nltk,
                        "target_manifest": target_manifest,
                    }
                )
            total += 1

    print(
        json.dumps(
            {
                "rows": total,
                "source_mismatches": source_mismatches,
                "target_mismatches": target_mismatches,
                "examples": examples,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
