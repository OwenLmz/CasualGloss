#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import nltk
import simalign
from simalign import SentenceAligner

from transllama_data import (
    OFFICIAL_SIMALIGN_COMMIT,
    OFFICIAL_TRANSLLAMA_COMMIT,
    align_pair,
    file_sha256,
    write_json,
)


ALLOWED_INPUT_FIELDS = {
    "id",
    "split",
    "source_text_en",
    "source_tokens",
    "target_pseudo_asl_gloss",
    "target_pseudo_asl_gloss_tokens",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-split", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard index")
    if args.expected_split == "test":
        raise SystemExit("locked test alignment requires the separate release command")

    input_rows: list[tuple[int, dict]] = []
    with args.input.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if line_index % args.num_shards != args.shard_index:
                continue
            if args.limit is not None and len(input_rows) >= args.limit:
                break
            raw = json.loads(line)
            row = {field: raw.get(field) for field in ALLOWED_INPUT_FIELDS}
            if row["split"] != args.expected_split:
                raise ValueError(f"unexpected split for {row['id']}: {row['split']}")
            input_rows.append((line_index, row))

    aligner = SentenceAligner(
        model="bert",
        token_type="bpe",
        matching_methods="mai",
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures: list[dict] = []
    counts = {
        "input": len(input_rows),
        "written": 0,
        "failed": 0,
        "inserted_wait": 0,
        "target_tail_wait": 0,
        "source_tail_filler": 0,
    }
    with args.output.open("w", encoding="utf-8") as output_handle:
        for completed, (line_index, row) in enumerate(input_rows, start=1):
            try:
                aligned = align_pair(
                    parent_id=str(row["id"]),
                    split=str(row["split"]),
                    source_text=str(row["source_text_en"]),
                    target_text=str(row["target_pseudo_asl_gloss"]),
                    source_words=list(row["source_tokens"]),
                    target_words=list(row["target_pseudo_asl_gloss_tokens"]),
                    aligner=aligner,
                    input_line_index=line_index,
                )
                output_handle.write(json.dumps(aligned, ensure_ascii=False) + "\n")
                counts["written"] += 1
                counts["inserted_wait"] += aligned["inserted_wait_count"]
                counts["target_tail_wait"] += aligned["target_tail_wait_count"]
                counts["source_tail_filler"] += aligned["source_tail_filler_count"]
            except Exception as error:
                failures.append(
                    {"id": row.get("id"), "input_line_index": line_index, "error": repr(error)}
                )
                counts["failed"] += 1
            if completed % 100 == 0:
                print(f"aligned {completed}/{len(input_rows)}", flush=True)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "independent_official_transllama_alignment_not_gold_asl",
        "input": str(args.input),
        "output": str(args.output),
        "expected_split": args.expected_split,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "limit": args.limit,
        "counts": counts,
        "failures": failures,
        "versions": {
            "python": platform.python_version(),
            "nltk": nltk.__version__,
            "simalign": getattr(simalign, "__version__", "0.3-fixed-commit"),
            "official_transllama_commit": OFFICIAL_TRANSLLAMA_COMMIT,
            "official_simalign_commit": OFFICIAL_SIMALIGN_COMMIT,
        },
        "sha256": {
            "input": file_sha256(args.input),
            "output": file_sha256(args.output),
        },
        "existing_alignment_fields_consumed": False,
        "tokenization": "frozen_shared_manifest_tokens",
    }
    write_json(args.report, report)
    if failures:
        raise SystemExit(f"alignment failures: {len(failures)}")


if __name__ == "__main__":
    main()
