#!/usr/bin/env python3
"""Create frozen Qwen-v4 requests from the source-level Epee manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epee-manifest", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=401)
    args = parser.parse_args()

    sources = read_jsonl(args.epee_manifest)
    if len(sources) != args.expected_count:
        raise SystemExit(f"Expected {args.expected_count} sources, found {len(sources)}")

    requests = []
    for index, source in enumerate(sources):
        source_id = str(source["id"])
        text = " ".join(str(source["source_text_en"]).split())
        requests.append(
            {
                "request_id": source_id,
                "request_index": index,
                "normalized_english_key": normalize_key(text),
                "source_text_en": text,
                "epee_phrase_id": source["phrase_id"],
                "epee_dataset_commit": source["dataset_commit"],
            }
        )

    ids = [row["request_id"] for row in requests]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate Epee source IDs")
    keys = [row["normalized_english_key"] for row in requests]
    if len(keys) != len(set(keys)):
        raise SystemExit("Epee source texts are not unique after normalization")

    write_jsonl(args.output_jsonl, requests)
    report = {
        "scientific_status": "frozen_external_human_gloss_generator_validation",
        "source_count": len(sources),
        "request_count": len(requests),
        "epee_manifest": str(args.epee_manifest),
        "epee_manifest_sha256": sha256(args.epee_manifest),
        "output_jsonl": str(args.output_jsonl),
        "dataset_commits": sorted({str(row["dataset_commit"]) for row in sources}),
        "no_training_or_tuning_on_epee": True,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
