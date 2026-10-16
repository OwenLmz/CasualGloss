#!/usr/bin/env python3
"""Export a Task 5 JSONL manifest to SeamlessStreaming TSV format."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.manifest.open(encoding="utf-8-sig")
        if line.strip()
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "audio", "tgt_text"],
            delimiter="\t",
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "audio": row["tts_audio_16k_mono"],
                    "tgt_text": row["reference_text_en"].replace("\t", " ").replace(
                        "\n", " "
                    ),
                }
            )
    print(json.dumps({"record_count": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
