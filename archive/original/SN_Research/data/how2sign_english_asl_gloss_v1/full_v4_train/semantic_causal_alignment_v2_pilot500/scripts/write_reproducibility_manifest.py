#!/usr/bin/env python3
"""Hash the v2 pilot's reproducible core artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    excluded_names = {
        args.output.name,
        "smoke_parent_001.jsonl",
        "smoke_scores_001.jsonl",
        "smoke_scores_001_report.json",
    }
    files = []
    for path in sorted(args.root.rglob("*")):
        if not path.is_file() or path.name in excluded_names or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(args.root).as_posix()
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    value = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_root": str(args.root),
        "file_count": len(files),
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"file_count": len(files), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
