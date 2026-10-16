#!/usr/bin/env python3
"""Verify archived source hashes from the staged Git objects, not just disk."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "provenance/source_manifest.json").read_text(encoding="utf-8"))
    tracked = subprocess.run(["git", "ls-files", "--stage", "-z"], cwd=ROOT, check=True, capture_output=True).stdout
    by_path = {}
    for item in tracked.split(b"\0"):
        if item:
            meta, name = item.split(b"\t", 1)
            by_path[name.decode("utf-8")] = meta.split()[1].decode("ascii")
    entries = manifest["files"]
    missing = [e["archive_path"] for e in entries if e["archive_path"] not in by_path]
    if missing:
        raise ValueError(f"Original files missing from index: {missing}")
    request = "".join(by_path[e["archive_path"]] + "\n" for e in entries).encode("ascii")
    objects = subprocess.run(["git", "cat-file", "--batch"], cwd=ROOT, input=request, capture_output=True, check=True).stdout
    cursor, failures = 0, []
    for entry in entries:
        end = objects.index(b"\n", cursor)
        header = objects[cursor:end].split()
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError("Unexpected Git object response")
        size = int(header[2])
        data = objects[end+1:end+1+size]
        cursor = end+1+size+1
        if len(data) != entry["size_bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            failures.append(entry["archive_path"])
    print(json.dumps({"git_index_originals_checked": len(entries), "byte_failures": failures, "passed": not failures}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
