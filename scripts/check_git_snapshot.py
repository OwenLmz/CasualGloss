#!/usr/bin/env python3
"""Validate the staged Git blobs against the curated source manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from check_release import source_expectation

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    manifest = json.loads((ROOT / "provenance/source_manifest.json").read_text(encoding="utf-8-sig"))
    output = subprocess.run(["git", "ls-files", "--stage", "-z"], cwd=ROOT, check=True, capture_output=True).stdout
    by_path = {}
    for item in output.split(b"\0"):
        if item:
            metadata, name = item.split(b"\t", 1)
            mode, sha, stage = metadata.split()
            if stage != b"0":
                raise ValueError("Git index contains unresolved merge stages")
            by_path[name.decode("utf-8")] = sha.decode("ascii")
    expected = [source_expectation(entry) for entry in manifest["files"]]
    missing = [relative for relative, _, _ in expected if relative not in by_path]
    if missing:
        print(json.dumps({"passed": False, "missing_from_index": missing}, indent=2))
        return 1
    requests = "".join(by_path[relative] + "\n" for relative, _, _ in expected).encode("ascii")
    objects = subprocess.run(["git", "cat-file", "--batch"], cwd=ROOT, input=requests,
                             capture_output=True, check=True).stdout
    cursor, failures = 0, []
    for relative, digest, size in expected:
        end = objects.index(b"\n", cursor)
        header = objects[cursor:end].split()
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError("Unexpected Git object response")
        length = int(header[2])
        data = objects[end + 1:end + 1 + length]
        cursor = end + length + 2
        if length != size or hashlib.sha256(data).hexdigest() != digest:
            failures.append(relative)
    print(json.dumps({"git_index_sources_checked": len(expected), "byte_failures": failures,
                      "passed": not failures}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
