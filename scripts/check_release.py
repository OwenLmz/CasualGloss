#!/usr/bin/env python3
"""Check original source bytes, syntax and release hygiene without importing ML."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = [
    re.compile(rb"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(rb"\bhf_[A-Za-z0-9]{25,}\b"),
    re.compile(rb"\bAKIA[A-Z0-9]{16}\b"),
]


def main() -> int:
    failures: list[str] = []
    record = json.loads((ROOT / "provenance/source_manifest.json").read_text(encoding="utf-8"))
    for entry in record["files"]:
        rel = entry["archive_path"]
        path = ROOT / rel
        if not path.is_file():
            failures.append(f"missing original: {rel}")
            continue
        data = path.read_bytes()
        if len(data) != entry["size_bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            failures.append(f"original bytes changed: {rel}")
    parsed = 0
    scanned = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() in {".safetensors", ".wav", ".mp4", ".pem", ".key", ".jsonl", ".parquet"} or path.name.startswith(".env"):
            failures.append(f"forbidden release asset: {rel}")
        if path.stat().st_size > 20_000_000:
            failures.append(f"large file requires external-artifact review: {rel}")
        data = path.read_bytes()
        scanned += 1
        if any(pattern.search(data) for pattern in SECRET_PATTERNS):
            # Never print credential contents.
            failures.append(f"credential-like pattern, inspect privately: {rel}")
        if path.suffix == ".json":
            try:
                obj = json.loads(data.decode("utf-8-sig"))
                def text_keys(value):
                    if isinstance(value, dict):
                        for key, item in value.items():
                            if key in {"source_text_en", "target_pseudo_asl_gloss", "predicted_pseudo_asl_gloss", "text_en"} and isinstance(item, (str, list)):
                                yield key
                            yield from text_keys(item)
                    elif isinstance(value, list):
                        for item in value:
                            yield from text_keys(item)
                keys = sorted(set(text_keys(obj)))
                if keys:
                    failures.append(f"source-text keys require private-data review: {rel}: {','.join(keys)}")
            except (UnicodeError, json.JSONDecodeError):
                failures.append(f"invalid JSON: {rel}")
        if path.suffix == ".py":
            try:
                ast.parse(data.decode("utf-8-sig"), filename=rel)
                parsed += 1
            except (SyntaxError, UnicodeError) as exc:
                failures.append(f"Python syntax: {rel}: {type(exc).__name__}")
    print(json.dumps({"original_count": len(record["files"]), "python_ast_count": parsed,
                      "scanned_files": scanned, "failures": failures, "passed": not failures}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
