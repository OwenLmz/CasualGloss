#!/usr/bin/env python3
"""Verify paper-only source provenance, syntax, and bounded release hygiene."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = [
    re.compile(rb"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(rb"\bhf_[A-Za-z0-9]{25,}\b"),
    re.compile(rb"\bAKIA[A-Z0-9]{16}\b"),
]


def source_expectation(entry: dict) -> tuple[str, str, int]:
    relative = entry["repository_path"]
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or "\\" in relative or ":" in relative:
        raise ValueError("repository_path must be a safe repository-relative POSIX path")
    return (
        relative,
        entry.get("repository_sha256", entry["sha256"]),
        entry.get("repository_size_bytes", entry["size_bytes"]),
    )


def sensitive_text_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"source_text_en", "target_pseudo_asl_gloss", "predicted_pseudo_asl_gloss", "text_en"} and isinstance(item, (str, list)):
                yield key
            yield from sensitive_text_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from sensitive_text_keys(item)


def main() -> int:
    failures = []
    manifest_path = ROOT / "provenance/source_manifest.json"
    try:
        record = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        print(json.dumps({"passed": False, "failures": [f"source manifest: {type(exc).__name__}"]}))
        return 1
    seen = set()
    for entry in record["files"]:
        try:
            relative, digest, size = source_expectation(entry)
        except (KeyError, ValueError) as exc:
            failures.append(f"invalid source manifest entry: {type(exc).__name__}")
            continue
        if relative in seen:
            failures.append(f"duplicate source destination: {relative}")
        seen.add(relative)
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing source: {relative}")
            continue
        data = path.read_bytes()
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            failures.append(f"source integrity mismatch: {relative}")
        if digest != entry["sha256"] and not entry.get("modifications"):
            failures.append(f"adapted source lacks modification record: {relative}")
    parsed = scanned = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or {".git", "__pycache__"} & set(path.parts):
            continue
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() in {".safetensors", ".wav", ".mp4", ".pem", ".key", ".jsonl", ".parquet"} or path.name.startswith(".env"):
            failures.append(f"data/credential asset requires separate review: {relative}")
        if path.stat().st_size > 20_000_000:
            failures.append(f"large asset requires separate review: {relative}")
        data = path.read_bytes()
        scanned += 1
        if any(pattern.search(data) for pattern in SECRET_PATTERNS):
            failures.append(f"credential-like pattern; inspect privately: {relative}")
        if path.suffix == ".json":
            try:
                obj = json.loads(data.decode("utf-8-sig"))
                keys = sorted(set(sensitive_text_keys(obj)))
                if keys:
                    failures.append(f"sample-text keys require review: {relative}: {','.join(keys)}")
            except (UnicodeError, ValueError):
                failures.append(f"invalid JSON: {relative}")
        if path.suffix == ".py":
            try:
                ast.parse(data.decode("utf-8-sig"), filename=relative)
                parsed += 1
            except (SyntaxError, UnicodeError):
                failures.append(f"Python syntax failure: {relative}")
    from paper import ENTRIES
    missing_entries = [name for name, entry in ENTRIES.items() if not (ROOT / entry.path).is_file()]
    if missing_entries:
        failures.append("missing dispatcher entries: " + ", ".join(missing_entries))
    print(json.dumps({"source_count": len(record["files"]), "python_ast_count": parsed,
                      "scanned_files": scanned, "failures": failures, "passed": not failures}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
