#!/usr/bin/env python3
"""List or materialize explicitly selected research artifacts from an authorized server.

No default bulk download. Does not fetch audio directories, overwrite files, or
relicense research assets. Existing authorized SSH authentication stays external.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def selected_entries(category=None, role=None):
    manifest = json.loads((ROOT / "provenance/artifact_manifest.json").read_text(encoding="utf-8"))
    return [entry for entry in manifest["entries"] if
            (category is None or entry["category"] == category) and
            (role is None or entry.get("role") == role or role in entry.get("additional_roles", []))]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["list", "fetch"])
    p.add_argument("--category")
    p.add_argument("--role")
    p.add_argument("--host")
    p.add_argument("--port", type=int, default=22)
    p.add_argument("--ssh-key", type=Path)
    p.add_argument("--destination", type=Path)
    p.add_argument("--max-bytes", type=int, default=100_000_000)
    p.add_argument("--accept-data-terms", action="store_true")
    args = p.parse_args()
    if not args.category and not args.role:
        p.error("Select an exact --category or --role; no unbounded operation")
    entries = selected_entries(args.category, args.role)
    if not entries:
        p.error("No matching registered artifacts")
    if args.mode == "list":
        print(json.dumps({"count": len(entries), "total_file_bytes": sum(e.get("size_bytes", 0) for e in entries if e["type"] == "file"), "entries": entries}, indent=2))
        return 0
    if not args.accept_data_terms:
        p.error("Review THIRD_PARTY.md and upstream terms, then supply --accept-data-terms")
    if not args.host or not re.fullmatch(r"[A-Za-z0-9_.@:-]+", args.host) or args.host.startswith("-"):
        p.error("A valid authorized --host USER@HOST is required")
    if args.destination is None or not args.ssh_key or not 1 <= args.port <= 65535:
        p.error("--destination, --ssh-key and valid --port are required")
    destination = args.destination.resolve()
    if destination == ROOT or ROOT in destination.parents:
        p.error("Use a PRIVATE destination outside the source repository")
    if any(e["type"] != "file" for e in entries):
        p.error("Directory transfer is deliberately unsupported; use the documented audio index")
    if sum(e["size_bytes"] for e in entries) > args.max_bytes:
        p.error("Selection exceeds --max-bytes; review the list and deliberately raise the limit if appropriate")
    ssh = ["ssh", "-i", str(args.ssh_key), "-p", str(args.port), "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=yes", args.host]
    completed = []
    for entry in entries:
        rel = PurePosixPath(entry["relative_to_remote_root"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe artifact path in registry")
        local = destination.joinpath(*rel.parts)
        resolved = local.resolve()
        if destination not in resolved.parents or resolved == ROOT or ROOT in resolved.parents:
            raise ValueError("Artifact destination escapes the private root through a link")
        expected = entry.get("sha256")
        if expected is None:
            # An un-hashed metadata entry is not silently treated as verified.
            response = subprocess.run(ssh + ["sha256sum " + shlex.quote(entry["remote_path"])], capture_output=True, text=True, check=True)
            expected = response.stdout.split()[0]
            if not re.fullmatch(r"[a-f0-9]{64}", expected):
                raise ValueError("Invalid remote SHA-256 response")
        if local.exists():
            if local.is_file() and local.stat().st_size == entry["size_bytes"] and digest(local) == expected:
                completed.append({"path": str(rel), "sha256": expected, "status": "already_verified"})
                continue
            raise FileExistsError(f"Refusing to overwrite nonmatching existing artifact: {local}")
        local.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["scp", "-i", str(args.ssh_key), "-P", str(args.port), "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=yes", args.host + ":" + entry["remote_path"], str(local)]
        subprocess.run(cmd, check=True)
        if local.stat().st_size != entry["size_bytes"] or digest(local) != expected:
            raise ValueError(f"Artifact verification failed; preserve for inspection, do not use: {local}")
        completed.append({"path": str(rel), "sha256": expected, "status": "downloaded_verified"})
        print("verified", rel)
    print(json.dumps({"verified_count": len(completed), "files": completed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
