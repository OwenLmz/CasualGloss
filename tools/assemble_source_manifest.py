"""Merge independently verified remote/local original manifests without changing originals."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PROV=ROOT/"provenance"

def main():
    target=PROV/"source_manifest.json"
    remote=json.loads(target.read_text("utf-8"))
    local=json.loads((PROV/"local_analysis_source_manifest.json").read_text("utf-8"))
    withheld_manifest=json.loads((PROV/"withheld_files.json").read_text("utf-8"))
    withheld={f["archive_path"] for f in withheld_manifest["files"]}
    for archive_path in withheld:
        if (ROOT/archive_path).exists():
            raise RuntimeError("Withheld case document must remain outside the public archive: "+archive_path)
    files=[f for f in remote["files"] if f.get("origin")!="local_workspace_original"
        and f["archive_path"] not in withheld]
    remote_count=len(files)
    remote_bytes=sum(f["size_bytes"] for f in files)
    for entry in local["entries"]:
        if entry["path"] in withheld:
            continue
        files.append({"relative_path":entry["relative_source_path"],
            "archive_path":entry["path"],"source_path":entry["source"],
            "sha256":entry["sha256"],"size_bytes":entry["size"],
            "category":entry["category"],"origin":"local_workspace_original"})
    for item in files:
        p=ROOT/item["archive_path"]
        raw=p.read_bytes()
        if len(raw)!=item["size_bytes"] or hashlib.sha256(raw).hexdigest()!=item["sha256"]:
            raise RuntimeError("Manifest integrity failure: "+item["archive_path"])
    remote.update(file_count=len(files),bytes=sum(f["size_bytes"] for f in files),files=files,
        source_roots=[{"origin":"authorized_remote_readonly_ssh","path":remote["source_root"]},
            {"origin":"local_workspace_original","path":local["source_workspace"]}],
        transport="remote_read_only_ssh_base64_and_local_byte_copy",
        remote_file_count=remote_count,remote_bytes=remote_bytes,
        local_file_count=local["file_count"],local_bytes=local["total_bytes"],
        public_source_only=True,withheld_file_count=len(withheld),
        preserved_original_total_count=len(files)+len(withheld),
        withheld_manifest="provenance/withheld_files.json",
        related_manifests=["provenance/source_allowlist.review.json",
            "provenance/local_analysis_source_manifest.json"])
    target.write_text(json.dumps(remote,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"total_originals":len(files),"bytes":remote["bytes"],
        "remote_originals":remote_count,"local_originals":local["file_count"],
        "integrity_failures":0}))

if __name__=="__main__":
    main()
