"""Collect reviewed small CausalGloss source files over read-only SSH.

The remote helper only enumerates or reads a fixed research allowlist. File
transport stays in memory; only the explicitly reviewed local allowlist is
written. It never starts training, modifies remote files, or copies datasets,
weights, environments, credentials, or third-party vendor trees.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from datetime import datetime, timezone

REMOTE_ROOT = "/123090274/SN_Research"
DATA = "data/how2sign_english_asl_gloss_v1"
FULL = DATA + "/full_v4_train"
FRONT = DATA + "/chinese_speech_frontend_v0"
RELEASE = Path(__file__).resolve().parents[1]
PROVENANCE = RELEASE / "provenance"
DEST = RELEASE / "archive/original/SN_Research"
ROOT_FILES = ["run_waitk_seed_20260914.sh", "queue_waitk5_after_k7_20260914.sh"]
FULL_DIRS = [
    "scripts", "configs", "prompts",
    "fine_tuning_v1", "causal_alignment_v1_corrected_full",
    "causal_alignment_v1_corrected_5k",
    "semantic_causal_alignment_v2_pilot500",
    "semantic_causal_alignment_v2_1_pilot500",
    "semantic_causal_alignment_v2_1_full",
    "semantic_causal_alignment_v2_2_full",
    "fine_tuning_v2_semantic_v2_2_full",
    "fine_tuning_v2_semantic_v2_1_full",
    "fine_tuning_v2_causal_full", "fine_tuning_v2_causal_full_seed2",
    "state_recovery_v2_3_full",
    "state_recovery_v2_3_multiseed_20260904",
    "state_recovery_v2_3_multiseed_20260905",
    "wait_k_baselines_v1", "wait_k_baselines_multiseed_20260914",
    "professor_feedback_experiments_20260912",
    "transllama_learned_adaptive_20260918",
    "clerc_epee_external_eval_v1",
    "clerc_epee_adaptive_comparison_20260920",
    "paper_experiments_20260903", "paper_multiseed_v2_3_20260905",
    "locked500_bleu_metric_v1_20260911",
    "local_agreement_causal_v2_3",
    "causal_alignment_latency_audit_v2_2",
    "causal_alignment_comparison_v1_v2_1_v2_2",
    "epee_pseudogloss_generator_validation_v1_20260911",
]
FRONT_DIRS = [
    "scripts", "configs", "patches", "reports",
    "paper_e2e_locked500_20260904", "paper_e2e_20260903",
    "chinese_speech_to_pseudo_asl_corpus_v1_20260905",
]
ALLOW_ROOTS = [FULL + "/" + name for name in FULL_DIRS]
ALLOW_ROOTS += [FRONT + "/" + name for name in FRONT_DIRS]
ALLOW_TOP = [FULL, FRONT]
DENIED_PARTS = {
    ".git", "__pycache__", "venv", "venvs", "envs", "_deps", "vendor",
    "third_party", "node_modules", "miniconda3", "huggingface_cache",
    "models", "audio", "videos", "downloads", "raw", "nltk_data",
    "logs", "pids", "checkpoints",
}
MAX_FILE = 512 * 1024
SENSITIVE_NAMES = re.compile(
    r"(^\.env($|\.)|credential|secret|private.?key|id_rsa|id_ed25519|"
    r"authorized_keys|known_hosts|\.pem$|\.key$)", re.I
)
SECRET_RE = re.compile(
    r"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|"
    r"AKIA[A-Z0-9]{16}|BEGIN [A-Z ]*PRIVATE KEY"
)

def ssh_python(script: str) -> dict:
    target = os.environ.get("CAUSALGLOSS_AUDIT_SSH_TARGET")
    identity = os.environ.get("CAUSALGLOSS_AUDIT_SSH_IDENTITY")
    port = os.environ.get("CAUSALGLOSS_AUDIT_SSH_PORT", "22")
    if not target or not identity:
        raise RuntimeError("Set CAUSALGLOSS_AUDIT_SSH_TARGET and CAUSALGLOSS_AUDIT_SSH_IDENTITY for the authorized server.")
    ssh = ["ssh", "-i", identity, "-p", port, "-o", "BatchMode=yes",
        "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=yes", target]
    proc = subprocess.run(
        ssh + ["python3", "-"], input=script.encode("utf-8"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180,
    )
    if proc.returncode:
        raise RuntimeError(
            "Read-only remote helper failed: " +
            proc.stderr.decode("utf-8", errors="replace")[:1000]
        )
    return json.loads(proc.stdout.decode("utf-8"))

def remote_prelude() -> str:
    settings = {
        "remote_root": REMOTE_ROOT, "allow_roots": ALLOW_ROOTS,
        "allow_top": ALLOW_TOP, "root_files": ROOT_FILES,
        "denied_parts": sorted(DENIED_PARTS), "max_file": MAX_FILE,
    }
    return (
        "import os,json,pathlib,hashlib,base64,re,datetime\n"
        "S=json.loads(" + repr(json.dumps(settings)) + ")\n"
        "ROOT=pathlib.Path(S['remote_root']).resolve()\n"
        "DENIED=set(S['denied_parts'])\n"
    )

def inventory_script() -> str:
    return remote_prelude() + """
def selected(p):
    rel=p.relative_to(ROOT)
    if any(x in DENIED for x in rel.parts):
        return False
    if p.suffix in (".py",".sh",".patch"):
        return True
    n=p.name.lower()
    if n.startswith(("readme","protocol","dataset_card","license","copying")):
        return p.suffix in (".md",".txt",".rst","")
    if p.suffix in (".md",".rst"):
        return True
    if n in ("adapter_config.json","selection.json","run_metadata.json","training_summary.json"):
        return True
    if p.suffix in (".json",".yaml",".yml",".toml",".txt"):
        return ("configs" in rel.parts or "prompts" in rel.parts or
                n.startswith("config_") or "protocol" in n or
                ("reports" in rel.parts and
                 ("environment" in n or "freeze" in n)))
    return False
paths=set()
missing=[]
for rel in S["allow_roots"]:
    d=ROOT/rel
    if not d.is_dir():
        missing.append(rel)
        continue
    for current,dirs,files in os.walk(d,followlinks=False):
        depth=len(pathlib.Path(current).relative_to(d).parts)
        dirs[:]=[n for n in dirs if n not in DENIED and depth<6
                 and not (pathlib.Path(current)/n).is_symlink()]
        for name in files:
            p=pathlib.Path(current)/name
            if p.is_symlink() or not selected(p):
                continue
            paths.add(p)
for rel in S["allow_top"]:
    d=ROOT/rel
    if d.is_dir():
        paths.update(p for p in d.iterdir() if p.is_file() and selected(p))
for rel in S["root_files"]:
    p=ROOT/rel
    if p.is_file():
        paths.add(p)
items=[]
oversize=[]
for p in sorted(paths):
    st=p.stat()
    record={"relative_path":p.relative_to(ROOT).as_posix(),
            "size_bytes":st.st_size,
            "mtime_utc":datetime.datetime.fromtimestamp(st.st_mtime,
                datetime.timezone.utc).isoformat()}
    if st.st_size>S["max_file"]:
        oversize.append(record)
    else:
        items.append(record)
print(json.dumps({"files":items,"missing_roots":missing,
                  "oversize_excluded":oversize},ensure_ascii=False))
"""

def categorize(path: str) -> str:
    if "transllama_learned_adaptive" in path:
        return "final_learned_baseline"
    if "clerc_epee_adaptive_comparison" in path:
        return "external_human_gloss_evaluation"
    if "professor_feedback_experiments" in path:
        return "component_ablation_and_historical_baseline"
    if "chinese_speech_to_pseudo_asl_corpus" in path:
        return "synthetic_speech_corpus_builder"
    if "chinese_speech_frontend" in path:
        return "synthetic_speech_frontend_and_e2e"
    if "state_recovery" in path:
        return "main_model_recovery"
    if "wait_k_baselines" in path or path.startswith("run_waitk"):
        return "fixed_wait_k_baseline"
    if "semantic_causal_alignment" in path:
        return "semantic_alignment"
    if "causal_alignment_v1_corrected" in path:
        return "lexical_alignment_and_frozen_splits"
    if "fine_tuning" in path:
        return "training_and_shared_inference"
    if "locked500_bleu" in path:
        return "benchmark_metric"
    if "local_agreement" in path or "latency_audit" in path:
        return "control_and_latency_audit"
    return "dataset_generation_or_paper_analysis"

def withheld_remote_paths() -> set[str]:
    registry=PROVENANCE/"withheld_files.json"
    if not registry.is_file():
        raise RuntimeError("Missing public-source withholding registry")
    prefix="archive/original/SN_Research/"
    return {item["archive_path"][len(prefix):] for item in
        json.loads(registry.read_text("utf-8"))["files"]
        if item["archive_path"].startswith(prefix)}

def inventory() -> None:
    result=ssh_python(inventory_script())
    good=[]
    denied=[]
    withheld=withheld_remote_paths()
    for item in result["files"]:
        p=PurePosixPath(item["relative_path"])
        if p.as_posix() in withheld:
            continue
        if SENSITIVE_NAMES.search(p.name):
            denied.append({"relative_path":p.as_posix(),"reason":"sensitive_filename"})
            continue
        item["category"]=categorize(p.as_posix())
        good.append(item)
    result["files"]=good
    result["denied_filename_paths"]=denied
    result["source_root"]=REMOTE_ROOT
    result["max_file_bytes"]=MAX_FILE
    PROVENANCE.mkdir(parents=True,exist_ok=True)
    (PROVENANCE/"source_allowlist.review.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print(json.dumps({"candidate_files":len(good),
        "bytes":sum(x["size_bytes"] for x in good),
        "missing_roots":result["missing_roots"],
        "oversize_excluded":result["oversize_excluded"],
        "denied_filename_paths":denied},ensure_ascii=False,indent=2))

def retrieve() -> None:
    review=json.loads((PROVENANCE/"source_allowlist.review.json").read_text("utf-8"))
    withheld=withheld_remote_paths()
    items=[item for item in review["files"] if item["relative_path"] not in withheld]
    if not items:
        raise RuntimeError("Empty reviewed allowlist")
    accepted=[]
    rejected=[]
    for offset in range(0,len(items),50):
        batch=items[offset:offset+50]
        request=[i["relative_path"] for i in batch]
        script=remote_prelude()+"REQUEST=json.loads("+repr(json.dumps(request))+")\n"
        script+="""
items=[]
for relative in REQUEST:
    rel=pathlib.PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts or any(n in DENIED for n in rel.parts):
        raise RuntimeError("Rejected relative path")
    p=ROOT.joinpath(*rel.parts)
    if p.is_symlink() or not p.is_file() or p.stat().st_size>S["max_file"]:
        raise RuntimeError("Rejected file metadata")
    if ROOT not in p.resolve().parents:
        raise RuntimeError("Path escaped remote research root")
    data=p.read_bytes()
    decoded=data.decode("utf-8-sig",errors="replace")
    secret=re.compile(r"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|BEGIN [A-Z ]*PRIVATE KEY")
    lines=[number for number,line in enumerate(decoded.splitlines(),1) if secret.search(line)]
    row={"relative_path":relative,"size_bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),
         "secret_line_numbers":lines}
    if not lines:
        row["base64"]=base64.b64encode(data).decode("ascii")
    items.append(row)
print(json.dumps({"files":items},ensure_ascii=False))
"""
        response=ssh_python(script)
        allowed={x["relative_path"]:x for x in batch}
        if set(x["relative_path"] for x in response["files"])!=set(allowed):
            raise RuntimeError("Server response does not match exact allowlist")
        for record in response["files"]:
            rel=record["relative_path"]
            if record["secret_line_numbers"]:
                rejected.append({"relative_path":rel,
                    "line_numbers":record["secret_line_numbers"],
                    "reason":"credential_pattern_excluded_without_copying_content"})
                continue
            raw=base64.b64decode(record.pop("base64"),validate=True)
            if len(raw)!=record["size_bytes"] or hashlib.sha256(raw).hexdigest()!=record["sha256"]:
                raise RuntimeError("Transport integrity error for "+rel)
            if len(raw)!=allowed[rel]["size_bytes"]:
                raise RuntimeError("Remote file changed since review for "+rel)
            target=DEST.joinpath(*PurePosixPath(rel).parts)
            if not target.resolve().is_relative_to(DEST.resolve()):
                raise RuntimeError("Local target escaped archive root")
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and target.read_bytes()!=raw:
                raise RuntimeError("Refusing to overwrite different existing original: "+rel)
            target.write_bytes(raw)
            record.pop("secret_line_numbers")
            record.update({k:allowed[rel][k] for k in ("category","mtime_utc")})
            record["source_path"]=REMOTE_ROOT+"/"+rel
            record["archive_path"]=target.relative_to(RELEASE).as_posix()
            record["origin"]="authorized_remote_readonly_ssh"
            accepted.append(record)
        print(json.dumps({"retrieved":len(accepted),"reviewed":min(offset+50,len(items)),
                          "total":len(items),"rejected":len(rejected)}),flush=True)
    result={"schema_version":1,"created_at_utc":datetime.now(timezone.utc).isoformat(),
        "source_root":REMOTE_ROOT,"transport":"read_only_ssh_base64",
        "originals_modified":False,"file_count":len(accepted),
        "bytes":sum(i["size_bytes"] for i in accepted),"files":accepted,
        "security_excluded_locations":rejected,
        "scope_exclusions":["weights","raw data","JSONL datasets","audio/video",
            "credentials","G2P-DDM/gloss-pose","vendor/third-party source","environments/caches"],
        "review_file":"provenance/source_allowlist.review.json"}
    (PROVENANCE/"source_manifest.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"file_count":len(accepted),"bytes":result["bytes"],
                      "security_excluded_locations":rejected},ensure_ascii=False))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["inventory","retrieve"])
    args=parser.parse_args()
    {"inventory":inventory,"retrieve":retrieve}[args.action]()

if __name__=="__main__":
    main()

