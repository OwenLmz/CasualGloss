#!/usr/bin/env python3
"""Fetch an explicit, bounded list of historical outputs into a PRIVATE folder.

This never executes remote code, changes remote files, or fetches model weights.
The destination contains restricted text and must not be committed to Git.
Authentication is delegated to the caller's SSH client; no credentials are read.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess

REMOTE_ROOT = "/123090274/SN_Research/data/how2sign_english_asl_gloss_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (20260829, 20260904, 20260905)
FULL = "full_v4_train/"
EPEE = FULL + "clerc_epee_adaptive_comparison_20260920/"
SPEECH = "chinese_speech_frontend_v0/paper_e2e_locked500_20260904/"

def locked_paths():
    paths={}
    for i,seed in enumerate(SEEDS,1):
        folder="state_recovery_v2_3_full" if i==1 else f"state_recovery_v2_3_multiseed_{seed}"
        evaluation="final_evaluation_v2_3" if i==1 else f"final_evaluation_seed{seed}"
        paths[f"cg_s{i}.jsonl"]=FULL+f"{folder}/{evaluation}/locked500/v2_3/sentence_rollouts.jsonl"
    for k in (1,3,5,7):
        run=f"qwen2_5_7b_lora_wait_k{k}_full_v1"+("_recovery_a6000_2gpu" if k==3 else "")
        paths[f"waitk{k}.jsonl"]=FULL+f"wait_k_baselines_v1/runs/{run}/rollout_locked500/sentence_rollouts.jsonl"
    for k in (5,7):
        for seed in (20260904,20260905):
            paths[f"waitk{k}_seed{seed}.jsonl"]=FULL+f"wait_k_baselines_multiseed_20260914/runs/qwen2_5_7b_lora_wait_k{k}_full_seed{seed}/rollout_locked500/sentence_rollouts.jsonl"
    return paths

def input_paths():
    paths = []
    for seed in SEEDS:
        for k in range(1, 9):
            paths.append(FULL + f"transllama_learned_adaptive_20260918/runs/final_test/final_seed_{seed}/k{k}/sentence_rollouts.jsonl")
        paths.append(EPEE + f"outputs/full/causalgloss_{seed}/sentence_rollouts.jsonl")
        for k in range(1, 9):
            paths.append(EPEE + f"outputs/full/transllama_{seed}/k{k}/sentence_rollouts.jsonl")
    # Only final-paper prediction records and references. Do not fetch exploratory
    # CLERC lexical-control records or mixed historical summary reports.
    paths.append(FULL + "clerc_epee_external_eval_v1/manifests/epee_sources_401.jsonl")
    for k in (1, 3, 5, 7):
        for seed in ((20260821,) if k in (1, 3) else (20260821, 20260904, 20260905)):
            paths.append(EPEE + f"fixed_waitk_extension_20260920/outputs/full/waitk{k}_seed{seed}/sentence_rollouts.jsonl")
    recovery = FULL + "state_recovery_v2_3_full/"
    paths += [recovery + "runs/qwen2_5_7b_lora_state_recovery_v2_3_full/rollout_confirmation_selected/sentence_rollouts.jsonl",
              recovery + "baselines/v2_2_native_confirmation/sentence_rollouts.jsonl",
              recovery + "baselines/v2_2_prompt_guard_confirmation/sentence_rollouts.jsonl"]
    for variant in ("no_semantic", "no_rescue", "no_negation", "no_occurrence", "no_burst"):
        paths.append(FULL + f"professor_feedback_experiments_20260912/ablations/{variant}/evaluation/confirmation/sentence_rollouts.jsonl")
    paths += [SPEECH + f"outputs/{name}.jsonl" for name in ("v1", "v2_3")]
    for seed in (20260904, 20260905):
        paths.append(SPEECH + f"multiseed/seed{seed}/outputs/v2_3_seed{seed}.jsonl")
    return paths

def target_paths():
    """Exactly 82 final-paper rollouts plus one 401-prompt reference manifest."""
    targets=[(rel,rel) for rel in input_paths()]+[(rel,"locked500_from_local/"+name) for name,rel in locked_paths().items()]
    for remote,local in targets:
        validate_relative_path(remote)
        validate_relative_path(local)
    if len(targets)!=83 or len({r for r,_ in targets})!=83 or len({d for _,d in targets})!=83:
        raise ValueError("Final-paper allowlist must contain exactly 83 unique remote/local paths")
    return targets

def validate_transport(host,port,workers):
    # Pass arguments directly to SSH/SCP, never through a local shell. Also
    # reject option-shaped hosts and remote-shell punctuation explicitly.
    if (not isinstance(host,str) or not re.fullmatch(r"[A-Za-z0-9_.@:-]+",host)
            or host.startswith("-") or host.count("@")>1
            or any(not part or part.startswith("-") for part in host.split("@"))):
        raise ValueError("A valid authorized host USER@HOST is required; no options, whitespace or shell characters")
    if type(port) is not int or not 1<=port<=65535:
        raise ValueError("Port must be an integer between 1 and 65535")
    if type(workers) is not int or not 1<=workers<=16:
        raise ValueError("Workers must be an integer between 1 and 16")

def validate_relative_path(value):
    if not isinstance(value,str) or not value or not re.fullmatch(r"[A-Za-z0-9_./-]+",value):
        raise ValueError("Unsafe path in verification allowlist")
    rel=PurePosixPath(value)
    if rel.is_absolute() or ".." in rel.parts or rel.as_posix()!=value or value==".":
        raise ValueError("Verification targets must be normalized relative paths without traversal")
    return rel

def validate_private_destination(destination):
    destination = destination.resolve()
    if destination == REPOSITORY_ROOT or REPOSITORY_ROOT in destination.parents:
        raise ValueError("Choose a PRIVATE destination outside this repository; raw inputs must not enter Git.")
    return destination

def safe_destination_file(destination,relative):
    destination=validate_private_destination(destination)
    rel=validate_relative_path(relative)
    local=destination.joinpath(*rel.parts)
    resolved=local.resolve()
    if destination not in resolved.parents or resolved==REPOSITORY_ROOT or REPOSITORY_ROOT in resolved.parents:
        raise ValueError("Verification target escapes the private root or enters the source repository through a link")
    return local

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def read_remote_hashes(targets,ssh_key,host,port):
    command="sha256sum -- "+" ".join(shlex.quote(REMOTE_ROOT+"/"+rel) for rel,_ in targets)
    response=subprocess.run(["ssh","-i",str(ssh_key),"-p",str(port),"-o","BatchMode=yes",
                             "-o","IdentitiesOnly=yes","-o","StrictHostKeyChecking=yes",
                             "-o","ConnectTimeout=10",host,command],capture_output=True,text=True,check=True)
    expected_paths={REMOTE_ROOT+"/"+rel for rel,_ in targets}
    hashes={}
    for line in response.stdout.splitlines():
        parts=line.split(None,1)
        if len(parts)!=2 or not re.fullmatch(r"[a-f0-9]{64}",parts[0]):
            raise ValueError("Invalid remote SHA-256 response; no inputs are verified")
        remote=parts[1].lstrip("*")
        if remote not in expected_paths or remote in hashes:
            raise ValueError("Unexpected or duplicate path in remote SHA-256 response")
        hashes[remote]=parts[0]
    if set(hashes)!=expected_paths:
        raise ValueError("Incomplete remote SHA-256 response; no inputs are verified")
    return hashes

def fetch_one(target,destination,ssh_key,host,port,expected_hash):
    rel,local_rel=target
    validate_relative_path(rel)
    if not isinstance(expected_hash,str) or not re.fullmatch(r"[a-f0-9]{64}",expected_hash):
        raise ValueError("A valid remote SHA-256 is required before materializing an input")
    dest=safe_destination_file(destination,local_rel)
    existed=dest.exists()
    if existed:
        if not dest.is_file() or digest(dest)!=expected_hash:
            raise FileExistsError(f"Refusing to overwrite or verify nonmatching existing input: {dest}")
    else:
        dest.parent.mkdir(parents=True,exist_ok=True)
        # Recheck after directory creation and before letting SCP write.
        dest=safe_destination_file(destination,local_rel)
        cmd=["scp","-q","-i",str(ssh_key),"-P",str(port),"-o","BatchMode=yes",
             "-o","IdentitiesOnly=yes","-o","StrictHostKeyChecking=yes","-o","ConnectTimeout=10",
             f"{host}:{REMOTE_ROOT}/{rel}",str(dest)]
        response=subprocess.run(cmd,capture_output=True,text=True)
        if response.returncode:
            return {"relative_path":local_rel,"status":"fetch_failed","error":response.stderr.strip()}
    dest=safe_destination_file(destination,local_rel)
    if not dest.is_file() or digest(dest)!=expected_hash:
        raise ValueError(f"Input verification failed; preserve for inspection, do not use: {dest}")
    return {"relative_path":local_rel,"status":"present","bytes":dest.stat().st_size,
            "sha256":expected_hash,"remote_path":REMOTE_ROOT+"/"+rel,
            "remote_sha256":expected_hash,"remote_hash_matches":True,
            "transfer_status":"already_verified" if existed else "downloaded_verified"}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--ssh-key", type=Path, required=True)
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--list-only", action="store_true")
    a = p.parse_args()
    try:
        validate_transport(a.host,a.port,a.workers)
    except ValueError as error:
        p.error(str(error))
    targets=target_paths()
    if a.list_only:
        print(json.dumps([{"remote_relative":r,"local_relative":d} for r,d in targets], indent=2)); return
    a.destination=validate_private_destination(a.destination)
    a.destination.mkdir(parents=True, exist_ok=True)
    # Acquire complete, validated hashes before accepting an existing local file.
    remote_hashes=read_remote_hashes(targets,a.ssh_key,a.host,a.port)
    def fetch(target):
        return fetch_one(target,a.destination,a.ssh_key,a.host,a.port,remote_hashes[REMOTE_ROOT+"/"+target[0]])
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(fetch, targets))
    safe_destination_file(a.destination,"verification_input_manifest.json").write_text(
        json.dumps({"contains_restricted_text": True, "files": rows}, indent=2) + "\n", encoding="utf-8")
    failed = [r for r in rows if r["status"] != "present"]
    print(json.dumps({"present": len(rows)-len(failed), "failed": failed}, indent=2))
    raise SystemExit(bool(failed))

if __name__ == "__main__":
    main()
