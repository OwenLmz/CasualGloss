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
from pathlib import Path
import shlex
import subprocess

REMOTE_ROOT = "/123090274/SN_Research/data/how2sign_english_asl_gloss_v1"
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
    paths += [EPEE + "outputs/full/lexical_v1/sentence_rollouts.jsonl",
              EPEE + "reports/full_analysis/per_source_scores.jsonl",
              EPEE + "reports/full_analysis/comparison_report.json",
              EPEE + "reports/posthoc_completed_only_20260921/completed_only_report.json",
              FULL + "clerc_epee_external_eval_v1/manifests/epee_sources_401.jsonl"]
    for k in (1, 3, 5, 7):
        for seed in ((20260821,) if k in (1, 3) else (20260821, 20260904, 20260905)):
            paths.append(EPEE + f"fixed_waitk_extension_20260920/outputs/full/waitk{k}_seed{seed}/sentence_rollouts.jsonl")
    paths.append(EPEE + "fixed_waitk_extension_20260920/reports/full_analysis/comparison_report.json")
    recovery = FULL + "state_recovery_v2_3_full/"
    paths += [recovery + "runs/qwen2_5_7b_lora_state_recovery_v2_3_full/rollout_confirmation_selected/sentence_rollouts.jsonl",
              recovery + "baselines/v2_2_native_confirmation/sentence_rollouts.jsonl",
              recovery + "baselines/v2_2_prompt_guard_confirmation/sentence_rollouts.jsonl"]
    for variant in ("no_semantic", "no_rescue", "no_negation", "no_occurrence", "no_burst"):
        paths.append(FULL + f"professor_feedback_experiments_20260912/ablations/{variant}/evaluation/confirmation/sentence_rollouts.jsonl")
    paths += [SPEECH + f"outputs/{name}.jsonl" for name in ("v1", "v2_3")]
    paths += [SPEECH + f"reports/{name}.json" for name in ("v1", "v2_3", "v2_3_vs_v1")]
    for seed in (20260904, 20260905):
        paths.append(SPEECH + f"multiseed/seed{seed}/outputs/v2_3_seed{seed}.jsonl")
    return paths

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--ssh-key", type=Path, required=True)
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--list-only", action="store_true")
    a = p.parse_args()
    targets=[(rel,rel) for rel in input_paths()]+[(rel,"locked500_from_local/"+name) for name,rel in locked_paths().items()]
    if a.list_only:
        print(json.dumps([{"remote_relative":r,"local_relative":d} for r,d in targets], indent=2)); return
    a.destination.mkdir(parents=True, exist_ok=True)
    def fetch(target):
        rel,local_rel=target
        dest = a.destination / local_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            cmd = ["scp", "-q", "-i", str(a.ssh_key), "-P", str(a.port),
                   "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
                   "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10",
                   f"{a.host}:{REMOTE_ROOT}/{rel}", str(dest)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                return {"relative_path": local_rel, "status": "fetch_failed", "error": r.stderr.strip()}
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        return {"relative_path": local_rel, "status": "present", "bytes": dest.stat().st_size,
                "sha256": digest, "remote_path": REMOTE_ROOT + "/" + rel}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(fetch, targets))
    command="sha256sum -- "+" ".join(shlex.quote(REMOTE_ROOT+"/"+rel) for rel,_ in targets)
    hash_result=subprocess.run(["ssh","-i",str(a.ssh_key),"-p",str(a.port),"-o","BatchMode=yes",
                                "-o","IdentitiesOnly=yes","-o","StrictHostKeyChecking=yes",
                                "-o","ConnectTimeout=10",a.host,command],capture_output=True,text=True)
    remote_hashes={}
    for line in hash_result.stdout.splitlines():
        parts=line.split(None,1)
        if len(parts)==2:remote_hashes[parts[1].lstrip("*")]=parts[0]
    for row in rows:
        if row["status"]=="present":
            row["remote_sha256"]=remote_hashes.get(row["remote_path"])
            row["remote_hash_matches"]=row["sha256"]==row["remote_sha256"]
            if not row["remote_hash_matches"]:row["status"]="remote_hash_unverified_or_mismatch"
    (a.destination / "verification_input_manifest.json").write_text(
        json.dumps({"contains_restricted_text": True, "files": rows}, indent=2) + "\n", encoding="utf-8")
    failed = [r for r in rows if r["status"] != "present"]
    print(json.dumps({"present": len(rows)-len(failed), "failed": failed}, indent=2))
    raise SystemExit(bool(failed))

if __name__ == "__main__":
    main()
