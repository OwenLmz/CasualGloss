"""Acquire the official fixed Qwen backbone, not a floating adapter default.

Downloads only to an explicitly selected external Hugging Face cache. The
pretrained model is not part of the private author-experiment data archive.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models-root',type=Path,required=True,help='External HF_HOME, containing a hub/ cache')
    p.add_argument('--accept-upstream-terms',action='store_true')
    p.add_argument('--local-only',action='store_true',help='Do not contact the Hub; verify an existing snapshot')
    a=p.parse_args()
    dest=a.models_root.resolve()
    if dest==ROOT or ROOT in dest.parents:p.error('Keep model files outside the source checkout')
    if not a.accept_upstream_terms:p.error('Read the official pinned model conditions and supply --accept-upstream-terms')
    from huggingface_hub import snapshot_download
    path=Path(snapshot_download(repo_id=MODEL,revision=REVISION,cache_dir=str(dest/'hub'),
          allow_patterns=['*.json','*.safetensors','tokenizer*','vocab*','merges*','LICENSE*','README.md'],
          local_files_only=a.local_only))
    from experiment_matrix import check_snapshot
    errors=check_snapshot(path)
    print(json.dumps({'model':MODEL,'revision':REVISION,'snapshot':str(path),'errors':errors,'passed':not errors},indent=2))
    return int(bool(errors))


if __name__=='__main__':raise SystemExit(main())
