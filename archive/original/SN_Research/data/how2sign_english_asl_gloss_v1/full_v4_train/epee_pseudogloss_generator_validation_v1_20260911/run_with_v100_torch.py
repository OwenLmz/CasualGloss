#!/usr/bin/env python3
"""Run a script with system V100 PyTorch and shared non-Torch dependencies."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_with_v100_torch.py SCRIPT [ARGS ...]")

    # Import system PyTorch before exposing the newer environment's site-packages.
    import torch  # noqa: F401

    shared_site = os.environ.get(
        "SHARED_NON_TORCH_SITE",
        "/123090274/SN_Research/venvs/wlasl-cu128-env/lib/python3.10/site-packages",
    )
    private_deps = os.environ.get("EXPERIMENT_PRIVATE_DEPS")
    for path in (private_deps, shared_site):
        if path and Path(path).is_dir():
            sys.path.append(path)

    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
