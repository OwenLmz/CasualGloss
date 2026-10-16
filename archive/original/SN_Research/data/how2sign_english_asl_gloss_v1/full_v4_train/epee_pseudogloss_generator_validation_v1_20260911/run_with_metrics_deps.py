#!/usr/bin/env python3
"""Run a metrics script with experiment-local pure-Python dependencies."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_with_metrics_deps.py SCRIPT [ARGS ...]")

    deps = Path(__file__).resolve().parent / "_deps"
    if deps.is_dir():
        sys.path.append(str(deps))

    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
