#!/usr/bin/env python3
"""Compare a new full rollout with frozen paper outputs; never hide differences."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean


def read(path):
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate parents")
    return rows


def f1(row):
    ref = row["target_pseudo_asl_gloss_tokens"]
    hyp = row["predicted_pseudo_asl_gloss_tokens"]
    return 2 * sum((Counter(ref) & Counter(hyp)).values()) / max(1, len(ref) + len(hyp))


def laal(row):
    positions = row["write_source_positions"]
    source_n = len(row["source_tokens"])
    reference_n = len(row["target_pseudo_asl_gloss_tokens"])
    if not positions or not source_n or not reference_n:
        return None
    gamma = max(len(positions), reference_n) / source_n
    tau = next((i for i, v in enumerate(positions, 1) if v >= source_n), len(positions))
    return mean(positions[i] - i / gamma for i in range(tau))


def aggregate(rows):
    lags = [laal(row) for row in rows]
    return {"token_f1": mean(f1(row) for row in rows),
            "laal_source_tokens": mean(v for v in lags if v is not None),
            "finite_laal_count": sum(v is not None for v in lags),
            "completion": mean(bool(row["completed_after_full_source"]) for row in rows)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--expected-count", type=int, default=500)
    p.add_argument("--context", type=Path, help="Optional independently recorded execution metadata; not inferred by this comparison")
    args = p.parse_args()
    if args.report.exists():
        raise FileExistsError("Use a new comparison report path")
    old, fresh = read(args.reference), read(args.new)
    if len(old) != args.expected_count or len(fresh) != args.expected_count:
        raise ValueError("Unexpected number of parents")
    old_by = {r["id"]: r for r in old}
    fresh_by = {r["id"]: r for r in fresh}
    if old_by.keys() != fresh_by.keys():
        raise ValueError("Parent ID sets differ")
    counts = Counter()
    for key, reference in old_by.items():
        current = fresh_by[key]
        for field in ("source_tokens", "target_pseudo_asl_gloss_tokens"):
            if reference[field] != current[field]:
                raise ValueError("Source/reference differs: cannot compare same benchmark")
        gloss = reference["predicted_pseudo_asl_gloss_tokens"] == current["predicted_pseudo_asl_gloss_tokens"]
        timing = reference["write_source_positions"] == current["write_source_positions"]
        end = reference["termination_reason"] == current["termination_reason"]
        counts["gloss_exact"] += gloss
        counts["write_positions_exact"] += timing
        counts["termination_exact"] += end
        counts["gloss_timing_termination_exact"] += gloss and timing and end
    before, after = aggregate(old), aggregate(fresh)
    report = {"classification": "full_single_checkpoint_inference_rerun_not_training_reproduction",
              "count": args.expected_count, "same_source_and_reference": True,
              "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              "rerun_sha256": hashlib.sha256(args.new.read_bytes()).hexdigest(),
              "reference": before, "rerun": after,
              "delta_new_minus_reference": {k: after[k]-before[k] for k in ("token_f1", "laal_source_tokens", "completion")},
              "exact_counts": dict(counts),
              "bit_identical_trajectories": counts["gloss_timing_termination_exact"] == args.expected_count,
              "supplied_execution_context": json.loads(args.context.read_text(encoding="utf-8")) if args.context else None,
              "scope": "Archived unmodified evaluator and selected S1 adapter. No new model selection; historical outputs remain authoritative for paper reaggregation.",
              "causality_limit": "Hardware differs from original training/evaluation. Differences are recorded, not conclusively attributed to hardware alone."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
