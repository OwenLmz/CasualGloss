#!/usr/bin/env python3
"""Gate a portability run against frozen original BF16 generator outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def token_f1(left: list[str], right: list[str]) -> float:
    overlap = sum((Counter(left) & Counter(right)).values())
    precision = overlap / max(1, len(right))
    recall = overlap / max(1, len(left))
    return 2 * precision * recall / max(1e-12, precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--replication", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--minimum-exact-rate", type=float, default=0.95)
    parser.add_argument("--minimum-mean-token-f1", type=float, default=0.995)
    parser.add_argument("--reference-execution", default="original RTX5090 BF16 batch4")
    parser.add_argument("--replication-execution", default="replication execution not specified")
    args = parser.parse_args()

    reference = {str(row["request_id"]): row for row in read_jsonl(args.reference)}
    replication = {str(row["request_id"]): row for row in read_jsonl(args.replication)}
    if set(reference) != set(replication):
        raise SystemExit("Smoke reference and replication IDs do not agree")

    comparisons = []
    for request_id in reference:
        expected = str(reference[request_id]["target_pseudo_asl_gloss"]).split()
        observed = str(replication[request_id]["target_pseudo_asl_gloss"]).split()
        comparisons.append(
            {
                "request_id": request_id,
                "expected": " ".join(expected),
                "observed": " ".join(observed),
                "exact": expected == observed,
                "token_f1": token_f1(expected, observed),
            }
        )

    exact_rate = mean(float(row["exact"]) for row in comparisons)
    mean_f1 = mean(float(row["token_f1"]) for row in comparisons)
    passed = exact_rate >= args.minimum_exact_rate and mean_f1 >= args.minimum_mean_token_f1
    report = {
        "scientific_status": "cross_hardware_dtype_portability_gate",
        "reference_execution": args.reference_execution,
        "replication_execution": args.replication_execution,
        "sample_count": len(comparisons),
        "exact_match_rate": exact_rate,
        "mean_token_f1": mean_f1,
        "minimum_exact_rate": args.minimum_exact_rate,
        "minimum_mean_token_f1": args.minimum_mean_token_f1,
        "passed": passed,
        "mismatches": [row for row in comparisons if not row["exact"]],
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit("Portability gate failed; full Epee generation is blocked")


if __name__ == "__main__":
    main()
