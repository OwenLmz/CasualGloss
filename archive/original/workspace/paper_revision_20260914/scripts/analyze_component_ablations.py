#!/usr/bin/env python3
"""Audit confirmation-250 component ablations and compute paired intervals."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import fmean


VARIANTS = [
    "no_semantic",
    "no_rescue",
    "no_negation",
    "no_occurrence",
    "no_burst",
]

F1_KEY = "pseudo_gloss_token_f1"
LAAL_KEY = "length_adaptive_average_lagging_source_tokens"


def load_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            row_id = row["id"]
            if row_id in rows:
                raise ValueError(f"Duplicate id in {path}: {row_id}")
            rows[row_id] = row
    return rows


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    boot_means = [
        fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(samples)
    ]
    return percentile(boot_means, 0.025), percentile(boot_means, 0.975)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()

    systems = ["full", *VARIANTS]
    rows_by_system = {
        name: load_rows(args.data_dir / name / "sentence_rollouts.jsonl")
        for name in systems
    }
    full = rows_by_system["full"]
    if len(full) != 250:
        raise ValueError(f"Expected 250 full-model rows, found {len(full)}")

    full_ids = set(full)
    for name in VARIANTS:
        rows = rows_by_system[name]
        if set(rows) != full_ids:
            raise ValueError(f"Parent-id mismatch for {name}")
        for row_id in full_ids:
            for key in ("source_text_en", "target_pseudo_asl_gloss"):
                if rows[row_id][key] != full[row_id][key]:
                    raise ValueError(f"{key} mismatch for {name}: {row_id}")

    ids = sorted(full_ids)
    output = {
        "audit": {
            "parent_count": len(ids),
            "identical_parent_ids_sources_and_references": True,
            "bootstrap_samples": args.bootstrap_samples,
            "delta_direction": "full_minus_ablated",
        },
        "systems": {},
        "paired_differences": {},
    }

    for name in systems:
        rows = rows_by_system[name]
        valid_laal = [
            rows[row_id][LAAL_KEY]
            for row_id in ids
            if rows[row_id][LAAL_KEY] is not None
        ]
        output["systems"][name] = {
            "token_f1": fmean(rows[row_id][F1_KEY] for row_id in ids),
            "laal": fmean(valid_laal),
            "laal_valid_count": len(valid_laal),
            "completion": fmean(float(rows[row_id]["completed_after_full_source"]) for row_id in ids),
        }

    for index, name in enumerate(VARIANTS):
        rows = rows_by_system[name]
        f1_diffs = [full[row_id][F1_KEY] - rows[row_id][F1_KEY] for row_id in ids]
        laal_ids = [
            row_id
            for row_id in ids
            if full[row_id][LAAL_KEY] is not None and rows[row_id][LAAL_KEY] is not None
        ]
        laal_diffs = [full[row_id][LAAL_KEY] - rows[row_id][LAAL_KEY] for row_id in laal_ids]
        f1_ci = bootstrap_ci(f1_diffs, args.bootstrap_samples, args.seed + index * 2)
        laal_ci = bootstrap_ci(laal_diffs, args.bootstrap_samples, args.seed + index * 2 + 1)
        output["paired_differences"][name] = {
            "delta_token_f1": fmean(f1_diffs),
            "delta_token_f1_ci95": list(f1_ci),
            "delta_laal": fmean(laal_diffs),
            "delta_laal_ci95": list(laal_ci),
            "laal_paired_count": len(laal_diffs),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "component_ablation_analysis.json"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    tsv_path = args.output_dir / "component_ablation_analysis.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "system", "token_f1", "laal", "completion",
            "full_minus_ablation_f1", "f1_ci_low", "f1_ci_high",
            "full_minus_ablation_laal", "laal_ci_low", "laal_ci_high",
        ])
        for name in systems:
            metrics = output["systems"][name]
            diff = output["paired_differences"].get(name, {})
            writer.writerow([
                name,
                f"{metrics['token_f1']:.6f}",
                f"{metrics['laal']:.6f}",
                f"{metrics['completion']:.4f}",
                f"{diff.get('delta_token_f1', 0.0):.6f}" if diff else "",
                f"{diff.get('delta_token_f1_ci95', ['', ''])[0]:.6f}" if diff else "",
                f"{diff.get('delta_token_f1_ci95', ['', ''])[1]:.6f}" if diff else "",
                f"{diff.get('delta_laal', 0.0):.6f}" if diff else "",
                f"{diff.get('delta_laal_ci95', ['', ''])[0]:.6f}" if diff else "",
                f"{diff.get('delta_laal_ci95', ['', ''])[1]:.6f}" if diff else "",
            ])

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
