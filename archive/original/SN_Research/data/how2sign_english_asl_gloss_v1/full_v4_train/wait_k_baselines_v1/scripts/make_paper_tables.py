#!/usr/bin/env python3
"""Convert the locked-test comparison report into paper-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = [
    ("sequence_similarity", "Seq. Sim.", "higher"),
    ("token_f1", "Token F1", "higher"),
    ("exact_match", "Exact", "higher"),
    ("reference_length_al", "AL", "lower"),
    ("laal", "LAAL", "lower"),
    ("completed_after_full_source", "Complete", "higher"),
    ("invalid_ratio", "Invalid", "lower"),
]

DISPLAY_NAMES = {
    "offline_finetune1": "Offline FT1",
    "wait_k3": "Wait-3",
    "v1": "v1",
    "v2_1": "v2.1",
    "v2_2": "v2.2",
}


def format_value(metric: str, value: float) -> str:
    if metric in {"exact_match", "completed_after_full_source", "invalid_ratio"}:
        return f"{100.0 * value:.2f}"
    if metric in {"reference_length_al", "laal"}:
        return f"{value:.3f}"
    return f"{value:.4f}"


def format_delta(metric: str, value: float) -> str:
    if metric in {"exact_match", "completed_after_full_source", "invalid_ratio"}:
        return f"{100.0 * value:+.2f}"
    if metric in {"reference_length_al", "laal"}:
        return f"{value:+.3f}"
    return f"{value:+.4f}"


def escape_latex(value: str) -> str:
    return value.replace("_", "\\_").replace("%", "\\%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--system-order", nargs="*")
    args = parser.parse_args()

    report: dict[str, Any] = json.loads(args.comparison.read_text(encoding="utf-8"))
    systems = args.system_order or list(report["systems"])
    missing = [name for name in systems if name not in report["systems"]]
    if missing:
        raise ValueError(f"Unknown systems: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in systems:
        means = report["systems"][name]["means"]
        rows.append(
            {
                "system": DISPLAY_NAMES.get(name, name),
                **{metric: format_value(metric, means[metric]) for metric, _, _ in METRICS},
            }
        )

    tsv_path = args.output_dir / "table_main_locked500.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    headers = ["System", *(label for _, label, _ in METRICS)]
    markdown = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", *(["---:"] * len(METRICS))]) + " |",
    ]
    for row in rows:
        markdown.append(
            "| " + " | ".join([row["system"], *(row[metric] for metric, _, _ in METRICS)]) + " |"
        )
    markdown.extend(
        [
            "",
            "Exact, Complete, and Invalid are percentages; all other values are proportions or source-token lag units.",
            "References are deterministic pseudo glosses, not expert gold ASL.",
        ]
    )
    (args.output_dir / "table_main_locked500.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    latex = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Locked-500 pseudo-gloss quality and latency. Higher is better for quality and completion; lower is better for AL, LAAL, and invalid output.}",
        "\\label{tab:main_locked500}",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "System & Seq. Sim. & Token F1 & Exact (\\%) & AL & LAAL & Complete (\\%) & Invalid (\\%) \\\\",
        "\\midrule",
    ]
    for row in rows:
        values = " & ".join(row[metric] for metric, _, _ in METRICS)
        latex.append(f"{escape_latex(row['system'])} & {values} \\\\")
    latex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    (args.output_dir / "table_main_locked500.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")

    baseline = report["baseline"]
    paired_rows = []
    for system, metric_results in report["paired_vs_baseline"].items():
        for metric, label, direction in METRICS:
            result = metric_results[metric]
            paired_rows.append(
                {
                    "comparison": f"{DISPLAY_NAMES.get(system, system)} - {DISPLAY_NAMES.get(baseline, baseline)}",
                    "metric": label,
                    "direction": direction,
                    "delta": format_delta(metric, result["mean_delta"]),
                    "ci95": (
                        f"[{format_delta(metric, result['ci95_lower'])}, "
                        f"{format_delta(metric, result['ci95_upper'])}]"
                    ),
                    "conclusion": result["conclusion"],
                    "paired_count": result["paired_count"],
                }
            )
    with (args.output_dir / "table_paired_bootstrap.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(paired_rows)

    claim_summary = {
        "scientific_status": report["scientific_status"],
        "baseline": baseline,
        "common_parent_count": report["common_parent_count"],
        "bootstrap": report["bootstrap"],
        "significant_results": [row for row in paired_rows if row["conclusion"] != "no_significant_difference"],
        "limitations": report["limitations"],
    }
    (args.output_dir / "claim_summary.json").write_text(
        json.dumps(claim_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "paired_rows": len(paired_rows), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
