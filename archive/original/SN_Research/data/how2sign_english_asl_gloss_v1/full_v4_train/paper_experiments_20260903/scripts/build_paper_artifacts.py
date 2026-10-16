#!/usr/bin/env python3
"""Build publication tables and figures from frozen aggregate reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "v1": "#4C78A8",
    "v2_1": "#9C9C9C",
    "v2_2": "#E6862A",
    "v2_3": "#138A72",
    "zero": "#6B7280",
    "grid": "#D9DEE7",
    "ink": "#172033",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def write_latex_table(
    path: Path, rows: list[dict[str, Any]], fields: list[str], headers: list[str]
) -> None:
    align = "l" + "r" * (len(fields) - 1)
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule"]
    lines.append(" & ".join(headers) + r" \\")
    lines.append("\\midrule")
    for row in rows:
        cells = [latex_escape(str(row[field])) for field in fields]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot_frontier(waitk_rows: list[dict[str, str]], locked: dict[str, Any], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.7), constrained_layout=True)
    for model, label, color in (
        ("v1", "v1 + modified Wait-k", COLORS["v1"]),
        ("v2.2", "v2.2 + modified Wait-k", COLORS["v2_2"]),
    ):
        rows = [
            row
            for row in waitk_rows
            if row["model"] == model and row["policy"] == "transllama_wait_k"
        ]
        rows.sort(key=lambda row: int(row["k"]))
        x = [float(row["laal"]) for row in rows]
        y = [float(row["token_f1"]) for row in rows]
        ax.plot(x, y, marker="o", markersize=5, linewidth=1.5, color=color, label=label)
        for row, x_value, y_value in zip(rows, x, y):
            ax.annotate(
                f"k={row['k']}",
                (x_value, y_value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6.5,
                color=color,
            )

    v23 = locked["versions"]["v2_3"]["means"]
    ax.scatter(
        [v23["laal"]],
        [v23["token_f1"]],
        s=120,
        marker="*",
        color=COLORS["v2_3"],
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
        label="v2.3 learned causal policy",
    )
    ax.annotate(
        "v2.3",
        (v23["laal"], v23["token_f1"]),
        xytext=(5, 8),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=COLORS["v2_3"],
        arrowprops={"arrowstyle": "-", "color": COLORS["v2_3"], "lw": 0.8},
    )
    ax.set_xlabel("Length-adaptive average lagging (source words, lower is better)")
    ax.set_ylabel("Pseudo-gloss token F1 (higher is better)")
    ax.set_title("v2.3 reaches the high-quality, low-latency corner")
    ax.set_ylim(0.36, 0.74)
    ax.grid(axis="both", color=COLORS["grid"], linewidth=0.6, alpha=0.75)
    ax.legend(loc="upper right")
    ax.text(
        0.01,
        0.02,
        "Locked How2Sign-derived test, n=500; reference is pseudo gloss, not expert gold ASL.",
        transform=ax.transAxes,
        fontsize=6.5,
        color="#5B6472",
    )
    save_figure(fig, output / "fig_quality_latency_frontier")


def forest_panel(
    ax: plt.Axes,
    entries: list[dict[str, Any]],
    title: str,
    xlabel: str,
) -> None:
    y = np.arange(len(entries))[::-1]
    means = np.array([entry["mean"] for entry in entries])
    lower = np.array([entry["lower"] for entry in entries])
    upper = np.array([entry["upper"] for entry in entries])
    colors = [entry.get("color", COLORS["v2_3"]) for entry in entries]
    ax.axvline(0, color=COLORS["zero"], lw=0.9, linestyle="--")
    for index, (mean_value, lo, hi, color) in enumerate(zip(means, lower, upper, colors)):
        ax.errorbar(
            mean_value,
            y[index],
            xerr=[[mean_value - lo], [hi - mean_value]],
            fmt="o",
            markersize=5,
            color=color,
            ecolor=color,
            elinewidth=1.4,
            capsize=2.5,
        )
    ax.set_yticks(y, [entry["label"] for entry in entries])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6, alpha=0.75)


def plot_paired_effects(
    locked: dict[str, Any],
    waitk_pair: dict[str, Any],
    e2e_pair: dict[str, Any],
    output: Path,
) -> None:
    comparisons = locked["paired_comparisons"]
    quality_entries = []
    latency_entries = []
    for key, label, color in (
        ("v2_3_vs_v1", "vs v1 native", COLORS["v1"]),
        ("v2_3_vs_v2_2", "vs v2.2 native", COLORS["v2_2"]),
    ):
        q = comparisons[key]["token_f1"]
        l = comparisons[key]["laal"]
        quality_entries.append(
            {"label": label, "mean": q["mean_delta"], "lower": q["ci95_lower"], "upper": q["ci95_upper"], "color": color}
        )
        latency_entries.append(
            {"label": label, "mean": l["mean_delta"], "lower": l["ci95_lower"], "upper": l["ci95_upper"], "color": color}
        )
    q = waitk_pair["paired_metrics"]["token_f1"]
    l = waitk_pair["paired_metrics"]["laal"]
    quality_entries.append(
        {"label": "vs modified Wait-k k=1", "mean": q["mean_delta"], "lower": q["ci95_lower"], "upper": q["ci95_upper"], "color": "#7655A6"}
    )
    latency_entries.append(
        {"label": "vs modified Wait-k k=1", "mean": l["mean_delta"], "lower": l["ci95_lower"], "upper": l["ci95_upper"], "color": "#7655A6"}
    )

    speech_quality = e2e_pair["paired_deltas"]["pseudo_gloss_token_f1"]
    speech_laal = e2e_pair["paired_deltas"]["speech_to_gloss_laal_seconds"]
    speech_compute = e2e_pair["paired_deltas"]["speech_to_gloss_compute_adjusted_laal_seconds"]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.4), constrained_layout=True)
    forest_panel(
        axes[0, 0],
        quality_entries,
        "A  Locked-500 quality effect",
        "Delta token F1 (v2.3 minus baseline)",
    )
    forest_panel(
        axes[0, 1],
        latency_entries,
        "B  Locked-500 latency effect",
        "Delta LAAL in source words",
    )
    forest_panel(
        axes[1, 0],
        [
            {
                "label": "v2.3 vs v1",
                "mean": speech_quality["mean_candidate_minus_baseline"],
                "lower": speech_quality["ci95_lower"],
                "upper": speech_quality["ci95_upper"],
                "color": COLORS["v2_3"],
            }
        ],
        "C  Chinese-speech quality effect",
        "Delta token F1 (v2.3 minus v1)",
    )
    forest_panel(
        axes[1, 1],
        [
            {
                "label": "Algorithmic speech-time LAAL",
                "mean": speech_laal["mean_candidate_minus_baseline"],
                "lower": speech_laal["ci95_lower"],
                "upper": speech_laal["ci95_upper"],
                "color": COLORS["v2_3"],
            },
            {
                "label": "Compute-adjusted LAAL",
                "mean": speech_compute["mean_candidate_minus_baseline"],
                "lower": speech_compute["ci95_lower"],
                "upper": speech_compute["ci95_upper"],
                "color": "#7655A6",
            },
        ],
        "D  Chinese-speech latency effect",
        "Delta LAAL in seconds",
    )
    fig.suptitle("Paired bootstrap effects with 95% confidence intervals", fontsize=11, fontweight="bold")
    fig.text(
        0.5,
        -0.015,
        "Locked test: n=500. Synthetic Mandarin speech pilot: n=100. 10,000 paired bootstrap samples.",
        ha="center",
        fontsize=7,
        color="#5B6472",
    )
    save_figure(fig, output / "fig_paired_effects")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--waitk-summary", type=Path, required=True)
    parser.add_argument("--locked-comparison", type=Path, required=True)
    parser.add_argument("--waitk-paired", type=Path, required=True)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--epee", type=Path, required=True)
    parser.add_argument("--e2e-v1", type=Path, required=True)
    parser.add_argument("--e2e-v2-3", type=Path, required=True)
    parser.add_argument("--e2e-paired", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    configure_plotting()
    waitk_rows = read_tsv(args.waitk_summary)
    locked = read_json(args.locked_comparison)
    waitk_pair = read_json(args.waitk_paired)
    ablation = read_json(args.ablation)
    epee = read_json(args.epee)
    e2e_v1 = read_json(args.e2e_v1)
    e2e_v23 = read_json(args.e2e_v2_3)
    e2e_pair = read_json(args.e2e_paired)
    output = args.output_dir
    tables = output / "tables"
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)

    main_rows = []
    for name, display in (
        ("v1", "v1 native"),
        ("v2_1", "v2.1 native"),
        ("v2_2", "v2.2 native"),
        ("v2_3", "v2.3 learned causal policy"),
    ):
        means = locked["versions"][name]["means"]
        main_rows.append(
            {
                "system": display,
                "token_f1": f"{means['token_f1']:.4f}",
                "sequence_similarity": f"{means['sequence_similarity']:.4f}",
                "laal": f"{means['laal']:.4f}",
                "completion_percent": f"{100 * means['completed_after_full_source']:.1f}",
                "invalid_percent": f"{100 * means['invalid_ratio']:.3f}",
            }
        )
    waitk_k1 = next(row for row in waitk_rows if row["label"] == "v1_transllama_k1")
    main_rows.append(
        {
            "system": "v1 + modified Wait-k k=1",
            "token_f1": f"{float(waitk_k1['token_f1']):.4f}",
            "sequence_similarity": f"{float(waitk_k1['sequence_similarity']):.4f}",
            "laal": f"{float(waitk_k1['laal']):.4f}",
            "completion_percent": f"{100 * float(waitk_k1['valid_completion']):.1f}",
            "invalid_percent": f"{100 * float(waitk_k1['invalid_ratio']):.3f}",
        }
    )
    main_fields = ["system", "token_f1", "sequence_similarity", "laal", "completion_percent", "invalid_percent"]
    write_tsv(tables / "table_main_locked500.tsv", main_rows, main_fields)
    write_latex_table(
        tables / "table_main_locked500.tex",
        main_rows,
        main_fields,
        ["System", "Token F1", "Seq. sim.", "LAAL", "Complete (\\%)", "Invalid (\\%)"],
    )

    frontier_rows = [row for row in waitk_rows if row["policy"] == "transllama_wait_k"]
    write_tsv(tables / "table_modified_waitk_sweep.tsv", frontier_rows, list(frontier_rows[0]))

    v23_native = ablation["v2_3_vs_v2_2_native"]["metrics"]
    v23_guard = ablation["v2_3_vs_v2_2_prompt_guard"]["metrics"]
    ablation_rows = [
        {
            "comparison": "v2.3 - v2.2 native",
            "token_f1_delta": f"{v23_native['token_f1']['mean_difference']:.4f}",
            "token_f1_ci95": f"[{v23_native['token_f1']['paired_bootstrap_95_ci'][0]:.4f}, {v23_native['token_f1']['paired_bootstrap_95_ci'][1]:.4f}]",
            "laal_delta": f"{v23_native['laal_source_tokens']['mean_difference']:.4f}",
            "laal_ci95": f"[{v23_native['laal_source_tokens']['paired_bootstrap_95_ci'][0]:.4f}, {v23_native['laal_source_tokens']['paired_bootstrap_95_ci'][1]:.4f}]",
        },
        {
            "comparison": "v2.3 - inference guard only",
            "token_f1_delta": f"{v23_guard['token_f1']['mean_difference']:.4f}",
            "token_f1_ci95": f"[{v23_guard['token_f1']['paired_bootstrap_95_ci'][0]:.4f}, {v23_guard['token_f1']['paired_bootstrap_95_ci'][1]:.4f}]",
            "laal_delta": f"{v23_guard['laal_source_tokens']['mean_difference']:.4f}",
            "laal_ci95": f"[{v23_guard['laal_source_tokens']['paired_bootstrap_95_ci'][0]:.4f}, {v23_guard['laal_source_tokens']['paired_bootstrap_95_ci'][1]:.4f}]",
        },
    ]
    write_tsv(tables / "table_ablation_confirmation250.tsv", ablation_rows, list(ablation_rows[0]))

    external = epee["subsets"]["all_401"]
    external_f1 = external["tracks"]["content"]["best_token_f1"]["means"]
    external_rows = []
    for name, display in (("v1", "v1"), ("v2_2", "v2.2"), ("v2_3", "v2.3")):
        external_rows.append(
            {
                "system": display,
                "best_content_token_f1": f"{external_f1[name]:.4f}",
                "laal": f"{external['laal']['means'][name]:.4f}",
                "completion_percent": f"{100 * external['completed']['means'][name]:.2f}",
            }
        )
    write_tsv(tables / "table_epee_external401.tsv", external_rows, list(external_rows[0]))

    e2e_rows = []
    for name, display, report in (("v1", "v1", e2e_v1), ("v2_3", "v2.3", e2e_v23)):
        e2e_rows.append(
            {
                "system": display,
                "token_f1": f"{report['mean_pseudo_gloss_token_f1']:.4f}",
                "sequence_similarity": f"{report['mean_pseudo_gloss_sequence_similarity']:.4f}",
                "speech_laal_seconds": f"{report['mean_laal_seconds']:.4f}",
                "compute_adjusted_laal_seconds": f"{report['mean_compute_adjusted_laal_seconds']:.4f}",
                "completion_percent": f"{100 * report['valid_end_completion_rate']:.1f}",
                "realtime_factor": f"{report['mean_processing_realtime_factor']:.4f}",
            }
        )
    write_tsv(tables / "table_synthetic_mandarin_e2e100.tsv", e2e_rows, list(e2e_rows[0]))

    paired_rows = []
    for metric in ("token_f1", "sequence_similarity", "laal"):
        result = waitk_pair["paired_metrics"][metric]
        paired_rows.append(
            {
                "comparison": "v2.3 - modified Wait-k v1 k=1",
                "metric": metric,
                "mean_delta": result["mean_delta"],
                "ci95_lower": result["ci95_lower"],
                "ci95_upper": result["ci95_upper"],
                "conclusion": result["superiority_conclusion"],
            }
        )
    for metric in ("pseudo_gloss_token_f1", "speech_to_gloss_laal_seconds", "speech_to_gloss_compute_adjusted_laal_seconds"):
        result = e2e_pair["paired_deltas"][metric]
        paired_rows.append(
            {
                "comparison": "v2.3 - v1 synthetic Mandarin E2E",
                "metric": metric,
                "mean_delta": result["mean_candidate_minus_baseline"],
                "ci95_lower": result["ci95_lower"],
                "ci95_upper": result["ci95_upper"],
                "conclusion": "ci_excludes_zero" if result["ci95_lower"] > 0 or result["ci95_upper"] < 0 else "ci_crosses_zero",
            }
        )
    write_tsv(tables / "table_key_paired_effects.tsv", paired_rows, list(paired_rows[0]))

    plot_frontier(waitk_rows, locked, figures)
    plot_paired_effects(locked, waitk_pair, e2e_pair, figures)

    summary = {
        "scientific_status": "paper_artifacts_from_frozen_aggregate_reports",
        "selection_used_locked_test": False,
        "primary_claim": (
            "On locked pseudo-gloss evaluation, v2.3 is quality-noninferior to the "
            "strongest modified Wait-k point and has significantly lower LAAL."
        ),
        "locked500_vs_waitk_k1": waitk_pair["paper_gate"],
        "speech100": {
            "token_f1_delta": e2e_pair["paired_deltas"]["pseudo_gloss_token_f1"],
            "algorithmic_laal_delta_seconds": e2e_pair["paired_deltas"]["speech_to_gloss_laal_seconds"],
            "compute_adjusted_laal_delta_seconds": e2e_pair["paired_deltas"]["speech_to_gloss_compute_adjusted_laal_seconds"],
        },
        "external_epee": epee["decision_summary"],
        "limitations": [
            "The primary locked target is weak pseudo ASL gloss, not expert gold ASL.",
            "The speech pilot uses synthetic Mandarin speech and 100 examples.",
            "Algorithmic and compute-adjusted latency must be reported separately.",
            "Epee is an external clip-level human-gloss corpus, not a direct text-to-gloss benchmark.",
        ],
    }
    (output / "paper_results_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
