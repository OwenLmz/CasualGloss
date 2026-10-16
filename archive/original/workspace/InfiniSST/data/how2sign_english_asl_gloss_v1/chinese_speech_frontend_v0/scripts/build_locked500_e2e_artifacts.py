#!/usr/bin/env python3
"""Build paper-ready tables, figures, and a factual locked-500 E2E report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt


METRICS = [
    ("pseudo_gloss_token_f1", "Pseudo-gloss token F1", "quality"),
    ("pseudo_gloss_sequence_similarity", "Sequence similarity", "quality"),
    ("pseudo_gloss_exact_match", "Exact match", "quality"),
    ("speech_to_gloss_completed_after_full_source", "Valid completion", "quality"),
    ("speech_to_gloss_reference_length_al_seconds", "Reference-length AL (s)", "latency"),
    ("speech_to_gloss_laal_seconds", "LAAL (s)", "latency"),
    ("speech_to_gloss_dal_seconds", "DAL (s)", "latency"),
    ("speech_to_gloss_compute_adjusted_laal_seconds", "Compute-adjusted LAAL (s)", "latency"),
    ("speech_to_gloss_processing_realtime_factor", "Processing RTF", "latency"),
    ("prediction_reference_length_absolute_error", "Relative length error", "latency"),
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-report", type=Path, required=True)
    parser.add_argument("--v2-3-report", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--frontend-report", type=Path, required=True)
    parser.add_argument("--translation-report", type=Path, required=True)
    parser.add_argument("--audio-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    v1 = read_json(args.v1_report)
    v23 = read_json(args.v2_3_report)
    comparison = read_json(args.comparison)
    frontend = read_json(args.frontend_report)
    translation = read_json(args.translation_report)
    audio = read_json(args.audio_report)
    output_root = args.output_root
    table_root = output_root / "tables"
    figure_root = output_root / "figures"
    table_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    main_rows: list[dict[str, Any]] = []
    for key, label, family in METRICS:
        means = comparison.get("means", {}).get(key)
        delta = comparison.get("paired_deltas", {}).get(key)
        if not means or not delta:
            continue
        main_rows.append(
            {
                "metric": label,
                "family": family,
                "v1": means["v1"],
                "v2_3": means["v2_3"],
                "delta_v2_3_minus_v1": delta["mean_candidate_minus_baseline"],
                "ci95_lower": delta["ci95_lower"],
                "ci95_upper": delta["ci95_upper"],
                "higher_is_better": means["higher_is_better"],
                "paired_n": delta["count"],
                "bootstrap_samples": comparison["bootstrap_samples"],
            }
        )
    write_tsv(table_root / "table_locked500_e2e.tsv", main_rows)

    frontend_rows = [
        {"metric": key, "value": value}
        for key, value in {
            "record_count": frontend["record_count"],
            "corpus_wer": frontend["translation_quality"]["corpus_wer"],
            "sacrebleu": frontend["translation_quality"]["sacrebleu"],
            "chrf": frontend["translation_quality"]["chrf"],
            "mean_first_commit_seconds": frontend["streaming_behavior"]["first_commit_audio_seconds"]["mean"],
            "translation_completed_count": translation["completed_count"],
            "translation_flagged_count": translation["flagged_count"],
            "valid_audio_count": audio["status_counts"].get("valid", 0),
            "flagged_audio_count": audio["status_counts"].get("flagged", 0),
        }.items()
    ]
    write_tsv(table_root / "table_locked500_frontend_audit.tsv", frontend_rows)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    blue = "#3478B8"
    orange = "#D97706"
    green = "#16856B"
    neutral = "#4B5563"

    fig = plt.figure(figsize=(7.2, 3.35), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.15])

    ax = fig.add_subplot(grid[0, 0])
    v1_f1 = comparison["means"]["pseudo_gloss_token_f1"]["v1"]
    v23_f1 = comparison["means"]["pseudo_gloss_token_f1"]["v2_3"]
    v1_laal = comparison["means"]["speech_to_gloss_laal_seconds"]["v1"]
    v23_laal = comparison["means"]["speech_to_gloss_laal_seconds"]["v2_3"]
    ax.scatter([v1_laal], [v1_f1], s=55, color=blue, zorder=3, label="v1")
    ax.scatter([v23_laal], [v23_f1], s=55, color=green, zorder=3, label="v2.3")
    ax.annotate(
        "",
        xy=(v23_laal, v23_f1),
        xytext=(v1_laal, v1_f1),
        arrowprops={"arrowstyle": "->", "color": neutral, "lw": 1.1},
    )
    ax.annotate("v1", (v1_laal, v1_f1), xytext=(5, -11), textcoords="offset points", color=blue)
    ax.annotate("v2.3", (v23_laal, v23_f1), xytext=(5, 5), textcoords="offset points", color=green)
    ax.set_xlabel("Speech-time LAAL (s)  ← lower is better")
    ax.set_ylabel("Pseudo-gloss token F1  (higher is better)")
    ax.set_title("Quality-latency operating point", loc="left", fontweight="bold")
    ax.grid(color="#E5E7EB", linewidth=0.6, zorder=0)

    quality_keys = ["pseudo_gloss_token_f1", "pseudo_gloss_sequence_similarity"]
    quality_labels = ["Token F1", "Sequence\nsimilarity"]
    ax = fig.add_subplot(grid[0, 1])
    for position, (key, label) in enumerate(zip(quality_keys, quality_labels)):
        effect = comparison["paired_deltas"][key]
        center = effect["mean_candidate_minus_baseline"]
        lower = effect["ci95_lower"]
        upper = effect["ci95_upper"]
        ax.errorbar(
            center,
            position,
            xerr=[[center - lower], [upper - center]],
            fmt="o",
            color=green,
            ecolor=green,
            capsize=3,
            markersize=4.5,
        )
    ax.axvline(0, color=neutral, linewidth=0.8, linestyle="--")
    ax.set_yticks(range(len(quality_labels)), quality_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Paired Δ (v2.3 − v1)")
    ax.set_title("Quality effects, 95% CI", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.6)

    latency_keys = [
        "speech_to_gloss_reference_length_al_seconds",
        "speech_to_gloss_laal_seconds",
        "speech_to_gloss_dal_seconds",
        "speech_to_gloss_compute_adjusted_laal_seconds",
    ]
    latency_labels = ["Ref. AL", "LAAL", "DAL", "Compute-adj.\nLAAL"]
    ax = fig.add_subplot(grid[0, 2])
    for position, (key, label) in enumerate(zip(latency_keys, latency_labels)):
        effect = comparison["paired_deltas"][key]
        center = effect["mean_candidate_minus_baseline"]
        lower = effect["ci95_lower"]
        upper = effect["ci95_upper"]
        color = green if upper < 0 else orange
        ax.errorbar(
            center,
            position,
            xerr=[[center - lower], [upper - center]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=4.5,
        )
    ax.axvline(0, color=neutral, linewidth=0.8, linestyle="--")
    ax.set_yticks(range(len(latency_labels)), latency_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Paired Δ seconds (v2.3 − v1)\n← lower latency")
    ax.set_title("Latency effects, 95% CI", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.6)

    fig.suptitle(
        f"Locked synthetic-Mandarin end-to-end evaluation (paired n={comparison['common_record_count']})",
        fontsize=9,
        fontweight="bold",
    )
    stem = figure_root / "fig_locked500_speech_e2e"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    checks = comparison["predeclared_checks"]
    report_lines = [
        "# Locked-500 Chinese-Speech End-to-End Evaluation",
        "",
        "## Scope",
        "",
        "- Fixed IDs: the pre-existing locked 500 test parents; no model or policy selection used these results.",
        "- Input: deterministic English-to-Mandarin translation followed by four-speaker synthetic TTS.",
        "- Frontend: the already selected SeamlessStreaming 2.5B monotonic S2TT system.",
        "- Comparison: v1 versus v2.3 on identical timestamped incremental English commits.",
        "- References: weak pseudo-ASL gloss, not expert gold ASL.",
        "",
        "## Main results",
        "",
        "| Metric | v1 | v2.3 | Paired delta | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label, _ in METRICS:
        means = comparison.get("means", {}).get(key)
        effect = comparison.get("paired_deltas", {}).get(key)
        if not means or not effect:
            continue
        report_lines.append(
            f"| {label} | {fmt(means['v1'])} | {fmt(means['v2_3'])} | "
            f"{fmt(effect['mean_candidate_minus_baseline'])} | "
            f"[{fmt(effect['ci95_lower'])}, {fmt(effect['ci95_upper'])}] |"
        )
    report_lines.extend(
        [
            "",
            "## Predeclared interpretation checks",
            "",
            *[f"- {key}: **{value}**" for key, value in checks.items()],
            f"- all checks accepted: **{comparison['accepted']}**",
            "",
            "## Frontend audit",
            "",
            f"- English BLEU: {fmt(frontend['translation_quality']['sacrebleu'])}",
            f"- English chrF: {fmt(frontend['translation_quality']['chrf'])}",
            f"- English WER: {fmt(frontend['translation_quality']['corpus_wer'])}",
            f"- Valid audio files: {audio['status_counts'].get('valid', 0)}/{audio['record_count']}",
            "",
            "## Required limitations",
            "",
            "1. Mandarin audio is synthetic, not recorded from real speakers or noisy environments.",
            "2. Pseudo-gloss content metrics do not establish expert ASL correctness.",
            "3. Speech-time latency excludes model runtime; compute-adjusted LAAL is reported separately.",
            "4. Synthetic Mandarin was generated from the English reference, so this is controlled system evaluation rather than a natural bilingual corpus.",
            "",
        ]
    )
    (output_root / "EXPERIMENT_REPORT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    summary = {
        "record_count": comparison["common_record_count"],
        "frontend_ok": frontend["ok"],
        "translation_ok": translation["ok"],
        "audio_ok": audio["ok"],
        "v1_ok": v1["ok"],
        "v2_3_ok": v23["ok"],
        "paired_comparison_accepted": comparison["accepted"],
        "predeclared_checks": checks,
        "artifacts": {
            "main_table": str(table_root / "table_locked500_e2e.tsv"),
            "frontend_table": str(table_root / "table_locked500_frontend_audit.tsv"),
            "figure_stem": str(stem),
            "report": str(output_root / "EXPERIMENT_REPORT.md"),
        },
    }
    (output_root / "artifact_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
