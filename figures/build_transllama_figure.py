"""Portable content-equivalent redraw of submitted Figure 1, not pixel-identical."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


COLORS = {
    "ink": "#17202A",
    "muted": "#5E6B78",
    "grid": "#E4E8EC",
    "fixed": "#63788E",
    "adaptive": "#D97706",
    "adaptive_light": "#F4C88C",
    "causal": "#087F78",
    "causal_light": "#BFE2DE",
    "zero": "#9AA4AE",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.7,
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def load_inputs(report_path: Path, seed_path: Path):
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    if report.get("result_kind") == "original_paper_aggregates":
        expected = json.loads(
            (Path(__file__).resolve().parents[1] / "results/expected_paper_results.json")
            .read_text(encoding="utf-8-sig")
        )
        if report.get("paper_sha256") != expected["paper_sha256"]:
            raise ValueError("Original-paper result anchor differs")
        for system, metrics in expected["table1"].items():
            for metric, (mean, sd) in metrics.items():
                digits = 2 if metric == "bleu4" else 4
                actual = report["locked500"]["table1"][system][metric]
                if round(actual["mean"], digits) != mean or round(actual["sample_sd"], digits) != sd:
                    raise ValueError(f"Paper display value differs: {system}/{metric}")
    elif report.get("all_numeric_checks_pass") is not True:
        raise ValueError("Input must be original-paper aggregates or a successful scoring report")
    locked = report["locked500"]
    seed_report = json.loads(seed_path.read_text(encoding="utf-8-sig"))
    if seed_report["paper_sha256"] != report["paper_sha256"]:
        raise ValueError("Paper anchors differ")
    causal = pd.DataFrame(seed_report["causalgloss_seed_means"])
    causal = causal.rename(columns={"laal": "laal_source_tokens"}).sort_values("seed")
    fixed = pd.DataFrame([
        {"k": int(k), "token_f1": r["token_f1"]["mean"], "token_f1_sd": r["token_f1"]["sample_sd"],
         "laal": r["laal"]["mean"], "laal_sd": r["laal"]["sample_sd"]}
        for k, r in locked["fixed_frontier"].items()]).sort_values("k")
    adaptive = pd.DataFrame([
        {"k": int(k), "token_f1": r["token_f1"]["mean"], "token_f1_sd": r["token_f1"]["sample_sd"],
         "laal": r["laal"]["mean"], "laal_sd": r["laal"]["sample_sd"]}
        for k, r in locked["learned_frontier"].items()]).sort_values("k")
    bootstrap_rows = []
    for k, entry in locked["learned_comparison_bootstrap"].items():
        row = {"k": int(k)}
        for metric in ("token_f1", "laal"):
            value = entry[metric]
            row[f"{metric}_delta"] = value["mean_difference"]
            row[f"{metric}_lower"], row[f"{metric}_upper"] = value["ci95"]
        bootstrap_rows.append(row)
    bootstrap = pd.DataFrame(bootstrap_rows).sort_values("k")
    for metric, column in (("token_f1", "token_f1"), ("laal", "laal_source_tokens")):
        actual = causal[column]
        expected = locked["table1"]["causalgloss"][metric]
        if not np.isclose(actual.mean(), expected["mean"], atol=1e-12, rtol=0):
            raise ValueError(f"Seed mean mismatch for {metric}")
        if not np.isclose(actual.std(ddof=1), expected["sample_sd"], atol=1e-12, rtol=0):
            raise ValueError(f"Seed SD mismatch for {metric}")

    if fixed["k"].tolist() != [1, 3, 5, 7]:
        raise ValueError("Fixed Wait-k curve is incomplete")
    if adaptive["k"].tolist() != list(range(1, 9)):
        raise ValueError("Learned adaptive k=1..8 sweep is incomplete")
    if bootstrap["k"].tolist() != list(range(1, 9)):
        raise ValueError("Paired k=1..8 intervals are incomplete")
    if len(causal) != 3:
        raise ValueError("Expected three CausalGloss seeds")
    if causal["seed"].tolist() != [20260829, 20260904, 20260905]:
        raise ValueError("Expected the three final seed identities")

    return fixed, causal, adaptive, bootstrap


def draw_delta_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    ylim: tuple[float, float],
) -> None:
    values = frame[f"{metric}_delta"].to_numpy(dtype=float)
    lower = frame[f"{metric}_lower"].to_numpy(dtype=float)
    upper = frame[f"{metric}_upper"].to_numpy(dtype=float)
    k = frame["k"].to_numpy(dtype=int)
    ax.axhline(0.0, color=COLORS["zero"], linewidth=0.8, linestyle=(0, (2.2, 2.2)))
    ax.errorbar(
        k,
        values,
        yerr=np.vstack([values - lower, upper - values]),
        fmt="o-",
        markersize=3.5,
        markerfacecolor=COLORS["causal"],
        markeredgecolor="white",
        markeredgewidth=0.45,
        color=COLORS["causal"],
        ecolor=COLORS["causal"],
        linewidth=0.9,
        elinewidth=0.9,
        capsize=1.8,
        zorder=3,
    )
    ax.set_xlim(0.65, 8.35)
    ax.set_ylim(*ylim)
    ax.set_xticks(np.arange(1, 9))
    ax.set_ylabel(ylabel, labelpad=2)
    ax.set_title(title, loc="left", pad=4, fontweight="bold")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, zorder=0)
    ax.tick_params(axis="both", length=2.5, width=0.6)


def add_panel_label(
    ax: plt.Axes,
    label: str,
    left_offset_pt: float = 10.0,
    top_offset_pt: float = 4.0,
) -> None:
    figure_width_pt, figure_height_pt = ax.figure.get_size_inches() * 72.0
    bbox = ax.get_position()
    x = -left_offset_pt / (bbox.width * figure_width_pt)
    y = 1.0 + top_offset_pt / (bbox.height * figure_height_pt)
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def build_figure(args) -> None:
    configure_style()
    fixed, causal, adaptive, bootstrap = load_inputs(args.input_report, args.seed_summary)

    fig = plt.figure(figsize=(7.20, 3.95), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.24, 1.0),
        height_ratios=(1, 1),
        left=0.085,
        right=0.985,
        bottom=0.16,
        top=0.88,
        wspace=0.42,
        hspace=0.94,
    )
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])

    ax_a.plot(
        fixed["laal"],
        fixed["token_f1"],
        color=COLORS["fixed"],
        linewidth=1.2,
        marker="o",
        markersize=4.0,
        markerfacecolor="white",
        markeredgewidth=1.0,
        zorder=2,
    )
    repeated_fixed = fixed[fixed["k"].isin([5, 7])]
    ax_a.errorbar(
        repeated_fixed["laal"],
        repeated_fixed["token_f1"],
        xerr=repeated_fixed["laal_sd"],
        yerr=repeated_fixed["token_f1_sd"],
        fmt="none",
        ecolor=COLORS["fixed"],
        elinewidth=0.8,
        capsize=1.7,
        zorder=2,
    )
    fixed_offsets = {1: (-2, 8), 3: (-2, 8), 5: (-2, 8), 7: (-10, -12)}
    for row in fixed.itertuples(index=False):
        dx, dy = fixed_offsets[int(row.k)]
        ax_a.annotate(
            f"{int(row.k)}",
            (row.laal, row.token_f1),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.0,
            color=COLORS["fixed"],
        )

    ax_a.errorbar(
        adaptive["laal"],
        adaptive["token_f1"],
        xerr=adaptive["laal_sd"],
        yerr=adaptive["token_f1_sd"],
        fmt="o-",
        markersize=3.8,
        markerfacecolor=COLORS["adaptive_light"],
        markeredgecolor=COLORS["adaptive"],
        markeredgewidth=0.8,
        color=COLORS["adaptive"],
        ecolor=COLORS["adaptive"],
        linewidth=1.2,
        elinewidth=0.75,
        capsize=1.5,
        zorder=3,
    )
    adaptive_offsets = {1: (-8, -12), 4: (-2, 8), 5: (-2, -13), 8: (-4, 8)}
    for row in adaptive.itertuples(index=False):
        if int(row.k) not in adaptive_offsets:
            continue
        dx, dy = adaptive_offsets[int(row.k)]
        ax_a.annotate(
            f"{int(row.k)}",
            (row.laal, row.token_f1),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.0,
            color=COLORS["adaptive"],
        )

    mean_laal = float(causal["laal_source_tokens"].mean())
    sd_laal = float(causal["laal_source_tokens"].std(ddof=1))
    mean_f1 = float(causal["token_f1"].mean())
    sd_f1 = float(causal["token_f1"].std(ddof=1))
    ax_a.scatter(
        causal["laal_source_tokens"],
        causal["token_f1"],
        s=27,
        marker="D",
        facecolor=COLORS["causal_light"],
        edgecolor=COLORS["causal"],
        linewidth=0.9,
        zorder=5,
    )
    ax_a.errorbar(
        mean_laal,
        mean_f1,
        xerr=sd_laal,
        yerr=sd_f1,
        fmt="*",
        markersize=9.5,
        markerfacecolor=COLORS["causal"],
        markeredgecolor="white",
        markeredgewidth=0.55,
        color=COLORS["causal"],
        ecolor=COLORS["causal"],
        elinewidth=1.05,
        capsize=2.2,
        zorder=6,
    )

    adaptive_k5 = adaptive[adaptive["k"] == 5].iloc[0]
    data_top = float((adaptive["token_f1"] + adaptive["token_f1_sd"]).max())
    reduction = 100 * (1 - mean_laal / float(adaptive_k5["laal"]))
    ax_a.text(
        0.5,
        0.97,
        f"{reduction:.1f}% lower LAAL\nat near-matched F1 (k=5)",
        transform=ax_a.transAxes,
        ha="center",
        va="top",
        fontsize=8.0,
        fontweight="bold",
        color=COLORS["causal"],
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["fixed"],
            marker="o",
            markersize=4.0,
            markerfacecolor="white",
            linewidth=1.2,
            label="Fixed Wait-$k$ (labels: $k$)",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["adaptive"],
            marker="o",
            markersize=3.8,
            markerfacecolor=COLORS["adaptive_light"],
            linewidth=1.2,
            label=r"TransLLaMA-style mean $\pm$ SD",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="D",
            markersize=4.5,
            markerfacecolor=COLORS["causal_light"],
            markeredgecolor=COLORS["causal"],
            label="CausalGloss seeds",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="*",
            markersize=7.5,
            markerfacecolor=COLORS["causal"],
            markeredgecolor=COLORS["causal"],
            label=r"CausalGloss mean $\pm$ SD",
        ),
    ]
    legend = ax_a.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        framealpha=0.97,
        edgecolor="#CBD5E1",
        borderpad=0.42,
        labelspacing=0.45,
        handlelength=1.45,
        handletextpad=0.42,
    )
    legend.get_frame().set_linewidth(0.6)

    ax_a.set_xlim(-1.45, 5.60)
    ax_a.set_ylim(0.285, data_top + 0.14)
    ax_a.set_xlabel("LAAL (source tokens; lower is better)")
    ax_a.set_ylabel("Token F1 (higher is better)")
    ax_a.set_title("Quality-latency operating points", loc="left", pad=5, fontweight="bold")
    ax_a.tick_params(axis="both", length=2.5, width=0.6)

    draw_delta_panel(
        ax_b,
        bootstrap,
        metric="token_f1",
        title="Paired quality difference",
        ylabel=r"$\Delta$ Token F1",
        ylim=(-0.055, 0.055),
    )
    draw_delta_panel(
        ax_c,
        bootstrap,
        metric="laal",
        title="Paired latency difference",
        ylabel=r"$\Delta$ LAAL",
        ylim=(-4.05, -0.75),
    )
    ax_b.tick_params(labelbottom=False)
    ax_c.set_xlabel("TransLLaMA-style minimum delay $k$", labelpad=2)

    add_panel_label(ax_a, "a")
    add_panel_label(ax_b, "b")
    add_panel_label(ax_c, "c")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.qa_output.mkdir(parents=True, exist_ok=True)
    base = args.output_dir / "figure1"
    fig.canvas.draw()
    if args.qa_tools:
        sys.path.insert(0, str(args.qa_tools))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(
            fig, axes=[ax_a, ax_b, ax_c], panel_ids=["a", "b", "c"],
            json_out=args.qa_output / "figure1.alignment.json",
            overlay_svg=args.qa_output / "figure1.alignment.svg",
            tolerance_pt=1.5, gutter_tolerance_pt=1.5,
            require_panel_labels=True, strict=True)
    else:
        print("NOTE: external render-time alignment QA not run; enable --qa-tools to audit.")
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".png"), dpi=600)
    fig.savefig(base.with_suffix(".tiff"), dpi=600)
    plt.close(fig)

    summary = {
        "fixed_wait_k": fixed.astype(object).where(pd.notna(fixed), None).to_dict(orient="records"),
        "causalgloss": {
            "token_f1_mean": mean_f1,
            "token_f1_sample_sd": sd_f1,
            "laal_mean": mean_laal,
            "laal_sample_sd": sd_laal,
        },
        "transllama_style": adaptive.to_dict(orient="records"),
        "paired_intervals": bootstrap.to_dict(orient="records"),
    }
    summary.update({"input_report_sha256": hashlib.sha256(args.input_report.read_bytes()).hexdigest(),
                    "seed_summary_sha256": hashlib.sha256(args.seed_summary.read_bytes()).hexdigest(),
                    "status": "portable content-equivalent redraw, not pixel-identical to submitted PDF",
                    "alignment_qa_enabled": bool(args.qa_tools), "minimum_requested_font_pt": 8})
    (args.qa_output / "figure1.source_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-summary", type=Path, default=Path(__file__).with_name("figure1_seed_summary.json"))
    parser.add_argument("--qa-tools", type=Path, help="Optional external nature-figure scripts directory")
    parser.add_argument("--qa-output", type=Path, default=Path("figure_qa"))
    build_figure(parser.parse_args())
