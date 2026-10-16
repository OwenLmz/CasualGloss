from __future__ import annotations

import json
import sys
from pathlib import Path

LOCAL_PACKAGES = Path(r"D:\Infinisst\.paper_python_packages")
sys.path.insert(0, str(LOCAL_PACKAGES))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(r"D:\Infinisst")
WORKING = ROOT / "paper_revision_20260919_transllama" / "working_v1"
FIGURES = WORKING / "figures"
QA = ROOT / "paper_revision_20260919_transllama" / "qa"

FIXED_PATH = ROOT / "paper_revision_20260916" / "outputs" / "fixed_waitk_summary.tsv"
CAUSAL_PATH = (
    ROOT / "icassp2027_paper" / "source_data" / "raw" / "v2_3_text_locked500_multiseed.csv"
)
ADAPTIVE_PATH = ROOT / "transllama_final_three_seed_aggregate.json"
BOOTSTRAP_PATH = ROOT / "transllama_vs_causalgloss_bootstrap.json"

SKILL_SCRIPTS = Path(
    r"C:\Users\DELL\.codex\plugins\cache\nature-skills\nature-skills\0.1.0"
    r"\skills\nature-figure\scripts"
)
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


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
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 5.9,
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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixed_all = pd.read_csv(FIXED_PATH, sep="\t")
    fixed = fixed_all[
        fixed_all["system"].isin(["waitk1", "waitk3", "waitk5", "waitk7"])
    ].copy()
    fixed["k"] = fixed["system"].str.extract(r"(\d+)$").astype(int)
    fixed["token_f1_sd"] = np.nan
    fixed["laal_sd"] = np.nan
    for k in (5, 7):
        spread = fixed_all[fixed_all["system"] == f"waitk{k}_sample_sd"].iloc[0]
        fixed.loc[fixed["k"] == k, "token_f1_sd"] = spread["token_f1"]
        fixed.loc[fixed["k"] == k, "laal_sd"] = spread["laal"]
    fixed = fixed.sort_values("k")

    causal = pd.read_csv(CAUSAL_PATH).sort_values("seed")

    adaptive_report = json.loads(ADAPTIVE_PATH.read_text(encoding="utf-8"))
    if not adaptive_report.get("all_checks_pass"):
        raise ValueError("The learned adaptive aggregate did not pass all checks")
    adaptive_rows = []
    for row in adaptive_report["aggregate_by_k"]:
        metrics = row["metrics"]
        adaptive_rows.append(
            {
                "k": int(row["wait_k"]),
                "token_f1": metrics["token_f1"]["mean"],
                "token_f1_sd": metrics["token_f1"]["sample_sd"],
                "sequence_similarity": metrics["sequence_similarity"]["mean"],
                "sequence_similarity_sd": metrics["sequence_similarity"]["sample_sd"],
                "bleu4": metrics["bleu4"]["mean"],
                "bleu4_sd": metrics["bleu4"]["sample_sd"],
                "laal": metrics["laal"]["mean"],
                "laal_sd": metrics["laal"]["sample_sd"],
                "completion_rate": metrics["completion_rate"]["mean"],
                "invalid_action_ratio": metrics["invalid_action_ratio"]["mean"],
            }
        )
    adaptive = pd.DataFrame(adaptive_rows).sort_values("k")

    bootstrap_report = json.loads(BOOTSTRAP_PATH.read_text(encoding="utf-8"))
    if not bootstrap_report.get("all_checks_pass"):
        raise ValueError("The paired learned-baseline bootstrap did not pass all checks")
    bootstrap_rows = []
    for row in bootstrap_report["results"]:
        output = {"k": int(row["wait_k"])}
        for metric in ("token_f1", "laal"):
            aggregate = row["metrics"][metric]["aggregate"]
            interval = aggregate["hierarchical_seed_and_parent_bootstrap_95_ci"]
            output[f"{metric}_delta"] = aggregate[
                "paired_mean_difference_causalgloss_minus_baseline"
            ]
            output[f"{metric}_lower"] = interval["lower"]
            output[f"{metric}_upper"] = interval["upper"]
        bootstrap_rows.append(output)
    bootstrap = pd.DataFrame(bootstrap_rows).sort_values("k")

    if fixed["k"].tolist() != [1, 3, 5, 7]:
        raise ValueError("Fixed Wait-k curve is incomplete")
    if adaptive["k"].tolist() != list(range(1, 9)):
        raise ValueError("Learned adaptive k=1..8 sweep is incomplete")
    if bootstrap["k"].tolist() != list(range(1, 9)):
        raise ValueError("Paired k=1..8 intervals are incomplete")
    if len(causal) != 3:
        raise ValueError("Expected three CausalGloss seeds")

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


def build_figure() -> None:
    configure_style()
    fixed, causal, adaptive, bootstrap = load_inputs()

    fig = plt.figure(figsize=(7.10, 3.15), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.24, 1.0),
        height_ratios=(1, 1),
        left=0.072,
        right=0.985,
        bottom=0.17,
        top=0.91,
        wspace=0.34,
        hspace=0.72,
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
            fontsize=6.2,
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
            fontsize=6.2,
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
    annotation_y = 0.755
    ax_a.annotate(
        "",
        xy=(mean_laal, annotation_y),
        xytext=(float(adaptive_k5["laal"]), annotation_y),
        arrowprops={
            "arrowstyle": "<|-|>",
            "color": COLORS["causal"],
            "linewidth": 0.9,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    ax_a.text(
        0.5 * (mean_laal + float(adaptive_k5["laal"])),
        annotation_y + 0.006,
        "54.4% lower LAAL at near-matched F1",
        ha="center",
        va="bottom",
        fontsize=6.25,
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
    ax_a.set_ylim(0.285, 0.780)
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

    FIGURES.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    base = FIGURES / "fig_quality_latency"
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        axes=[ax_a, ax_b, ax_c],
        panel_ids=["a", "b", "c"],
        json_out=QA / "fig_quality_latency.alignment.json",
        overlay_svg=QA / "fig_quality_latency.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".png"), dpi=600)
    fig.savefig(base.with_suffix(".tiff"), dpi=600)
    plt.close(fig)

    summary = {
        "fixed_wait_k": fixed.to_dict(orient="records"),
        "causalgloss": {
            "token_f1_mean": mean_f1,
            "token_f1_sample_sd": sd_f1,
            "laal_mean": mean_laal,
            "laal_sample_sd": sd_laal,
        },
        "transllama_style": adaptive.to_dict(orient="records"),
        "paired_intervals": bootstrap.to_dict(orient="records"),
    }
    (QA / "fig_quality_latency.source_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    build_figure()
