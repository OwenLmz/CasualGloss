from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "source_data" / "raw"
DERIVED = ROOT / "source_data" / "derived"
FIGURES = ROOT / "manuscript" / "figures"
TABLES = ROOT / "manuscript" / "tables"
QA = ROOT / "qa"

AUDIT_SCRIPTS = Path(
    r"C:\Users\DELL\.codex\plugins\cache\nature-skills\nature-skills\0.1.0"
    r"\skills\nature-figure\scripts"
)
sys.path.insert(0, str(AUDIT_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


WAIT_K_PATH = RAW / "exact_transllama_wait_k_summary.tsv"
SEEDS_PATH = RAW / "v2_3_text_locked500_multiseed.csv"
PAIRED_PATHS = {
    20260829: RAW / "exact_transllama_v2_3_seed20260829_paired.json",
    20260904: RAW / "exact_transllama_v2_3_seed20260904_paired.json",
    20260905: RAW / "exact_transllama_v2_3_seed20260905_paired.json",
}
ABLATION_PATH = RAW / "confirmation_ablation.json"
EPEE_PATH = RAW / "epee_comparison.json"
EPEE_GENERATOR_PATH = RAW / "epee_generator_human_agreement.json"
BLEU_PATH = RAW / "locked500_bleu_report.json"
SPEECH_PATH = RAW / "v2_3_multiseed_summary.json"
PROGRESSION_PATH = RAW / "method_progression_locked500.json"


COLORS = {
    "ink": "#17202A",
    "muted": "#5E6B78",
    "grid": "#E4E8EC",
    "waitk": "#63788E",
    "waitk_light": "#DCE4EA",
    "causal": "#087F78",
    "causal_light": "#BFE2DE",
    "accent": "#D97706",
    "zero": "#9AA4AE",
    "band": "#F0F3F5",
}


def read_inputs() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict, dict, dict, dict, dict
]:
    waitk = pd.read_csv(WAIT_K_PATH, sep="\t")
    seeds = pd.read_csv(SEEDS_PATH)
    paired_rows = []
    for seed, path in PAIRED_PATHS.items():
        with path.open(encoding="utf-8") as handle:
            report = json.load(handle)
        if report["common_parent_count"] != 500:
            raise ValueError(f"Seed {seed} paired report is not locked-500")
        if not report["same_parent_ids"] or not report["same_source_and_reference"]:
            raise ValueError(f"Seed {seed} does not use identical paired inputs")
        if report["bootstrap"]["repetitions"] != 10000:
            raise ValueError(f"Seed {seed} does not use 10,000 bootstrap resamples")
        for metric in ("token_f1", "sequence_similarity", "laal"):
            values = report["paired_metrics"][metric]
            paired_rows.append(
                {
                    "evaluation": "text_locked500_vs_exact_transllama_v1_k1",
                    "seed": seed,
                    "metric": metric,
                    "mean_delta": values["mean_delta"],
                    "ci95_lower": values["ci95_lower"],
                    "ci95_upper": values["ci95_upper"],
                }
            )
    paired = pd.DataFrame(paired_rows)
    with ABLATION_PATH.open(encoding="utf-8") as handle:
        ablation = json.load(handle)
    with EPEE_PATH.open(encoding="utf-8") as handle:
        epee = json.load(handle)
    with EPEE_GENERATOR_PATH.open(encoding="utf-8") as handle:
        epee_generator = json.load(handle)
    with BLEU_PATH.open(encoding="utf-8") as handle:
        bleu = json.load(handle)
    with SPEECH_PATH.open(encoding="utf-8") as handle:
        speech = json.load(handle)
    with PROGRESSION_PATH.open(encoding="utf-8") as handle:
        progression = json.load(handle)
    return waitk, seeds, paired, ablation, epee, epee_generator, bleu, speech, progression


def validate_inputs(waitk: pd.DataFrame, seeds: pd.DataFrame, paired: pd.DataFrame) -> None:
    frontier = waitk[(waitk["model"] == "v1") & (waitk["policy"] == "modified_wait_k")]
    observed_k = frontier["k"].astype(int).tolist()
    if sorted(observed_k) != list(range(1, 9)):
        raise ValueError(f"Expected complete k=1..8 sweep, found {observed_k}")
    expected_seeds = {20260829, 20260904, 20260905}
    if set(seeds["seed"].astype(int)) != expected_seeds:
        raise ValueError("Unexpected or missing CausalGloss seeds")
    text = paired[paired["evaluation"] == "text_locked500_vs_exact_transllama_v1_k1"]
    if set(text["seed"].astype(int)) != expected_seeds:
        raise ValueError("Paired locked-500 confidence intervals are incomplete")
    if set(text["metric"]) != {"token_f1", "sequence_similarity", "laal"}:
        raise ValueError("Unexpected paired metric set")
    numeric = np.concatenate(
        [
            frontier[["token_f1", "sequence_similarity", "laal"]].to_numpy(dtype=float).ravel(),
            seeds[["token_f1", "sequence_similarity", "laal_source_tokens"]]
            .to_numpy(dtype=float)
            .ravel(),
            text[["mean_delta", "ci95_lower", "ci95_upper"]].to_numpy(dtype=float).ravel(),
        ]
    )
    if not np.isfinite(numeric).all():
        raise ValueError("Non-finite values found in plotting inputs")


def validate_progression(progression: dict) -> None:
    if progression.get("common_parent_count") != 500:
        raise ValueError("Method progression must use all 500 locked parents")
    if not progression.get("same_parent_ids") or not progression.get("same_source_and_reference"):
        raise ValueError("Method progression inputs are not paired on identical examples")
    if progression.get("selection_used_locked_test") is not False:
        raise ValueError("Locked-test selection status is not explicitly false")
    expected = {"v1", "v2_1", "v2_2", "v2_3"}
    if set(progression.get("versions", {})) != expected:
        raise ValueError("Method progression is missing a required system stage")


def validate_additional_metrics(epee_generator: dict, bleu: dict) -> None:
    if epee_generator.get("source_count") != 401:
        raise ValueError("Direct Epee generator validation must cover 401 prompts")
    if epee_generator.get("bootstrap", {}).get("repetitions") != 10000:
        raise ValueError("Direct Epee validation must use 10,000 bootstrap resamples")
    if bleu.get("locked_parent_count") != 500:
        raise ValueError("BLEU report must use all 500 locked parents")
    if not bleu.get("same_parent_ids_source_and_reference"):
        raise ValueError("BLEU systems do not share identical locked inputs")
    if bleu.get("selection_used_locked_test") is not False:
        raise ValueError("BLEU report does not explicitly rule out test-set selection")
    if bleu.get("bootstrap", {}).get("repetitions") != 10000:
        raise ValueError("BLEU report must use 10,000 paired bootstrap resamples")


def save_source_manifest() -> None:
    rows = []
    for path in sorted(RAW.iterdir()):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append((path.name, path.stat().st_size, digest))
    out = ROOT / "source_data" / "SOURCE_MANIFEST.tsv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["file", "bytes", "sha256"])
        writer.writerows(rows)


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
            "legend.fontsize": 6.6,
            "axes.linewidth": 0.7,
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


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


def draw_forest(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    title: str,
    xlabel: str,
    xlim: tuple[float, float],
    reference_line: float | None = None,
) -> None:
    subset = data[data["metric"] == metric].sort_values("seed", ascending=False).reset_index(drop=True)
    y = np.arange(len(subset))
    if reference_line is not None:
        ax.axvline(
            reference_line,
            color=COLORS["waitk_light"],
            linewidth=1.0,
            linestyle=(0, (3.0, 2.0)),
            zorder=1,
        )
    ax.axvline(0.0, color=COLORS["zero"], linewidth=0.8, linestyle=(0, (2.2, 2.2)), zorder=1)
    lower = subset["mean_delta"].to_numpy() - subset["ci95_lower"].to_numpy()
    upper = subset["ci95_upper"].to_numpy() - subset["mean_delta"].to_numpy()
    ax.errorbar(
        subset["mean_delta"],
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        markersize=4.3,
        markerfacecolor=COLORS["causal"],
        markeredgecolor="white",
        markeredgewidth=0.6,
        color=COLORS["causal"],
        ecolor=COLORS["causal"],
        elinewidth=1.1,
        capsize=2.0,
        zorder=3,
    )
    seed_labels = {20260829: "S1", 20260904: "S2", 20260905: "S3"}
    ax.set_yticks(y, [seed_labels[int(seed)] for seed in subset["seed"]])
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, len(subset) - 0.4)
    ax.set_title(title, loc="left", pad=4, fontweight="bold")
    ax.set_xlabel(xlabel, labelpad=2)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, zorder=0)
    ax.tick_params(axis="both", length=2.5, width=0.6)


def build_core_figure(waitk: pd.DataFrame, seeds: pd.DataFrame, paired: pd.DataFrame) -> None:
    configure_style()
    frontier = (
        waitk[(waitk["model"] == "v1") & (waitk["policy"] == "modified_wait_k")]
        .assign(k=lambda frame: frame["k"].astype(int))
        .sort_values("k")
    )
    text_ci = paired[
        paired["evaluation"] == "text_locked500_vs_exact_transllama_v1_k1"
    ].copy()

    DERIVED.mkdir(parents=True, exist_ok=True)
    frontier[["k", "token_f1", "sequence_similarity", "laal", "valid_completion"]].to_csv(
        DERIVED / "figure2_wait_k_frontier.csv", index=False
    )
    seeds.to_csv(DERIVED / "figure2_causalgloss_seeds.csv", index=False)
    text_ci.to_csv(DERIVED / "figure2_paired_effects.csv", index=False)

    fig = plt.figure(figsize=(7.01, 3.35))
    grid = fig.add_gridspec(2, 2, hspace=0.72, wspace=0.34)
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.91)

    ax_a.plot(
        frontier["laal"],
        frontier["token_f1"],
        color=COLORS["waitk"],
        linewidth=1.25,
        marker="o",
        markersize=4.3,
        markerfacecolor="white",
        markeredgewidth=1.0,
        label="TransLLaMA-style modified Wait-k",
        zorder=2,
    )
    label_offsets = {
        1: (7, 5),
        2: (6, 6),
        3: (6, 6),
        4: (6, 6),
        5: (6, 6),
        6: (6, -10),
        7: (-10, 5),
        8: (-5, -10),
    }
    for row in frontier.itertuples(index=False):
        dx, dy = label_offsets[row.k]
        ax_a.annotate(
            f"k={row.k}",
            (row.laal, row.token_f1),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.5,
            color=COLORS["muted"],
        )

    ax_a.scatter(
        seeds["laal_source_tokens"],
        seeds["token_f1"],
        s=26,
        marker="D",
        facecolor=COLORS["causal_light"],
        edgecolor=COLORS["causal"],
        linewidth=0.9,
        label="CausalGloss seeds",
        zorder=4,
    )
    mean_laal = float(seeds["laal_source_tokens"].mean())
    sd_laal = float(seeds["laal_source_tokens"].std(ddof=1))
    mean_f1 = float(seeds["token_f1"].mean())
    sd_f1 = float(seeds["token_f1"].std(ddof=1))
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
        elinewidth=1.1,
        capsize=2.4,
        label="CausalGloss mean (SD)",
        zorder=5,
    )
    baseline_k1 = frontier[frontier["k"] == 1].iloc[0]
    annotation_y = max(float(frontier["token_f1"].max()), float(seeds["token_f1"].max())) + 0.016
    ax_a.annotate(
        "",
        xy=(mean_laal, annotation_y),
        xytext=(float(baseline_k1["laal"]), annotation_y),
        arrowprops={
            "arrowstyle": "<|-|>",
            "color": COLORS["causal"],
            "linewidth": 0.9,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    latency_reduction = 100.0 * (float(baseline_k1["laal"]) - mean_laal) / float(
        baseline_k1["laal"]
    )
    ax_a.text(
        mean_laal + 0.03,
        annotation_y + 0.004,
        f"{latency_reduction:.1f}% lower LAAL",
        ha="left",
        va="bottom",
        fontsize=6.5,
        fontweight="bold",
        color=COLORS["causal"],
    )
    ax_a.set_xlim(1.25, 12.65)
    ax_a.set_ylim(0.46, annotation_y + 0.009)
    ax_a.set_xticks([2, 4, 6, 8, 10, 12])
    ax_a.set_yticks([0.50, 0.55, 0.60, 0.65, 0.70])
    ax_a.set_xlabel("LAAL (source tokens; lower is better)")
    ax_a.set_ylabel("Token F1 (higher is better)")
    ax_a.set_title("Quality-latency operating points", loc="left", pad=5, fontweight="bold")
    ax_a.legend(loc="lower left", bbox_to_anchor=(0.01, 0.01), handlelength=1.7)
    ax_a.tick_params(axis="both", length=2.5, width=0.6)

    draw_forest(
        ax_b,
        text_ci,
        metric="token_f1",
        title="Paired quality difference",
        xlabel="Delta Token F1 (CausalGloss - TransLLaMA k=1)",
        xlim=(-0.035, 0.027),
        reference_line=-0.03,
    )
    draw_forest(
        ax_c,
        text_ci,
        metric="laal",
        title="Paired latency difference",
        xlabel="Delta LAAL (CausalGloss - TransLLaMA k=1)",
        xlim=(-1.30, 0.10),
    )
    add_panel_label(ax_a, "a")
    add_panel_label(ax_b, "b")
    add_panel_label(ax_c, "c")

    FIGURES.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    base = FIGURES / "fig_quality_latency"
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


def fmt_ci(delta: float, low: float, high: float, digits: int = 4, suffix: str = "") -> str:
    return f"{delta:+.{digits}f} [{low:+.{digits}f}, {high:+.{digits}f}]{suffix}"


def build_tables(
    waitk: pd.DataFrame,
    seeds: pd.DataFrame,
    paired: pd.DataFrame,
    ablation: dict,
    epee: dict,
    epee_generator: dict,
    bleu: dict,
    speech: dict,
    progression: dict,
) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    frontier = waitk[(waitk["model"] == "v1") & (waitk["policy"] == "modified_wait_k")].copy()
    k1 = frontier[frontier["k"].astype(int) == 1].iloc[0]
    means = seeds[["token_f1", "sequence_similarity", "laal_source_tokens", "completion_rate"]].mean()
    sds = seeds[["token_f1", "sequence_similarity", "laal_source_tokens", "completion_rate"]].std(ddof=1)
    k1_bleu4 = bleu["systems"]["transllama_wait_k1"]["bleu_4"]
    causal_bleu4 = bleu["causalgloss_mean_and_sample_sd"]["bleu_4"]
    table1 = rf"""% Generated from frozen artifacts by scripts/build_core_results.py.
\begin{{table}}[t]
  \caption{{Locked-500 results. CausalGloss is mean $\pm$ sample SD over three
  seeds; the baseline is the highest-quality point in the $k=1,\ldots,8$ sweep.
  BLEU-4 is post-hoc.}}
  \label{{tab:main_results}}
  \centering
  \scriptsize
  \setlength{{\tabcolsep}}{{1.7pt}}
  \begin{{tabular*}}{{\columnwidth}}{{@{{\extracolsep{{\fill}}}}lcccc@{{}}}}
    \toprule
    System & F1 $\uparrow$ & Sim. $\uparrow$ & BLEU-4 $\uparrow$ & LAAL $\downarrow$ \\
    \midrule
    TransLLaMA-style $k=1$ & {k1['token_f1']:.4f} & {k1['sequence_similarity']:.4f} & {k1_bleu4:.2f} & {k1['laal']:.4f} \\
    CausalGloss mean & {means['token_f1']:.4f} & {means['sequence_similarity']:.4f} & {causal_bleu4['mean']:.2f} & {means['laal_source_tokens']:.4f} \\
    $\quad\pm$ sample SD & {sds['token_f1']:.4f} & {sds['sequence_similarity']:.4f} & {causal_bleu4['sample_standard_deviation']:.2f} & {sds['laal_source_tokens']:.4f} \\
    \bottomrule
  \end{{tabular*}}
\end{{table}}
"""
    (TABLES / "table_main_locked500.tex").write_text(table1, encoding="utf-8")

    abl_native = ablation["v2_3_vs_v2_2_native"]["metrics"]
    abl_guard = ablation["v2_3_vs_v2_2_prompt_guard"]["metrics"]
    epee_all = epee["subsets"]["all_401"]
    epee_f1 = epee_all["tracks"]["content"]["best_token_f1"]["comparisons"]["v2_3_vs_v1"]
    epee_laal = epee_all["laal"]["comparisons"]["v2_3_vs_v1"]
    speech_seed = speech["chinese_speech_e2e_locked500"]["per_seed"]["20260829"]["paired_vs_v1"]
    speech_f1 = speech_seed["pseudo_gloss_token_f1"]
    speech_laal = speech_seed["speech_to_gloss_laal_seconds"]
    speech_compute = speech_seed["speech_to_gloss_compute_adjusted_laal_seconds"]
    speech_len = speech_seed["prediction_reference_length_absolute_error"]
    progression_rows = progression["versions"]

    def abl_cell(metric: dict) -> str:
        return fmt_ci(
            metric["mean_difference"],
            metric["paired_bootstrap_95_ci"][0],
            metric["paired_bootstrap_95_ci"][1],
        )

    def compact_ci(delta: float, low: float, high: float, digits: int = 3) -> str:
        point = f"{delta:+.{digits}f}"
        interval = f"[{low:+.{digits}f},{high:+.{digits}f}]"
        return rf"\shortstack{{{point}\\{{\scriptsize {interval}}}}}"

    table2 = rf"""% Generated from frozen artifacts by scripts/build_core_results.py.
\begin{{table}}[t]
  \caption{{Method progression and recovery ablation. Panel A gives absolute
  results on identical locked-500 parents; CausalGloss reports S1 for direct
  pairing. Panel B gives recovery-minus-comparator deltas on the frozen
  confirmation set with 95\% bootstrap CIs. LAAL is in source tokens.}}
  \label{{tab:progression_ablation}}
  \centering
  \scriptsize
  \setlength{{\tabcolsep}}{{2.0pt}}
  \renewcommand{{\arraystretch}}{{1.03}}
  \begin{{tabular*}}{{\columnwidth}}{{@{{\extracolsep{{\fill}}}}lcc@{{}}}}
    \toprule
    Setting & F1 & LAAL \\
    \midrule
    \multicolumn{{3}}{{l}}{{\textit{{A. Successive system revisions (absolute)}}}} \\
    Lexical alignment & {progression_rows['v1']['means']['token_f1']:.4f} & {progression_rows['v1']['means']['laal']:.4f} \\
    Semantic alignment & {progression_rows['v2_1']['means']['token_f1']:.4f} & {progression_rows['v2_1']['means']['laal']:.4f} \\
    Multi-evidence + burst & {progression_rows['v2_2']['means']['token_f1']:.4f} & {progression_rows['v2_2']['means']['laal']:.4f} \\
    CausalGloss (S1) & {progression_rows['v2_3']['means']['token_f1']:.4f} & {progression_rows['v2_3']['means']['laal']:.4f} \\
    \midrule
    \multicolumn{{3}}{{l}}{{\textit{{B. Paired checks, $\Delta$ [95\% CI]}}}} \\
    Recovery vs. action-only & {compact_ci(abl_native['token_f1']['mean_difference'], abl_native['token_f1']['paired_bootstrap_95_ci'][0], abl_native['token_f1']['paired_bootstrap_95_ci'][1])} & {compact_ci(abl_native['laal_source_tokens']['mean_difference'], abl_native['laal_source_tokens']['paired_bootstrap_95_ci'][0], abl_native['laal_source_tokens']['paired_bootstrap_95_ci'][1])} \\
    Recovery vs. guard-only & {compact_ci(abl_guard['token_f1']['mean_difference'], abl_guard['token_f1']['paired_bootstrap_95_ci'][0], abl_guard['token_f1']['paired_bootstrap_95_ci'][1])} & {compact_ci(abl_guard['laal_source_tokens']['mean_difference'], abl_guard['laal_source_tokens']['paired_bootstrap_95_ci'][0], abl_guard['laal_source_tokens']['paired_bootstrap_95_ci'][1])} \\
    \bottomrule
  \end{{tabular*}}
\end{{table}}
"""
    (TABLES / "table_supporting_results.tex").write_text(table2, encoding="utf-8")


def main() -> None:
    waitk, seeds, paired, ablation, epee, epee_generator, bleu, speech, progression = read_inputs()
    validate_inputs(waitk, seeds, paired)
    validate_progression(progression)
    validate_additional_metrics(epee_generator, bleu)
    save_source_manifest()
    build_core_figure(waitk, seeds, paired)
    build_tables(
        waitk, seeds, paired, ablation, epee, epee_generator, bleu, speech, progression
    )
    print("Core results figure and tables generated successfully.")


if __name__ == "__main__":
    main()
