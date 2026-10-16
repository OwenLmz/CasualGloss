from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

LOCAL_PACKAGES = Path(__file__).resolve().parents[2] / ".paper_python_packages"
sys.path.insert(0, str(LOCAL_PACKAGES))

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

SKILL_SCRIPTS = Path(
    r"C:\Users\DELL\.codex\plugins\cache\nature-skills\nature-skills\0.1.0"
    r"\skills\nature-figure\scripts"
)
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


SYSTEM_FILES = {
    "waitk1": DATA_DIR / "waitk1.jsonl",
    "waitk3": DATA_DIR / "waitk3.jsonl",
    "waitk5": DATA_DIR / "waitk5.jsonl",
    "waitk7": DATA_DIR / "waitk7.jsonl",
    "waitk5_seed20260904": DATA_DIR / "waitk5_seed20260904.jsonl",
    "waitk5_seed20260905": DATA_DIR / "waitk5_seed20260905.jsonl",
    "waitk7_seed20260904": DATA_DIR / "waitk7_seed20260904.jsonl",
    "waitk7_seed20260905": DATA_DIR / "waitk7_seed20260905.jsonl",
    "causal_s1": DATA_DIR / "cg_s1.jsonl",
    "causal_s2": DATA_DIR / "cg_s2.jsonl",
    "causal_s3": DATA_DIR / "cg_s3.jsonl",
}

BASELINE_SEEDS = {
    "waitk5": ("waitk5", "waitk5_seed20260904", "waitk5_seed20260905"),
    "waitk7": ("waitk7", "waitk7_seed20260904", "waitk7_seed20260905"),
}

METRICS = {
    "precision": "pseudo_gloss_token_precision",
    "recall": "pseudo_gloss_token_recall",
    "token_f1": "pseudo_gloss_token_f1",
    "sequence_similarity": "pseudo_gloss_sequence_similarity",
    "laal": "length_adaptive_average_lagging_source_tokens",
    "predicted_length": "predicted_gloss_token_count",
    "reference_length": "target_gloss_token_count",
    "hallucinated_ratio": "hallucinated_token_ratio",
    "action_count": "action_count",
}

SEED_LABELS = {
    "causal_s1": "S1",
    "causal_s2": "S2",
    "causal_s3": "S3",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_alignment(systems: dict[str, list[dict]]) -> list[str]:
    reference_name = next(iter(systems))
    reference_rows = systems[reference_name]
    if len(reference_rows) != 500:
        raise ValueError(f"{reference_name} has {len(reference_rows)} rows, expected 500")

    ids = [row["id"] for row in reference_rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{reference_name} contains duplicate parent ids")

    reference_by_id = {row["id"]: row for row in reference_rows}
    for name, rows in systems.items():
        if len(rows) != 500:
            raise ValueError(f"{name} has {len(rows)} rows, expected 500")
        row_ids = [row["id"] for row in rows]
        if len(set(row_ids)) != len(row_ids):
            raise ValueError(f"{name} contains duplicate parent ids")
        if set(row_ids) != set(ids):
            raise ValueError(f"{name} parent ids differ from {reference_name}")
        by_id = {row["id"]: row for row in rows}
        for parent_id in ids:
            expected = reference_by_id[parent_id]
            observed = by_id[parent_id]
            if observed["source_tokens"] != expected["source_tokens"]:
                raise ValueError(f"source mismatch for {parent_id} in {name}")
            if observed["target_pseudo_asl_gloss_tokens"] != expected[
                "target_pseudo_asl_gloss_tokens"
            ]:
                raise ValueError(f"reference mismatch for {parent_id} in {name}")
    return ids


def ngrams(tokens: list[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1))


def corpus_bleu(rows: list[dict], max_order: int = 4) -> dict[str, float]:
    matches = [0] * max_order
    totals = [0] * max_order
    hyp_len = 0
    ref_len = 0
    for row in rows:
        hypothesis = row["predicted_pseudo_asl_gloss_tokens"]
        reference = row["target_pseudo_asl_gloss_tokens"]
        hyp_len += len(hypothesis)
        ref_len += len(reference)
        for order in range(1, max_order + 1):
            hyp_counts = ngrams(hypothesis, order)
            ref_counts = ngrams(reference, order)
            matches[order - 1] += sum(
                min(count, ref_counts[gram]) for gram, count in hyp_counts.items()
            )
            totals[order - 1] += sum(hyp_counts.values())

    bp = 1.0 if hyp_len >= ref_len else math.exp(1.0 - ref_len / max(hyp_len, 1))
    output: dict[str, float] = {
        "hypothesis_length": float(hyp_len),
        "reference_length": float(ref_len),
        "length_ratio": hyp_len / ref_len,
        "brevity_penalty": bp,
    }
    for order in range(1, max_order + 1):
        precisions = [
            matches[index] / totals[index] if totals[index] else 0.0
            for index in range(order)
        ]
        precision_array = np.asarray(precisions)
        if np.any(precision_array <= 0):
            score = 0.0
        else:
            score = 100.0 * bp * math.exp(sum(math.log(value) for value in precisions) / order)
        output[f"bleu_{order}"] = score
    return output


def summarize(rows: list[dict]) -> dict[str, float]:
    summary = {
        metric: float(np.mean([row[field] for row in rows]))
        for metric, field in METRICS.items()
    }
    summary["completion"] = float(np.mean([bool(row["completed_with_end"]) for row in rows]))
    summary["premature_end"] = float(np.mean([bool(row["premature_end"]) for row in rows]))
    summary["relative_length_error"] = float(
        np.mean(
            [
                abs(row["predicted_gloss_token_count"] - row["target_gloss_token_count"])
                / max(row["target_gloss_token_count"], 1)
                for row in rows
            ]
        )
    )
    summary.update(corpus_bleu(rows))
    return summary


def paired_bootstrap(
    candidate: np.ndarray,
    baseline: np.ndarray,
    rng: np.random.Generator,
    repetitions: int,
) -> dict[str, float]:
    differences = candidate - baseline
    indices = rng.integers(0, differences.size, size=(repetitions, differences.size))
    bootstrap_means = differences[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "observed_difference": float(differences.mean()),
        "bootstrap_mean_difference": float(bootstrap_means.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "repetitions": repetitions,
    }


def compute_paired_results(
    ordered: dict[str, list[dict]],
    repetitions: int,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    result: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    rng = np.random.Generator(np.random.PCG64(20260914))
    for seed_name in SEED_LABELS:
        result[seed_name] = {}
        candidate_rows = ordered[seed_name]
        for baseline_name in ("waitk1", "waitk3", "waitk5", "waitk7"):
            baseline_names = BASELINE_SEEDS.get(baseline_name, (baseline_name,))
            result[seed_name][baseline_name] = {}
            for metric_name, field in (
                ("token_f1", METRICS["token_f1"]),
                ("sequence_similarity", METRICS["sequence_similarity"]),
                ("laal", METRICS["laal"]),
            ):
                candidate = np.asarray([row[field] for row in candidate_rows], dtype=float)
                baseline = np.mean(
                    np.asarray(
                        [[row[field] for row in ordered[name]] for name in baseline_names],
                        dtype=float,
                    ),
                    axis=0,
                )
                result[seed_name][baseline_name][metric_name] = paired_bootstrap(
                    candidate, baseline, rng, repetitions
                )
    return result


def hierarchical_bootstrap(
    ordered: dict[str, list[dict]],
    baseline_names: tuple[str, ...],
    field: str,
    repetitions: int,
) -> dict[str, float]:
    candidates = np.asarray(
        [[row[field] for row in ordered[name]] for name in SEED_LABELS], dtype=float
    )
    baselines = np.asarray(
        [[row[field] for row in ordered[name]] for name in baseline_names], dtype=float
    )
    rng = np.random.Generator(np.random.PCG64(20260916))
    draws = np.empty(repetitions)
    for index in range(repetitions):
        parents = rng.integers(0, candidates.shape[1], candidates.shape[1])
        candidate_seeds = rng.integers(0, candidates.shape[0], candidates.shape[0])
        baseline_seeds = rng.integers(0, baselines.shape[0], baselines.shape[0])
        draws[index] = candidates[candidate_seeds][:, parents].mean() - baselines[
            baseline_seeds
        ][:, parents].mean()
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "observed_difference": float(candidates.mean() - baselines.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "repetitions": repetitions,
        "resampling_units": "three training seeds per system and 500 paired parents",
    }


def write_tsv(
    summaries: dict[str, dict[str, float]],
    paired: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> None:
    metric_columns = [
        "precision",
        "recall",
        "token_f1",
        "sequence_similarity",
        "bleu_4",
        "laal",
        "predicted_length",
        "relative_length_error",
        "completion",
        "premature_end",
        "hallucinated_ratio",
        "action_count",
    ]
    with (OUTPUT_DIR / "fixed_waitk_summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["system", *metric_columns])
        for name, summary in summaries.items():
            writer.writerow([name, *[summary[column] for column in metric_columns]])

    with (OUTPUT_DIR / "paired_differences.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "candidate",
                "baseline",
                "metric",
                "observed_difference",
                "ci95_lower",
                "ci95_upper",
                "repetitions",
            ]
        )
        for candidate, baseline_results in paired.items():
            for baseline, metric_results in baseline_results.items():
                for metric, values in metric_results.items():
                    writer.writerow(
                        [
                            candidate,
                            baseline,
                            metric,
                            values["observed_difference"],
                            values["ci95_lower"],
                            values["ci95_upper"],
                            values["repetitions"],
                        ]
                    )


def write_latex_table(summaries: dict[str, dict[str, float]]) -> None:
    causal_names = list(SEED_LABELS)
    columns = ("token_f1", "sequence_similarity", "bleu_4", "laal")
    causal_values = {
        column: np.asarray([summaries[name][column] for name in causal_names], dtype=float)
        for column in columns
    }
    means = {column: float(values.mean()) for column, values in causal_values.items()}
    sds = {column: float(values.std(ddof=1)) for column, values in causal_values.items()}

    lines = [
        r"\begin{table}[t]",
        r"  \caption{Core locked-500 comparison; Fig.~\ref{fig:quality_latency} shows",
        r"  the full fixed Wait-$k$ curve. Wait-$k=7$ and CausalGloss report",
        r"  three-seed means and sample SDs;",
        r"  BLEU-4 is post-hoc.}",
        r"  \label{tab:main_locked500}",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{2.4pt}",
        r"  \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrr@{}}",
        r"    \toprule",
        r"    System & F1 $\uparrow$ & Sim. $\uparrow$ & BLEU-4 $\uparrow$ & LAAL $\downarrow$ \\",
        r"    \midrule",
    ]
    summary = summaries["waitk7"]
    spread = summaries["waitk7_sample_sd"]
    lines.append(
        "    Wait-$k=7$ mean & "
        f"{summary['token_f1']:.4f} & {summary['sequence_similarity']:.4f} & "
        f"{summary['bleu_4']:.2f} & {summary['laal']:.4f} \\\\"
    )
    lines.append(
        "    $\\quad\\pm$ sample SD & "
        f"{spread['token_f1']:.4f} & {spread['sequence_similarity']:.4f} & "
        f"{spread['bleu_4']:.2f} & {spread['laal']:.4f} \\\\"
    )
    lines.extend(
        [
            r"    \midrule",
            "    CausalGloss mean & "
            f"{means['token_f1']:.4f} & {means['sequence_similarity']:.4f} & "
            f"{means['bleu_4']:.2f} & {means['laal']:.4f} \\\\ ",
            "    $\\quad\\pm$ sample SD & "
            f"{sds['token_f1']:.4f} & {sds['sequence_similarity']:.4f} & "
            f"{sds['bleu_4']:.2f} & {sds['laal']:.4f} \\\\ ",
            r"    \bottomrule",
            r"  \end{tabular*}",
            r"\end{table}",
        ]
    )
    (OUTPUT_DIR / "table_main_locked500.tex").write_text("\n".join(lines) + "\n", encoding="ascii")


def plot_figure(
    summaries: dict[str, dict[str, float]],
    paired: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.5,
            "axes.linewidth": 0.75,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )

    baseline_color = "#64748B"
    candidate_color = "#0F766E"
    grid_color = "#E2E8F0"

    fig = plt.figure(figsize=(7.10, 3.15), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.18, 1.0),
        height_ratios=(1, 1),
        left=0.075,
        right=0.985,
        bottom=0.155,
        top=0.91,
        wspace=0.36,
        hspace=0.70,
    )
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])

    ks = (1, 3, 5, 7)
    x_wait = [summaries[f"waitk{k}"]["laal"] for k in ks]
    y_wait = [summaries[f"waitk{k}"]["token_f1"] for k in ks]
    ax_a.plot(
        x_wait,
        y_wait,
        color=baseline_color,
        linewidth=1.35,
        marker="o",
        markersize=4.4,
        markerfacecolor="white",
        markeredgewidth=1.15,
        label="Fixed Wait-$k$",
        zorder=2,
    )
    for k in (5, 7):
        center = summaries[f"waitk{k}"]
        spread = summaries[f"waitk{k}_sample_sd"]
        ax_a.errorbar(
            center["laal"],
            center["token_f1"],
            xerr=spread["laal"],
            yerr=spread["token_f1"],
            fmt="none",
            ecolor=baseline_color,
            elinewidth=0.9,
            capsize=2.0,
            zorder=3,
        )
    annotation_positions = {
        1: (-1.35, 0.340),
        3: (0.48, 0.535),
        5: (2.30, 0.625),
        7: (3.85, 0.672),
    }
    for k in ks:
        x_value, y_value = annotation_positions[k]
        ax_a.text(
            x_value,
            y_value,
            f"$k={k}$",
            color="#334155",
            fontsize=6.7,
        )

    causal_x = np.asarray([summaries[name]["laal"] for name in SEED_LABELS])
    causal_y = np.asarray([summaries[name]["token_f1"] for name in SEED_LABELS])
    ax_a.scatter(
        causal_x,
        causal_y,
        marker="D",
        s=34,
        color="#CCFBF1",
        edgecolor=candidate_color,
        linewidth=1.0,
        label="CausalGloss seeds",
        zorder=4,
    )
    mean_x = float(causal_x.mean())
    mean_y = float(causal_y.mean())
    sd_x = float(causal_x.std(ddof=1))
    sd_y = float(causal_y.std(ddof=1))
    ax_a.errorbar(
        mean_x,
        mean_y,
        xerr=sd_x,
        yerr=sd_y,
        fmt="*",
        markersize=10,
        color=candidate_color,
        ecolor=candidate_color,
        elinewidth=1.1,
        capsize=2.2,
        label="CausalGloss mean (SD)",
        zorder=5,
    )
    laal_reduction = 100 * (1 - mean_x / summaries["waitk7"]["laal"])
    f1_gain = mean_y - summaries["waitk7"]["token_f1"]
    ax_a.annotate(
        f"{laal_reduction:.1f}% lower LAAL\n+{f1_gain:.3f} Token F1",
        xy=(mean_x, mean_y),
        xytext=(30, 20),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=6.8,
        fontweight="bold",
        color=candidate_color,
        arrowprops={"arrowstyle": "->", "color": candidate_color, "lw": 0.8},
    )
    ax_a.set_xlabel("LAAL (source tokens; lower is better)")
    ax_a.set_ylabel("Token F1 (higher is better)")
    ax_a.set_xlim(-1.45, 4.75)
    ax_a.set_ylim(0.285, 0.735)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=baseline_color,
            linewidth=1.35,
            marker="o",
            markersize=4.4,
            markerfacecolor="white",
            markeredgewidth=1.15,
            label="Fixed Wait-$k$",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="D",
            markersize=4.5,
            markerfacecolor="#CCFBF1",
            markeredgecolor=candidate_color,
            markeredgewidth=1.0,
            label="CausalGloss seeds",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="*",
            markersize=7.5,
            markerfacecolor=candidate_color,
            markeredgecolor=candidate_color,
            label="CausalGloss mean (SD)",
        ),
    ]
    legend = ax_a.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=6.0,
        frameon=True,
        framealpha=0.96,
        edgecolor="#CBD5E1",
        borderpad=0.48,
        labelspacing=0.58,
        handlelength=1.50,
        handleheight=1.10,
        handletextpad=0.45,
    )
    legend.get_frame().set_linewidth(0.6)

    y_positions = np.arange(3)
    labels = ["S1", "S2", "S3"]
    for axis, metric, xlabel in (
        (ax_b, "token_f1", "$\\Delta$ Token F1 (CausalGloss - Wait-$k=7$ mean)"),
        (ax_c, "laal", "$\\Delta$ LAAL (CausalGloss - Wait-$k=7$ mean)"),
    ):
        observed = np.asarray(
            [paired[name]["waitk7"][metric]["observed_difference"] for name in SEED_LABELS]
        )
        lower = np.asarray(
            [paired[name]["waitk7"][metric]["ci95_lower"] for name in SEED_LABELS]
        )
        upper = np.asarray(
            [paired[name]["waitk7"][metric]["ci95_upper"] for name in SEED_LABELS]
        )
        axis.errorbar(
            observed,
            y_positions,
            xerr=np.vstack((observed - lower, upper - observed)),
            fmt="o",
            markersize=3.6,
            color=candidate_color,
            ecolor=candidate_color,
            elinewidth=1.05,
            capsize=2.0,
        )
        axis.axvline(0.0, color="#94A3B8", linewidth=0.8, linestyle="--")
        axis.set_yticks(y_positions, labels)
        axis.invert_yaxis()
        axis.set_xlabel(xlabel, labelpad=2.0)
        axis.grid(axis="x", color=grid_color, linewidth=0.55)

    ax_b.set_xlim(-0.005, 0.09)
    ax_c.set_xlim(-3.0, 0.1)

    top_label_y = ax_a.get_position().y1 + 0.038
    lower_label_y = ax_c.get_position().y1 + 0.022
    for label, axis, y_value in (
        ("a", ax_a, top_label_y),
        ("b", ax_b, top_label_y),
        ("c", ax_c, lower_label_y),
    ):
        fig.text(
            axis.get_position().x0 - 0.025,
            y_value,
            label,
            fontsize=8.5,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    ax_a.set_title("Quality-latency operating points", loc="left", fontweight="bold", pad=8)
    ax_b.set_title("Paired quality difference", loc="left", fontweight="bold", pad=5)
    ax_c.set_title("Paired latency difference", loc="left", fontweight="bold", pad=5)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    base = FIGURE_DIR / "fig_quality_latency_fixed_waitk"
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(base) + ".alignment.json",
        overlay_svg=str(base) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".svg", bbox_inches="tight")
    fig.savefig(str(base) + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(str(base) + ".tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()

    systems = {name: load_jsonl(path) for name, path in SYSTEM_FILES.items()}
    ordered_ids = validate_alignment(systems)
    ordered = {
        name: sorted(rows, key=lambda row: ordered_ids.index(row["id"]))
        for name, rows in systems.items()
    }
    summaries = {name: summarize(rows) for name, rows in ordered.items()}
    for baseline_name, seed_names in BASELINE_SEEDS.items():
        values = {metric: np.asarray([summaries[name][metric] for name in seed_names])
                  for metric in summaries[baseline_name]}
        summaries[f"{baseline_name}_original"] = summaries[baseline_name]
        summaries[baseline_name] = {
            metric: float(data.mean()) for metric, data in values.items()
        }
        summaries[f"{baseline_name}_sample_sd"] = {
            metric: float(data.std(ddof=1)) for metric, data in values.items()
        }

    prior_bleu_path = ROOT.parent / "icassp2027_paper" / "source_data" / "raw" / "locked500_bleu_report.json"
    prior_bleu = json.loads(prior_bleu_path.read_text(encoding="utf-8"))
    for name, prior_key in (
        ("causal_s1", "causalgloss_seed20260829"),
        ("causal_s2", "causalgloss_seed20260904"),
        ("causal_s3", "causalgloss_seed20260905"),
    ):
        expected = prior_bleu["systems"][prior_key]["bleu_4"]
        if not math.isclose(summaries[name]["bleu_4"], expected, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(
                f"BLEU implementation mismatch for {name}: {summaries[name]['bleu_4']} != {expected}"
            )

    paired = compute_paired_results(ordered, args.bootstrap)
    hierarchical = {
        baseline_name: {
            metric: hierarchical_bootstrap(ordered, names, field, args.bootstrap)
            for metric, field in (
                ("token_f1", METRICS["token_f1"]),
                ("laal", METRICS["laal"]),
            )
        }
        for baseline_name, names in BASELINE_SEEDS.items()
    }
    causal_metrics = {}
    for metric in ("token_f1", "sequence_similarity", "bleu_4", "laal"):
        values = np.asarray([summaries[name][metric] for name in SEED_LABELS], dtype=float)
        causal_metrics[metric] = {
            "mean": float(values.mean()),
            "sample_standard_deviation": float(values.std(ddof=1)),
            "seed_count": 3,
        }

    output = {
        "scientific_status": "frozen_locked500_reanalysis",
        "reference_status": "LLM_generated_weak_ASL_gloss_not_expert_gold",
        "baseline_definition": (
            "Schedule-specific fixed Wait-k models, trained with the same Qwen2.5-7B "
            "backbone, Fine-tune 1 initialization, English-gloss data, effective batch "
            "size 32, and one-epoch budget."
        ),
        "locked_parent_count": 500,
        "same_parent_ids_source_and_reference": True,
        "selection_used_locked_test": False,
        "bootstrap": {
            "repetitions": args.bootstrap,
            "parent_paired_seed": 20260914,
            "seed_and_parent_hierarchical_seed": 20260916,
        },
        "systems": summaries,
        "causalgloss_mean_and_sample_sd": causal_metrics,
        "paired_candidate_minus_fixed_waitk": paired,
        "hierarchical_candidate_minus_fixed_waitk": hierarchical,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "fixed_waitk_analysis.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_tsv(summaries, paired)
    write_latex_table(summaries)
    plot_figure(summaries, paired)

    k7 = summaries["waitk7"]
    causal = causal_metrics
    laal_reduction = 100.0 * (k7["laal"] - causal["laal"]["mean"]) / k7["laal"]
    print(f"Validated {len(ordered_ids)} aligned locked parents across {len(systems)} systems.")
    print(
        "CausalGloss mean: "
        f"F1={causal['token_f1']['mean']:.6f}, "
        f"BLEU-4={causal['bleu_4']['mean']:.3f}, "
        f"LAAL={causal['laal']['mean']:.6f}."
    )
    print(
        "Fixed Wait-k=7: "
        f"F1={k7['token_f1']:.6f}, BLEU-4={k7['bleu_4']:.3f}, LAAL={k7['laal']:.6f}."
    )
    print(f"Relative LAAL reduction versus fixed Wait-k=7: {laal_reduction:.2f}%.")


if __name__ == "__main__":
    main()
