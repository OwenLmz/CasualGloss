#!/usr/bin/env python3
"""Tune a confident-prefix burst policy on the held-out pilot development set.

The policy uses lexical anchors as hard evidence for exact/stem matches and
Fine-tune-1 prefix likelihoods for weaker or interpolated tokens.  It allows
multiple target tokens to share a source position; only target order, not a
one-token-per-prefix clock, is enforced.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


NEGATION = {"NOT", "NO", "NONE", "NEVER", "NOTHING", "NOBODY", "WITHOUT", "CANNOT", "CAN'T", "DON'T"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_positions(values: list[float], full: float, margin: float, minimum_probability: float, window: int) -> list[int]:
    source_count = len(values) - 1
    threshold = max(full - margin, math.log(minimum_probability))
    positions = []
    for position in range(1, source_count + 1):
        end = position + window - 1
        if position == source_count or (
            end <= source_count and all(values[index] >= threshold for index in range(position, end + 1))
        ):
            positions.append(position)
    return positions


def al(delays: list[int], source_length: int, target_length: int) -> float:
    gamma = target_length / source_length
    tau = next((index for index, delay in enumerate(delays, 1) if delay >= source_length), len(delays))
    return mean(delays[index - 1] - (index - 1) / gamma for index in range(1, tau + 1))


def choose_positions(pair: dict[str, Any], score: dict[str, Any], config: dict[str, Any]) -> tuple[list[int], list[dict[str, Any]]]:
    matrix = score["gloss_token_mean_logprob_by_source_position"]
    source_count = len(pair["source_tokens"])
    chosen = []
    diagnostics = []
    previous = 1
    for index, (token, meta) in enumerate(zip(pair["target_pseudo_asl_gloss_tokens"], pair["semantic_gloss_token_alignment"])):
        values = [float(row[index]) for row in matrix]
        full = values[-1]
        anchor = max(1, int(meta["source_position"]))
        stable = [
            value
            for value in stable_positions(
                values,
                full,
                float(config["semantic_margin"]),
                float(config["minimum_probability"]),
                int(config["stability_window"]),
            )
            if value >= anchor
        ]
        earliest_stable = stable[0] if stable else source_count
        anchor_gap = max(0.0, full - values[anchor])
        anchor_probability = math.exp(values[anchor])
        method = str(meta.get("match_method", ""))
        strong_lexical = bool(meta.get("anchor_is_lexical")) and (
            int(meta.get("match_strength", 0)) >= 2 or "exact" in method or "stem" in method
        )

        if token.upper() in NEGATION and strong_lexical:
            position = anchor
            reason = "negation_lexical_anchor"
        elif strong_lexical and config["lexical_mode"] == "always_anchor":
            position = anchor
            reason = "strong_lexical_anchor"
        elif strong_lexical and anchor_gap <= float(config["lexical_gap_margin"]) and anchor_probability >= float(config["minimum_probability"]):
            position = anchor
            reason = "lexical_anchor_semantically_supported"
        else:
            rescue_cap = min(source_count, anchor + int(config["maximum_rescue_delay"]))
            rescue_gap = max(0.0, full - values[rescue_cap])
            if earliest_stable <= rescue_cap:
                position = earliest_stable
                reason = "stable_prefix"
            elif rescue_gap <= float(config["rescue_gap_margin"]):
                position = rescue_cap
                reason = "bounded_semantic_rescue"
            else:
                position = earliest_stable
                reason = "late_stable_prefix"

        position = max(previous, min(source_count, position))
        previous = position
        diagnostics.append(
            {
                "token": token,
                "position": position,
                "anchor": anchor,
                "reason": reason,
                "gap": max(0.0, full - values[position]),
                "probability": math.exp(values[position]),
                "strong_lexical": strong_lexical,
            }
        )
        chosen.append(position)
    return chosen, diagnostics


def evaluate_config(pairs: list[dict[str, Any]], scores: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    nals = []
    laals = []
    gaps = []
    sufficient = 0
    total = 0
    full_source = 0
    burst_tokens = 0
    late_negation = 0
    reason_counts: dict[str, int] = {}
    for pair in pairs:
        positions, diagnostics = choose_positions(pair, scores[pair["id"]], config)
        source_count = len(pair["source_tokens"])
        target_count = len(pair["target_pseudo_asl_gloss_tokens"])
        value = al(positions, source_count, target_count)
        nals.append(value / source_count)
        laals.append(value)
        burst_tokens += sum(positions[index] == positions[index - 1] for index in range(1, len(positions)))
        full_source += sum(position >= source_count for position in positions)
        for item in diagnostics:
            total += 1
            gaps.append(item["gap"])
            sufficient += item["gap"] <= 0.75 and item["probability"] >= 0.01
            reason_counts[item["reason"]] = reason_counts.get(item["reason"], 0) + 1
            if item["token"].upper() in NEGATION and item["position"] > item["anchor"] + 1:
                late_negation += 1
    return {
        **config,
        "parent_count": len(pairs),
        "token_count": total,
        "mean_normalized_al": mean(nals),
        "mean_laal": mean(laals),
        "mean_semantic_gap_nats": mean(gaps),
        "semantic_sufficiency_ratio": sufficient / max(1, total),
        "mean_full_source_tokens_per_parent": full_source / max(1, len(pairs)),
        "mean_burst_tokens_per_parent": burst_tokens / max(1, len(pairs)),
        "late_negation_violation_count": late_negation,
        "reason_counts": reason_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--output-grid", type=Path, required=True)
    parser.add_argument("--output-selection", type=Path, required=True)
    args = parser.parse_args()

    pairs = [row for row in read_jsonl(args.pairs) if row.get("eligible_for_finetune2_v2_1", True)]
    scores = {row["id"]: row for path in args.scores for row in read_jsonl(path)}
    results = []
    for lexical_mode, semantic_margin, lexical_gap_margin, minimum_probability, window, delay, rescue_gap in itertools.product(
        ["always_anchor", "gap_guarded"],
        [0.5, 0.75, 1.0],
        [0.25, 0.5, 0.75, 1.0],
        [0.005, 0.01],
        [1, 2],
        [1, 2, 3],
        [0.75, 1.0, 1.5],
    ):
        config = {
            "lexical_mode": lexical_mode,
            "semantic_margin": semantic_margin,
            "lexical_gap_margin": lexical_gap_margin,
            "minimum_probability": minimum_probability,
            "stability_window": window,
            "maximum_rescue_delay": delay,
            "rescue_gap_margin": rescue_gap,
        }
        results.append(evaluate_config(pairs, scores, config))

    # Selection is development-only. First retain policies with strong semantic
    # coverage, then treat latency differences below 2e-5 NAL as practically tied.
    # Within that narrow latency band, prefer the policy with better semantic
    # coverage instead of overfitting to an imperceptible latency difference.
    eligible = [
        row
        for row in results
        if row["semantic_sufficiency_ratio"] >= 0.995
        and row["mean_semantic_gap_nats"] <= 0.10
        and row["late_negation_violation_count"] <= 1
    ]
    selection_pool = eligible or results
    best_latency = min(row["mean_normalized_al"] for row in selection_pool)
    near_fastest = [
        row for row in selection_pool if row["mean_normalized_al"] <= best_latency + 0.00002
    ]
    selected = min(
        near_fastest,
        key=lambda row: (
            -row["semantic_sufficiency_ratio"],
            row["mean_semantic_gap_nats"],
            row["lexical_gap_margin"],
        ),
    )
    pareto = []
    for row in results:
        dominated = any(
            other["mean_normalized_al"] <= row["mean_normalized_al"]
            and other["mean_semantic_gap_nats"] <= row["mean_semantic_gap_nats"]
            and other["semantic_sufficiency_ratio"] >= row["semantic_sufficiency_ratio"]
            and (
                other["mean_normalized_al"] < row["mean_normalized_al"]
                or other["mean_semantic_gap_nats"] < row["mean_semantic_gap_nats"]
                or other["semantic_sufficiency_ratio"] > row["semantic_sufficiency_ratio"]
            )
            for other in results
        )
        if not dominated:
            pareto.append(row)

    args.output_grid.parent.mkdir(parents=True, exist_ok=True)
    flat = [{key: (json.dumps(value, sort_keys=True) if isinstance(value, dict) else value) for key, value in row.items()} for row in results]
    with args.output_grid.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    output = {
        "scientific_status": "development_set_policy_selection_over_pseudo_gloss_not_gold_asl",
        "grid_size": len(results),
        "eligible_grid_size": len(eligible),
        "near_fastest_grid_size": len(near_fastest),
        "selected": selected,
        "pareto_front_size": len(pareto),
        "pareto_front": sorted(pareto, key=lambda row: row["mean_normalized_al"]),
        "selection_constraints": {
            "minimum_semantic_sufficiency_ratio": 0.995,
            "maximum_mean_semantic_gap_nats": 0.10,
            "maximum_late_negation_violations": 1,
            "near_fastest_nal_tolerance": 0.00002,
        },
    }
    args.output_selection.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
