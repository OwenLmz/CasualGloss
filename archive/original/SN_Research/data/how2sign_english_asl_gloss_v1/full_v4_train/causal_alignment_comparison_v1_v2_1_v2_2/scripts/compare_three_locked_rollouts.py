#!/usr/bin/env python3
"""Unified paired evaluation for v1, v2.1, and v2.2 locked-500 rollouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


AUDIT_SCRIPTS = Path(__file__).resolve().parents[2] / "causal_alignment_latency_audit_v2_2" / "scripts"
sys.path.insert(0, str(AUDIT_SCRIPTS))

from audit_latency import rollout_rows  # noqa: E402


METRICS = {
    "sequence_similarity": {"label": "Sequence Similarity", "direction": "higher", "unit": "ratio"},
    "token_f1": {"label": "Token F1", "direction": "higher", "unit": "ratio"},
    "reference_length_al": {"label": "Reference-length AL", "direction": "lower", "unit": "source_tokens"},
    "laal": {"label": "LAAL", "direction": "lower", "unit": "source_tokens"},
    "completed_after_full_source": {"label": "Valid Completion Rate", "direction": "higher", "unit": "ratio"},
    "invalid_ratio": {"label": "Invalid Output Ratio", "direction": "lower", "unit": "ratio"},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap(deltas: list[float], repetitions: int, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    count = len(deltas)
    estimates = [mean(deltas[rng.randrange(count)] for _ in range(count)) for _ in range(repetitions)]
    return {
        "mean_delta": round(mean(deltas), 8),
        "ci95_lower": round(percentile(estimates, 0.025), 8),
        "ci95_upper": round(percentile(estimates, 0.975), 8),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_version(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=PATH for --version")
    name, path = value.split("=", 1)
    return name, Path(path)


def as_number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def validate_inputs(raw_versions: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    if set(raw_versions) != {"v1", "v2_1", "v2_2"}:
        raise ValueError(f"Expected v1, v2_1, and v2_2, found {sorted(raw_versions)}")
    for name, rows in raw_versions.items():
        if len(rows) != 500:
            raise ValueError(f"{name} must contain exactly 500 unique parents, found {len(rows)}")
    id_sets = [set(rows) for rows in raw_versions.values()]
    if not all(ids == id_sets[0] for ids in id_sets[1:]):
        raise ValueError("The three rollout files do not contain identical parent IDs")

    ids = sorted(id_sets[0])
    fields = ["source_text_en", "source_tokens", "target_pseudo_asl_gloss", "target_pseudo_asl_gloss_tokens"]
    for parent_id in ids:
        reference = raw_versions["v1"][parent_id]
        for name in ("v2_1", "v2_2"):
            candidate = raw_versions[name][parent_id]
            for field in fields:
                if reference.get(field) != candidate.get(field):
                    raise ValueError(f"{field} mismatch for {parent_id}: v1 versus {name}")
    return ids


def max_write_burst(row: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    best: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_prefix: int | None = None
    for event in row.get("rollout_events", []):
        if event.get("predicted_action_class") == "WRITE":
            prefix = int(event["source_prefix_length"])
            if current and prefix == current_prefix:
                current.append(event)
            else:
                current = [event]
                current_prefix = prefix
            if len(current) > len(best):
                best = list(current)
        else:
            current = []
            current_prefix = None
    return len(best), best


def missing_target_tokens(row: dict[str, Any]) -> list[str]:
    remaining = Counter(row["predicted_pseudo_asl_gloss_tokens"])
    missing = []
    for token in row["target_pseudo_asl_gloss_tokens"]:
        if remaining[token] > 0:
            remaining[token] -= 1
        else:
            missing.append(token)
    return missing


def compact_event_trace(row: dict[str, Any], limit: int = 40) -> str:
    items = []
    for event in row.get("rollout_events", [])[:limit]:
        action_class = event.get("predicted_action_class")
        action = event.get("predicted_action")
        prefix = event.get("source_prefix_length")
        items.append(f"p{prefix}:{action_class}" + (f"({action})" if action_class == "WRITE" else ""))
    if len(row.get("rollout_events", [])) > limit:
        items.append("...")
    return " ".join(items)


def case_payload(
    category: str,
    parent_id: str,
    raw_versions: dict[str, dict[str, dict[str, Any]]],
    normalized: dict[str, dict[str, dict[str, Any]]],
    note: str,
) -> dict[str, Any]:
    v1 = raw_versions["v1"][parent_id]
    v22 = raw_versions["v2_2"][parent_id]
    burst_size, burst_events = max_write_burst(v22)
    result: dict[str, Any] = {
        "category": category,
        "id": parent_id,
        "selection_note": note,
        "source_text_en": v1["source_text_en"],
        "reference_pseudo_gloss": v1["target_pseudo_asl_gloss"],
        "reference_is_gold": False,
        "missing_v2_2_reference_tokens": missing_target_tokens(v22),
        "v2_2_max_same_prefix_write_burst": burst_size,
        "v2_2_burst_events": burst_events,
        "v2_2_compact_event_trace": compact_event_trace(v22),
        "versions": {},
    }
    for name in ("v1", "v2_1", "v2_2"):
        raw = raw_versions[name][parent_id]
        norm = normalized[name][parent_id]
        result["versions"][name] = {
            "prediction": raw["predicted_pseudo_asl_gloss"],
            "prediction_tokens": raw["predicted_pseudo_asl_gloss_tokens"],
            "write_source_positions": raw.get("write_source_positions", []),
            "sequence_similarity": norm["sequence_similarity"],
            "token_f1": norm["token_f1"],
            "reference_length_al": norm["reference_length_al"],
            "laal": norm["laal"],
            "completed_after_full_source": norm["completed_after_full_source"],
            "invalid_ratio": norm["invalid_ratio"],
        }
    return result


def select_cases(
    ids: list[str],
    raw_versions: dict[str, dict[str, dict[str, Any]]],
    normalized: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used: set[str] = set()

    exact_candidates = []
    for parent_id in ids:
        v1 = normalized["v1"][parent_id]
        v22 = normalized["v2_2"][parent_id]
        if (
            v22["exact_match"]
            and v22["completed_after_full_source"]
            and v22["invalid_ratio"] == 0
            and v22["reference_length_al"] is not None
            and v1["reference_length_al"] is not None
            and v22["laal"] is not None
            and v1["laal"] is not None
            and v22["reference_length_al"] < v1["reference_length_al"]
            and v22["laal"] < v1["laal"]
        ):
            exact_candidates.append((v22["laal"] - v1["laal"], parent_id))
    for _, parent_id in sorted(exact_candidates)[:3]:
        selected.append(
            case_payload(
                "latency_improved_exact_pseudo_reference_match",
                parent_id,
                raw_versions,
                normalized,
                "v2.2 exactly matches the fixed pseudo reference while both AL and LAAL are lower than v1.",
            )
        )
        used.add(parent_id)

    omission_candidates = []
    for parent_id in ids:
        if parent_id in used:
            continue
        raw1 = raw_versions["v1"][parent_id]
        raw22 = raw_versions["v2_2"][parent_id]
        v1 = normalized["v1"][parent_id]
        v22 = normalized["v2_2"][parent_id]
        missing = missing_target_tokens(raw22)
        v22_positions = raw22.get("write_source_positions", [])
        v1_positions = raw1.get("write_source_positions", [])
        first_shift = (
            (v22_positions[0] if v22_positions else 10**9)
            - (v1_positions[0] if v1_positions else 10**9)
        )
        if (
            len(missing) >= 2
            and len(raw22["predicted_pseudo_asl_gloss_tokens"]) < len(raw22["target_pseudo_asl_gloss_tokens"])
            and v22["sequence_similarity"] + 0.08 < v1["sequence_similarity"]
            and v22["reference_length_al"] is not None
            and v1["reference_length_al"] is not None
            and v22["reference_length_al"] < v1["reference_length_al"]
            and first_shift <= 0
        ):
            omission_candidates.append((-len(missing), v22["sequence_similarity"] - v1["sequence_similarity"], parent_id))
    for _, _, parent_id in sorted(omission_candidates)[:3]:
        selected.append(
            case_payload(
                "early_commitment_associated_omission_risk",
                parent_id,
                raw_versions,
                normalized,
                "Lower latency and earlier-or-equal first commitment coincide with missing reference tokens. This is diagnostic association, not causal proof.",
            )
        )
        used.add(parent_id)

    burst_candidates = []
    for parent_id in ids:
        if parent_id in used:
            continue
        raw22 = raw_versions["v2_2"][parent_id]
        v22 = normalized["v2_2"][parent_id]
        burst_size, burst_events = max_write_burst(raw22)
        burst_prefix = int(burst_events[0]["source_prefix_length"]) if burst_events else 10**9
        if (
            burst_size >= 2
            and burst_prefix < int(raw22["source_token_count"])
            and v22["completed_after_full_source"]
            and v22["invalid_ratio"] == 0
        ):
            burst_candidates.append((-burst_size, -v22["sequence_similarity"], parent_id))
    for _, _, parent_id in sorted(burst_candidates)[:3]:
        selected.append(
            case_payload(
                "burst_write_at_fixed_source_prefix",
                parent_id,
                raw_versions,
                normalized,
                "Before the source is complete, v2.2 commits multiple WRITE tokens at one observed prefix, demonstrating the corrected WRITE-holds-source state machine.",
            )
        )
        used.add(parent_id)

    counts = Counter(row["category"] for row in selected)
    expected = {
        "latency_improved_exact_pseudo_reference_match": 3,
        "early_commitment_associated_omission_risk": 3,
        "burst_write_at_fixed_source_prefix": 3,
    }
    if any(counts[name] != count for name, count in expected.items()):
        raise ValueError(f"Could not select all requested case types: {dict(counts)}")
    return selected


def fmt(value: float) -> str:
    return f"{value:.4f}"


def write_case_markdown(path: Path, cases: list[dict[str, Any]]) -> None:
    lines = [
        "# Locked-500 Case Studies",
        "",
        "> The reference is a fixed Qwen-generated pseudo ASL gloss, not expert gold ASL.",
        "> Omission cases show an association between earlier commitment and missing tokens; they do not prove causality.",
        "",
    ]
    headings = {
        "latency_improved_exact_pseudo_reference_match": "Latency improved with exact pseudo-reference match",
        "early_commitment_associated_omission_risk": "Early-commitment-associated omission risk",
        "burst_write_at_fixed_source_prefix": "Burst WRITE at a fixed source prefix",
    }
    for category in headings:
        lines.extend([f"## {headings[category]}", ""])
        for index, case in enumerate([row for row in cases if row["category"] == category], 1):
            lines.extend(
                [
                    f"### {index}. `{case['id']}`",
                    "",
                    f"- English: {case['source_text_en']}",
                    f"- Pseudo reference: `{case['reference_pseudo_gloss']}`",
                    f"- v1: `{case['versions']['v1']['prediction']}`",
                    f"- v2.1: `{case['versions']['v2_1']['prediction']}`",
                    f"- v2.2: `{case['versions']['v2_2']['prediction']}`",
                    f"- Missing v2.2 reference tokens: `{case['missing_v2_2_reference_tokens']}`",
                    f"- v1 similarity/F1/AL/LAAL: {fmt(case['versions']['v1']['sequence_similarity'])} / {fmt(case['versions']['v1']['token_f1'])} / {fmt(case['versions']['v1']['reference_length_al'])} / {fmt(case['versions']['v1']['laal'])}",
                    f"- v2.2 similarity/F1/AL/LAAL: {fmt(case['versions']['v2_2']['sequence_similarity'])} / {fmt(case['versions']['v2_2']['token_f1'])} / {fmt(case['versions']['v2_2']['reference_length_al'])} / {fmt(case['versions']['v2_2']['laal'])}",
                    f"- v2.2 WRITE positions: `{case['versions']['v2_2']['write_source_positions']}`",
                    f"- Maximum same-prefix WRITE burst: {case['v2_2_max_same_prefix_write_burst']}",
                    f"- Selection note: {case['selection_note']}",
                    "",
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_interpretation(path: Path, report: dict[str, Any]) -> None:
    versions = report["versions"]
    comparison = report["paired_comparisons"]["v2_2_vs_v1"]
    lines = [
        "# Unified Locked-500 Interpretation",
        "",
        "## Scope",
        "",
        "All three systems are evaluated on the same 500 parent IDs, identical English source text, identical source tokens, and identical fixed pseudo-gloss references.",
        "The reference is not expert-verified gold ASL, so these results measure pseudo-label imitation and streaming behavior.",
        "",
        "## Mean results",
        "",
        "| Metric | v1 | v2.1 | v2.2 |",
        "|---|---:|---:|---:|",
    ]
    for metric, metadata in METRICS.items():
        lines.append(
            f"| {metadata['label']} | {fmt(versions['v1']['means'][metric])} | {fmt(versions['v2_1']['means'][metric])} | {fmt(versions['v2_2']['means'][metric])} |"
        )
    lines.extend(["", "## Paired v2.2 versus v1", "", "| Metric | Delta | 95% CI | Conclusion |", "|---|---:|---:|---|"])
    for metric, metadata in METRICS.items():
        values = comparison[metric]
        lines.append(
            f"| {metadata['label']} | {fmt(values['mean_delta'])} | [{fmt(values['ci95_lower'])}, {fmt(values['ci95_upper'])}] | {values['conclusion']} |"
        )
    lines.extend(
        [
            "",
            "## Main conclusion",
            "",
            "v2.2 produces statistically shorter LAAL than v1; its mean reference-length AL is lower, but that AL difference is not statistically clear.",
            "Sequence Similarity and Token F1 are both statistically lower for v2.2.",
            "The result is therefore a latency-quality trade-off rather than an unqualified system improvement.",
            "Valid completion and invalid-output behavior should be interpreted from their paired confidence intervals rather than raw means alone.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="append", type=parse_version, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()

    version_paths = dict(args.version)
    raw_versions: dict[str, dict[str, dict[str, Any]]] = {}
    normalized: dict[str, dict[str, dict[str, Any]]] = {}
    for name, path in version_paths.items():
        raw_rows = read_jsonl(path)
        if len({row["id"] for row in raw_rows}) != len(raw_rows):
            raise ValueError(f"Duplicate parent IDs in {name}")
        raw_versions[name] = {row["id"]: row for row in raw_rows}
        normalized_rows, _ = rollout_rows(path)
        normalized[name] = {row["id"]: row for row in normalized_rows}

    ids = validate_inputs(raw_versions)
    report: dict[str, Any] = {
        "scientific_status": "paired_locked500_pseudo_gloss_evaluation_not_gold_asl",
        "same_parent_ids": True,
        "same_source_text": True,
        "same_source_tokens": True,
        "same_reference_pseudo_gloss": True,
        "common_parent_count": len(ids),
        "bootstrap": {"method": "paired_parent_level_percentile_bootstrap", "repetitions": args.bootstrap_repetitions, "seed": args.seed},
        "metric_definitions": METRICS,
        "input_files": {name: {"path": str(path), "sha256": sha256(path)} for name, path in version_paths.items()},
        "versions": {},
        "paired_comparisons": {},
        "limitations": [
            "The reference gloss is Qwen-generated pseudo ASL, not expert gold ASL.",
            "Causal timing is pseudo-aligned rather than video-verified gold timing.",
            "The rollout begins from English token prefixes rather than streaming speech chunks.",
            "Paired bootstrap quantifies sampling uncertainty on these 500 parents; it does not remove pseudo-label bias.",
        ],
    }

    for name in ("v1", "v2_1", "v2_2"):
        rows = normalized[name]
        metric_values = {
            metric: [
                value
                for parent_id in ids
                if (value := as_number(rows[parent_id][metric])) is not None
            ]
            for metric in METRICS
        }
        report["versions"][name] = {
            "parent_count": len(rows),
            "means": {
                metric: round(mean(values), 8)
                for metric, values in metric_values.items()
            },
            "valid_counts": {
                metric: len(values)
                for metric, values in metric_values.items()
            },
            "missing_counts": {
                metric: len(ids) - len(metric_values[metric])
                for metric in METRICS
            },
            "termination_reason_counts": dict(Counter(rows[parent_id]["termination_reason"] for parent_id in ids)),
        }

    delta_rows = []
    for parent_id in ids:
        row: dict[str, Any] = {"id": parent_id}
        for name in ("v1", "v2_1", "v2_2"):
            for metric in METRICS:
                row[f"{name}_{metric}"] = normalized[name][parent_id][metric]
        for metric in METRICS:
            right = as_number(normalized["v2_2"][parent_id][metric])
            left = as_number(normalized["v1"][parent_id][metric])
            row[f"v2_2_minus_v1_{metric}"] = None if right is None or left is None else right - left
        delta_rows.append(row)

    for comparison_index, name in enumerate(("v2_1", "v2_2"), 1):
        metric_results = {}
        for metric_index, (metric, metadata) in enumerate(METRICS.items()):
            pairs = [
                (left, right)
                for parent_id in ids
                if (left := as_number(normalized["v1"][parent_id][metric])) is not None
                and (right := as_number(normalized[name][parent_id][metric])) is not None
            ]
            deltas = [right - left for left, right in pairs]
            result = paired_bootstrap(deltas, args.bootstrap_repetitions, args.seed + comparison_index * 100 + metric_index)
            if metadata["direction"] == "higher":
                if result["ci95_lower"] > 0:
                    conclusion = "statistically_clear_improvement"
                elif result["ci95_upper"] < 0:
                    conclusion = "statistically_clear_regression"
                else:
                    conclusion = "no_statistically_clear_difference"
            else:
                if result["ci95_upper"] < 0:
                    conclusion = "statistically_clear_improvement"
                elif result["ci95_lower"] > 0:
                    conclusion = "statistically_clear_regression"
                else:
                    conclusion = "no_statistically_clear_difference"
            result.update({
                "difference": f"{name}_minus_v1",
                "direction": metadata["direction"],
                "paired_count": len(deltas),
                "excluded_missing_count": len(ids) - len(deltas),
                "conclusion": conclusion,
            })
            metric_results[metric] = result
        report["paired_comparisons"][f"{name}_vs_v1"] = metric_results

    output_dir = args.output_dir
    write_json(output_dir / "reports/unified_locked500_comparison.json", report)
    output_dir.joinpath("reports").mkdir(parents=True, exist_ok=True)
    with (output_dir / "reports/unified_locked500_summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["metric", "direction", "v1_mean", "v1_valid_count", "v2_1_mean", "v2_1_valid_count", "v2_2_mean", "v2_2_valid_count", "paired_count", "excluded_missing_count", "v2_2_minus_v1", "ci95_lower", "ci95_upper", "conclusion"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        comparison = report["paired_comparisons"]["v2_2_vs_v1"]
        for metric, metadata in METRICS.items():
            values = comparison[metric]
            writer.writerow({
                "metric": metric,
                "direction": metadata["direction"],
                "v1_mean": report["versions"]["v1"]["means"][metric],
                "v1_valid_count": report["versions"]["v1"]["valid_counts"][metric],
                "v2_1_mean": report["versions"]["v2_1"]["means"][metric],
                "v2_1_valid_count": report["versions"]["v2_1"]["valid_counts"][metric],
                "v2_2_mean": report["versions"]["v2_2"]["means"][metric],
                "v2_2_valid_count": report["versions"]["v2_2"]["valid_counts"][metric],
                "paired_count": values["paired_count"],
                "excluded_missing_count": values["excluded_missing_count"],
                "v2_2_minus_v1": values["mean_delta"],
                "ci95_lower": values["ci95_lower"],
                "ci95_upper": values["ci95_upper"],
                "conclusion": values["conclusion"],
            })

    with (output_dir / "reports/v2_2_vs_v1_parent_deltas.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(delta_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(delta_rows)

    cases = select_cases(ids, raw_versions, normalized)
    write_jsonl(output_dir / "cases/selected_case_studies.jsonl", cases)
    write_case_markdown(output_dir / "cases/selected_case_studies.md", cases)
    write_interpretation(output_dir / "RESULTS_INTERPRETATION.md", report)
    write_json(
        output_dir / "reports/reproducibility_manifest.json",
        {
            "script_sha256": sha256(Path(__file__)),
            "input_files": report["input_files"],
            "bootstrap": report["bootstrap"],
            "common_parent_count": len(ids),
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
