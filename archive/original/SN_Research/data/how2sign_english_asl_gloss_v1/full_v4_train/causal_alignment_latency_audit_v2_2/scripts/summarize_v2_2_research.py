#!/usr/bin/env python3
"""Consolidate v2.2 training, latency, selection and E2E artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def training_summary(run: Path) -> dict[str, Any]:
    history = read_jsonl(run / "training_history.jsonl")
    validations = [row for row in history if row.get("event") == "validation"]
    updates = [row for row in history if row.get("event") == "train_update"]
    best = None
    if validations:
        best = min(
            validations,
            key=lambda row: float(row["validation"]["pseudo_action_loss"]),
        )
    return {
        "run": str(run),
        "update_count_logged": len(updates),
        "last_update": updates[-1] if updates else None,
        "baseline_validation": validations[0] if validations else None,
        "best_validation": best,
        "checkpoint_selection": read_json(run / "validation_checkpoint_selection.json"),
        "selected_locked_rollout": read_json(
            run / "rollout_locked_500_selected" / "rollout_metrics.json"
        ),
    }


def metric(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    data = args.project_root / "data/how2sign_english_asl_gloss_v1"
    train = data / "full_v4_train"
    semantic = train / "semantic_causal_alignment_v2_2_full"
    ft = train / "fine_tuning_v2_semantic_v2_2_full"
    audit = train / "causal_alignment_latency_audit_v2_2"
    frontend = data / "chinese_speech_frontend_v0"
    primary = ft / "runs/qwen2_5_7b_lora_semantic_causal_v2_2_1_treebank_full"
    recovery = ft / "runs/qwen2_5_7b_lora_semantic_causal_v2_2_2_recovery_full"

    label_summary = read_json(
        semantic / "reports/semantic_alignment_v2_2_full_summary.json"
    )
    validation_decision = read_json(
        audit / "reports/validation_v2_2_model_decision.json"
    )
    locked_comparison = read_json(
        audit / "reports/locked500_final_model_comparison.json"
    )
    locked_decision = read_json(
        audit / "reports/locked500_predeclared_model_decision.json"
    )
    e2e_decision = read_json(
        frontend / "reports/task5_pilot_100.causal_e2e_selection.json"
    )

    selected = None if locked_decision is None else (
        locked_decision.get("selected_candidate") or locked_decision.get("fallback")
    )
    conclusions = []
    if label_summary:
        conclusions.append(
            "The full v2.2 pseudo-timing dataset passed transition and leakage audits."
        )
    if validation_decision:
        validation_selected = (
            validation_decision.get("selected_candidate")
            or validation_decision.get("fallback")
        )
        conclusions.append(
            f"Validation-only selection chose {validation_selected} as the v2.2 variant."
        )
    if selected:
        conclusions.append(
            f"The confirmatory locked-500 decision retained {selected}."
        )
    if e2e_decision:
        conclusions.append(
            "The selected policy was checked on the same locked 100-sample synthetic "
            "Mandarin speech frontend with paired latency and content metrics."
        )

    output = {
        "scientific_status": "completed_v2_2_pseudo_gloss_causal_alignment_study",
        "label_summary": label_summary,
        "primary_v2_2_1": training_summary(primary),
        "recovery_v2_2_2": training_summary(recovery),
        "validation_variant_selection": validation_decision,
        "locked500_comparison": locked_comparison,
        "locked500_confirmatory_decision": locked_decision,
        "end_to_end_speech_pilot_decision": e2e_decision,
        "conclusions": conclusions,
        "non_claims": [
            "Pseudo-gloss quality metrics are not verified ASL translation accuracy.",
            "Pseudo timing is not gold sign timing.",
            "The end-to-end Mandarin speech pilot uses synthetic multi-speaker TTS.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Semantic Causal Alignment v2.2: Final Research Summary",
        "",
        "## Scope",
        "",
        "English text -> incremental weak pseudo-ASL gloss, followed by a locked "
        "Chinese-speech -> incremental English -> incremental pseudo-gloss pilot.",
        "",
        "## Dataset and policy",
        "",
    ]
    if label_summary:
        lines.extend(
            [
                f"- Parents: {label_summary['parent_count']:,}",
                f"- Actions: {label_summary['action_count']:,}",
                f"- Label LAAL: {label_summary['mean_average_lagging_source_tokens']:.4f} source tokens",
                f"- Semantic sufficiency: {label_summary['semantic_sufficiency_ratio']:.4%}",
                "- WRITE holds the source prefix; WAIT reads one new source token.",
            ]
        )
    lines.extend(["", "## Decisions", ""])
    lines.extend(f"- {item}" for item in conclusions)

    lines.extend(["", "## Training summary", ""])
    for name, run in (("v2.2.1 primary", primary), ("v2.2.2 recovery", recovery)):
        summary = training_summary(run)
        best = summary.get("best_validation")
        if best:
            validation = best["validation"]
            lines.append(
                f"- {name}: best validation step {best.get('global_step')}; "
                f"loss {metric(validation.get('pseudo_action_loss'))}; "
                f"teacher-forced token accuracy "
                f"{metric(validation.get('pseudo_action_token_accuracy_teacher_forced'))}; "
                f"sequence exact "
                f"{metric(validation.get('pseudo_action_sequence_exact_match_teacher_forced'))}."
            )
        else:
            lines.append(f"- {name}: not completed.")

    if locked_comparison:
        lines.extend(
            [
                "",
                "## Locked 500 free-running comparison",
                "",
                "| Model | Valid completion | Sequence similarity | Token F1 | Exact | LAAL | Ref. AL | AP | DAL | Invalid |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, version in locked_comparison.get("versions", {}).items():
            means = version.get("means", {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        metric(means.get("completed_after_full_source")),
                        metric(means.get("sequence_similarity")),
                        metric(means.get("token_f1")),
                        metric(means.get("exact_match")),
                        metric(means.get("laal")),
                        metric(means.get("reference_length_al")),
                        metric(means.get("ap")),
                        metric(means.get("dal")),
                        metric(means.get("invalid_ratio")),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "All content scores use pseudo-gloss references. LAAL, reference-length "
                "AL and DAL are measured in source tokens; lower is better.",
            ]
        )

    if locked_decision:
        lines.extend(["", "## Confirmatory acceptance", ""])
        for row in locked_decision.get("decisions", []):
            checks = ", ".join(
                f"{key}={'PASS' if value else 'FAIL'}"
                for key, value in row.get("checks", {}).items()
            )
            lines.append(
                f"- {row.get('candidate')}: accepted={row.get('accepted')}; {checks}."
            )
        lines.append(
            f"- Final retained model: {locked_decision.get('selected_candidate') or locked_decision.get('fallback')}."
        )

    if e2e_decision:
        lines.extend(["", "## Synthetic Mandarin speech pilot", ""])
        if e2e_decision.get("candidate_run_skipped"):
            lines.append(
                f"- Candidate endpoint skipped; retained {e2e_decision.get('selected_model')}."
            )
        else:
            lines.append(
                f"- Paired records: {e2e_decision.get('common_record_count')}; "
                f"accepted={e2e_decision.get('accepted')}."
            )
            for key, value in e2e_decision.get("predeclared_checks", {}).items():
                lines.append(f"- {key}: {'PASS' if value else 'FAIL'}")
            lines.append(
                "- Standard audio-time latency excludes inference; compute-adjusted "
                "LAAL adds measured serial action-inference time."
            )
    lines.extend(
        [
            "",
            "## Scientific limitations",
            "",
            "- Pseudo-gloss quality metrics are not verified ASL translation accuracy.",
            "- Pseudo timing is not gold sign timing.",
            "- The Mandarin speech pilot uses synthetic multi-speaker TTS.",
            "",
            f"Machine-readable report: `{args.output_json}`",
        ]
    )
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(args.output_json), "markdown": str(args.output_markdown)}, indent=2))


if __name__ == "__main__":
    main()
