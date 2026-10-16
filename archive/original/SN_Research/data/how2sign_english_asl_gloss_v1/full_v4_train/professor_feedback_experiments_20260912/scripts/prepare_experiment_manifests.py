#!/usr/bin/env python3
"""Prepare frozen learned-baseline and one-factor ablation manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from nltk.tokenize import TreebankWordDetokenizer


DETOKENIZER = TreebankWordDetokenizer()
VARIANTS = ("no_semantic", "no_rescue", "no_negation", "no_occurrence", "no_burst")


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def state_prompt(source: list[str], prefix: int, committed: list[str]) -> str:
    return (
        "Observed English prefix:\n"
        + DETOKENIZER.detokenize(source[:prefix])
        + "\n\nAlready committed pseudo ASL gloss:\n"
        + (" ".join(committed) if committed else "<EMPTY>")
        + "\n\nSource status: "
        + ("COMPLETE" if prefix >= len(source) else "INCOMPLETE")
        + "\nReturn exactly one next action."
    )


def monotonic(values: list[int], source_count: int) -> list[int]:
    result: list[int] = []
    previous = 1
    for value in values:
        previous = max(previous, min(source_count, max(1, int(value))))
        result.append(previous)
    return result


def variant_positions(
    pair: dict[str, Any], base_pair: dict[str, Any], variant: str
) -> tuple[list[int], int]:
    source_count = len(pair["source_tokens"])
    meta = pair["semantic_v2_2_gloss_token_alignment"]
    full = [int(item["v2_2_emit_source_position"]) for item in meta]
    changed_components = 0

    if variant == "no_semantic":
        raw = [
            int(item["earliest_ordered_emit_source_position"])
            for item in base_pair["gloss_token_alignment"]
        ]
        changed_components = sum(a != b for a, b in zip(raw, full))
    elif variant == "no_rescue":
        raw = []
        for item in meta:
            value = int(item["v2_2_proposed_source_position"])
            if item.get("v2_2_decision_reason") == "bounded_semantic_rescue":
                value = int(item["v2_2_earliest_stable_source_position"])
                changed_components += 1
            raw.append(value)
    elif variant == "no_negation":
        raw = []
        for item in meta:
            value = int(item["v2_2_proposed_source_position"])
            if item.get("v2_2_decision_reason") == "negation_lexical_anchor":
                value = int(item["v2_2_earliest_stable_source_position"])
                changed_components += 1
            raw.append(value)
    elif variant == "no_occurrence":
        raw = []
        for item in meta:
            value = int(item["v2_2_proposed_source_position"])
            candidates = [int(v) for v in item.get("candidate_source_positions", [])]
            if "occurrence_aware" in str(item.get("match_method", "")) and len(candidates) > 1:
                value = min(candidates)
                changed_components += 1
            raw.append(value)
    elif variant == "no_burst":
        raw = []
        previous = 0
        for value in full:
            adjusted = int(value)
            if previous < source_count and adjusted <= previous:
                adjusted = min(source_count, previous + 1)
                changed_components += 1
            raw.append(adjusted)
            previous = adjusted
    else:
        raise ValueError(variant)
    positions = monotonic(raw, source_count)
    return positions, changed_components


def action_rows(pair: dict[str, Any], positions: list[int], variant: str) -> list[dict[str, Any]]:
    source = pair["source_tokens"]
    target = pair["target_pseudo_asl_gloss_tokens"]
    source_count = len(source)
    output: list[dict[str, Any]] = []
    committed: list[str] = []
    target_index = 0

    def add(prefix: int, action: str, kind: str, gloss_index: int | None) -> None:
        output.append(
            {
                "id": f"{pair['id']}__{variant}_action_{len(output):03d}",
                "parent_id": pair["id"],
                "split": pair.get("split", "train"),
                "state_prompt": state_prompt(source, prefix, committed),
                "target_action": action,
                "target_action_type": kind,
                "action_index": len(output),
                "source_prefix_length": prefix,
                "source_token_count": source_count,
                "source_complete": prefix >= source_count,
                "partial_source_text_en": DETOKENIZER.detokenize(source[:prefix]),
                "committed_pseudo_asl_gloss": " ".join(committed),
                "parent_target_pseudo_asl_gloss": pair["target_pseudo_asl_gloss"],
                "target_is_gold": False,
                "alignment_is_gold": False,
                "alignment_version": f"causalgloss_component_ablation_{variant}_v1",
                "transition_semantics": "WRITE_keeps_prefix_WAIT_advances_prefix",
                "target_gloss_index": gloss_index,
            }
        )

    for prefix in range(1, source_count + 1):
        while target_index < len(target) and positions[target_index] <= prefix:
            add(prefix, target[target_index], "WRITE_GLOSS", target_index)
            committed.append(target[target_index])
            target_index += 1
        if prefix < source_count:
            add(prefix, "<WAIT>", "READ_WAIT", None)
    while target_index < len(target):
        add(source_count, target[target_index], "WRITE_AFTER_SOURCE_END", target_index)
        committed.append(target[target_index])
        target_index += 1
    add(source_count, "<END>", "END", None)
    return output


def balanced(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    waits = [index for index, row in enumerate(actions) if row["target_action"] == "<WAIT>"]
    write_count = sum(row["target_action"] not in {"<WAIT>", "<END>"} for row in actions)
    keep_count = min(len(waits), max(2, math.ceil(max(1, write_count) * 1.1)))
    if keep_count >= len(waits):
        keep_waits = set(waits)
    elif keep_count <= 1:
        keep_waits = {waits[len(waits) // 2]}
    else:
        keep_waits = {
            waits[min(len(waits) - 1, round(i * (len(waits) - 1) / (keep_count - 1)))]
            for i in range(keep_count)
        }
    return [row for index, row in enumerate(actions) if row["target_action"] != "<WAIT>" or index in keep_waits]


def recovery_rows(pair: dict[str, Any], positions: list[int], variant: str) -> list[dict[str, Any]]:
    source = pair["source_tokens"]
    target = pair["target_pseudo_asl_gloss_tokens"]
    if not source or not target:
        return []
    states = [
        (len(source), 0, "full_source_empty_history"),
        (len(source), len(target) // 2, "full_source_mid_history"),
        (len(source), len(target) - 1, "full_source_tail_history"),
    ]
    if len(target) >= 2:
        due_index = max(1, round((len(target) - 1) * 0.67))
        prefix = min(len(source), positions[due_index])
        committed_count = max(0, due_index - 1)
        if positions[committed_count] <= prefix:
            states.append((prefix, committed_count, "prefix_burst_catchup"))
    unique: dict[tuple[int, int], tuple[int, int, str]] = {}
    for prefix, committed_count, name in states:
        if 0 <= committed_count < len(target):
            unique.setdefault((prefix, committed_count), (prefix, committed_count, name))
    result = []
    for prefix, committed_count, name in unique.values():
        committed = target[:committed_count]
        result.append(
            {
                "id": f"{pair['id']}__{variant}_recovery_{name}_{committed_count:03d}_p{prefix:03d}",
                "parent_id": pair["id"],
                "split": "train",
                "state_prompt": state_prompt(source, prefix, committed),
                "target_action": target[committed_count],
                "target_action_type": "WRITE_STATE_RECOVERY",
                "source_prefix_length": prefix,
                "source_token_count": len(source),
                "source_complete": prefix >= len(source),
                "partial_source_text_en": DETOKENIZER.detokenize(source[:prefix]),
                "committed_pseudo_asl_gloss": " ".join(committed),
                "parent_target_pseudo_asl_gloss": pair["target_pseudo_asl_gloss"],
                "target_is_gold": False,
                "alignment_is_gold": False,
                "alignment_version": f"causalgloss_component_ablation_{variant}_recovery_v1",
                "transition_semantics": "WRITE_keeps_prefix_WAIT_advances_prefix",
                "augmentation_type": name,
                "target_gloss_index": committed_count,
            }
        )
    return result


def config(name: str, stage: str) -> dict[str, Any]:
    second = stage == "stage_b"
    return {
        "experiment_name": f"{name}_{stage}",
        "dataset_version": f"professor_feedback_{name}_{stage}_20260912",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "initialization": "stage_a_best" if second else "finetune1_best_lora",
        "seed": 20260829 if second else 20260817,
        "system_prompt": (
            "You are a simultaneous English-to-pseudo-ASL-gloss translator. Given the observed English prefix "
            "and already committed gloss, output exactly one next action: one uppercase gloss token, "
            "<WAIT>, or <END>. A gloss token commits output without reading new English; <WAIT> requests "
            "exactly one more English token. Several gloss tokens may be written at the same observed "
            "prefix. When Source status is COMPLETE, never output <WAIT>: WRITE if translation remains "
            "incomplete, otherwise output <END>. Do not explain."
        ),
        "wait_token": "<WAIT>",
        "end_token": "<END>",
        "max_length": 512,
        "max_new_tokens": 16,
        "epochs": 1,
        "micro_batch_size_per_gpu": 8,
        "evaluation_batch_size_per_gpu": 16,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 32,
        "learning_rate": 1e-5 if second else 3e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03 if second else 0.05,
        "max_grad_norm": 1.0,
        "evaluation_steps": 2000,
        "logging_steps": 100,
        "early_stopping_patience_evaluations": 3,
        "early_stopping_min_delta": 0.001,
        "early_stopping_minimum_updates": 4000,
        "precision": "bfloat16",
        "trainable_parameters_fp32": False,
        "attention_implementation": "sdpa",
        "gradient_checkpointing": True,
        "gpu_count": 1,
        "training_budget_policy": "one_epoch_per_stage_effective_batch_32_shared_across_all_comparisons",
        "important_warning": "Weak gloss targets and pseudo timing; confirmation and locked sets are never used for selection.",
    }


def write_config(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v22-pairs", type=Path, required=True)
    parser.add_argument("--v1-pairs", type=Path, required=True)
    parser.add_argument("--v1-validation-actions", type=Path, required=True)
    parser.add_argument("--development-pairs", type=Path, required=True)
    parser.add_argument("--confirmation-pairs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    ready = args.output_root / "reports/PREPARATION_COMPLETE.json"
    if ready.exists():
        print(ready.read_text(encoding="utf-8"))
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    development_ids = {row["id"] for row in rows(args.development_pairs)}
    confirmation_ids = {row["id"] for row in rows(args.confirmation_pairs)}
    if development_ids & confirmation_ids or len(development_ids) != 250 or len(confirmation_ids) != 250:
        raise RuntimeError("Frozen development/confirmation split is invalid")

    baseline_dir = args.output_root / "learned_baseline/manifests"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    with (baseline_dir / "development_actions.jsonl").open("w", encoding="utf-8") as dev, (
        baseline_dir / "confirmation_actions.jsonl"
    ).open("w", encoding="utf-8") as conf:
        for row in rows(args.v1_validation_actions):
            if row["parent_id"] in development_ids:
                dev.write(json.dumps(row, ensure_ascii=False) + "\n")
            elif row["parent_id"] in confirmation_ids:
                conf.write(json.dumps(row, ensure_ascii=False) + "\n")

    base_by_id = {row["id"]: row for row in rows(args.v1_pairs)}
    handles: dict[str, dict[str, Any]] = {}
    reports = {name: Counter() for name in VARIANTS}
    for name in VARIANTS:
        manifest_dir = args.output_root / "ablations" / name / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        handles[name] = {
            key: (manifest_dir / f"{key}.jsonl").open("w", encoding="utf-8")
            for key in ("stage_a_train", "stage_b_train", "development_actions", "confirmation_actions")
        }

    try:
        for pair_index, pair in enumerate(rows(args.v22_pairs), 1):
            parent_id = pair["id"]
            if parent_id not in base_by_id:
                raise RuntimeError(f"Missing v1 pair: {parent_id}")
            for name in VARIANTS:
                positions, affected = variant_positions(pair, base_by_id[parent_id], name)
                actions = action_rows(pair, positions, name)
                reports[name]["parent_count"] += 1
                reports[name]["component_affected_token_count"] += affected
                reports[name]["action_count"] += len(actions)
                if pair.get("split") == "train":
                    for row in actions:
                        handles[name]["stage_a_train"].write(json.dumps(row, ensure_ascii=False) + "\n")
                    stage_b = balanced(actions) + recovery_rows(pair, positions, name)
                    for row in stage_b:
                        handles[name]["stage_b_train"].write(json.dumps(row, ensure_ascii=False) + "\n")
                    reports[name]["stage_a_train_action_count"] += len(actions)
                    reports[name]["stage_b_train_action_count"] += len(stage_b)
                elif parent_id in development_ids:
                    for row in actions:
                        handles[name]["development_actions"].write(json.dumps(row, ensure_ascii=False) + "\n")
                    reports[name]["development_parent_count"] += 1
                elif parent_id in confirmation_ids:
                    for row in actions:
                        handles[name]["confirmation_actions"].write(json.dumps(row, ensure_ascii=False) + "\n")
                    reports[name]["confirmation_parent_count"] += 1
            if pair_index % 1000 == 0:
                print(f"[prepare] parents={pair_index}", flush=True)
    finally:
        for variant_handles in handles.values():
            for handle in variant_handles.values():
                handle.close()

    for name, report in reports.items():
        if report["component_affected_token_count"] <= 0:
            raise RuntimeError(f"Ablation {name} changed no component decisions")
        if report["development_parent_count"] != 246 or report["confirmation_parent_count"] != 245:
            raise RuntimeError(
                f"Ablation {name} does not match the frozen v2.3 teacher-forced availability "
                f"(246 development, 245 confirmation): {report}"
            )
        report_path = args.output_root / "ablations" / name / "manifest_report.json"
        report_path.write_text(json.dumps(dict(report), indent=2) + "\n", encoding="utf-8")
        write_config(args.output_root / "ablations" / name / "configs/stage_a.json", config(name, "stage_a"))
        write_config(args.output_root / "ablations" / name / "configs/stage_b.json", config(name, "stage_b"))

    write_config(args.output_root / "learned_baseline/configs/stage_a.json", config("transllama_alignment", "stage_a"))
    write_config(args.output_root / "learned_baseline/configs/stage_b.json", config("transllama_alignment", "stage_b"))
    ready.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "frozen_protocol": True,
        "development_parent_count": len(development_ids),
        "confirmation_parent_count": len(confirmation_ids),
        "teacher_forced_action_parent_count": {
            "development": 246,
            "confirmation": 245,
        },
        "free_running_evaluation_parent_count": {
            "development": 250,
            "confirmation": 250,
        },
        "locked_test_used_during_preparation": False,
        "variants": {name: dict(value) for name, value in reports.items()},
        "input_sha256": {
            "v22_pairs": digest(args.v22_pairs),
            "v1_pairs": digest(args.v1_pairs),
            "v1_validation_actions": digest(args.v1_validation_actions),
            "development_pairs": digest(args.development_pairs),
            "confirmation_pairs": digest(args.confirmation_pairs),
        },
    }
    ready.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
