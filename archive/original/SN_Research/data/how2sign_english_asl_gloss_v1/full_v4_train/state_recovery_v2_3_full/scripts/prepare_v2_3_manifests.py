#!/usr/bin/env python3
"""Freeze v2.3 recovery data, development split, and confirmation split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from nltk.tokenize import TreebankWordDetokenizer


DETOKENIZER = TreebankWordDetokenizer()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(seed: int, parent_id: str) -> str:
    return hashlib.sha256(f"{seed}:{parent_id}".encode("utf-8")).hexdigest()


def state_prompt(source_tokens: list[str], prefix_length: int, committed: list[str]) -> str:
    source = DETOKENIZER.detokenize(source_tokens[:prefix_length])
    committed_text = " ".join(committed) if committed else "<EMPTY>"
    status = "COMPLETE" if prefix_length >= len(source_tokens) else "INCOMPLETE"
    return (
        "Observed English prefix:\n"
        + source
        + "\n\nAlready committed pseudo ASL gloss:\n"
        + committed_text
        + "\n\nSource status: "
        + status
        + "\nReturn exactly one next action."
    )


def recovery_row(
    pair: dict[str, Any],
    *,
    prefix_length: int,
    committed_count: int,
    recovery_type: str,
) -> dict[str, Any]:
    source_tokens = pair["source_tokens"]
    target_tokens = pair["target_pseudo_asl_gloss_tokens"]
    committed = target_tokens[:committed_count]
    return {
        "id": f"{pair['id']}__v2_3_{recovery_type}_{committed_count:03d}_p{prefix_length:03d}",
        "parent_id": pair["id"],
        "split": "train",
        "state_prompt": state_prompt(source_tokens, prefix_length, committed),
        "target_action": target_tokens[committed_count],
        "target_action_type": "WRITE_STATE_RECOVERY",
        "source_prefix_length": prefix_length,
        "source_token_count": len(source_tokens),
        "source_complete": prefix_length >= len(source_tokens),
        "partial_source_text_en": DETOKENIZER.detokenize(source_tokens[:prefix_length]),
        "committed_pseudo_asl_gloss": " ".join(committed),
        "parent_target_pseudo_asl_gloss": pair["target_pseudo_asl_gloss"],
        "target_is_gold": False,
        "alignment_is_gold": False,
        "alignment_version": "semantic_causal_alignment_v2_3_state_recovery_v1",
        "transition_semantics": "WRITE_keeps_prefix_WAIT_advances_prefix",
        "augmentation_type": recovery_type,
        "target_gloss_index": committed_count,
        "recovery_lag_gloss_tokens": max(1, len(target_tokens) - committed_count),
        "method_note": (
            "Off-policy state built only from How2Sign training pseudo labels. "
            "It teaches WRITE recovery when committed history is behind available source evidence."
        ),
    }


def build_recovery_rows(pair: dict[str, Any]) -> list[dict[str, Any]]:
    source_tokens = pair["source_tokens"]
    target_tokens = pair["target_pseudo_asl_gloss_tokens"]
    if not source_tokens or not target_tokens:
        return []
    source_count = len(source_tokens)
    target_count = len(target_tokens)
    positions = [
        int(item.get("v2_2_emit_source_position", item["scheduled_emit_source_position"]))
        for item in pair["semantic_gloss_token_alignment"]
    ]
    states: list[tuple[int, int, str]] = []

    # Full-source catch-up states cover empty, middle, and one-token-behind histories.
    for committed_count, name in (
        (0, "full_source_empty_history"),
        (target_count // 2, "full_source_mid_history"),
        (target_count - 1, "full_source_tail_history"),
    ):
        states.append((source_count, committed_count, name))

    # A prefix where at least two glosses are already due teaches burst catch-up.
    if target_count >= 2:
        due_index = max(1, round((target_count - 1) * 0.67))
        prefix_length = min(source_count, positions[due_index])
        committed_count = max(0, due_index - 1)
        if committed_count < target_count and positions[committed_count] <= prefix_length:
            states.append((prefix_length, committed_count, "prefix_burst_catchup"))

    unique: dict[tuple[int, int], tuple[int, int, str]] = {}
    for prefix_length, committed_count, name in states:
        if 0 <= committed_count < target_count:
            unique.setdefault((prefix_length, committed_count), (prefix_length, committed_count, name))
    return [
        recovery_row(
            pair,
            prefix_length=prefix_length,
            committed_count=committed_count,
            recovery_type=name,
        )
        for prefix_length, committed_count, name in unique.values()
    ]


def action_class(row: dict[str, Any]) -> str:
    action = str(row.get("target_action", ""))
    if action == "<WAIT>":
        return "WAIT"
    if action == "<END>":
        return "END"
    return "WRITE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal-pairs", type=Path, required=True)
    parser.add_argument("--evaluation-pairs", type=Path, required=True)
    parser.add_argument("--balanced-train-actions", type=Path, required=True)
    parser.add_argument("--validation-actions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--development-parents", type=int, default=250)
    parser.add_argument("--screen-train-parents", type=int, default=2000)
    args = parser.parse_args()

    report_path = args.output_root / "reports/manifest_build_report.json"
    expected = [
        args.output_root / "manifests/train_actions_full.jsonl",
        args.output_root / "manifests/train_actions_screen.jsonl",
        args.output_root / "manifests/development_actions.jsonl",
        args.output_root / "manifests/confirmation_actions.jsonl",
        args.output_root / "manifests/development_pairs.jsonl",
        args.output_root / "manifests/confirmation_pairs.jsonl",
    ]
    if report_path.exists() and all(path.exists() and path.stat().st_size for path in expected):
        print(report_path.read_text(encoding="utf-8"))
        return
    if any(path.exists() for path in expected + [report_path]):
        raise FileExistsError("Refusing to overwrite a partial v2.3 manifest build")

    pairs = list(read_jsonl(args.causal_pairs))
    train_pairs = [row for row in pairs if row.get("split") == "train"]
    validation_pairs = [
        row for row in read_jsonl(args.evaluation_pairs) if row.get("split") == "validation"
    ]
    if len(validation_pairs) != 500:
        raise RuntimeError(f"Expected 500 validation parents, found {len(validation_pairs)}")
    ordered_validation = sorted(
        validation_pairs, key=lambda row: stable_rank(args.seed, str(row["id"]))
    )
    development_pairs = [dict(row, split="v2_3_development") for row in ordered_validation[: args.development_parents]]
    confirmation_pairs = [dict(row, split="v2_3_confirmation") for row in ordered_validation[args.development_parents :]]
    development_ids = {row["id"] for row in development_pairs}
    confirmation_ids = {row["id"] for row in confirmation_pairs}
    if development_ids & confirmation_ids:
        raise RuntimeError("Development and confirmation parent overlap")

    validation_actions = list(read_jsonl(args.validation_actions))
    development_actions = [row for row in validation_actions if row["parent_id"] in development_ids]
    confirmation_actions = [row for row in validation_actions if row["parent_id"] in confirmation_ids]
    action_parent_ids = {row["parent_id"] for row in validation_actions}
    missing_development_action_ids = sorted(development_ids - action_parent_ids)
    missing_confirmation_action_ids = sorted(confirmation_ids - action_parent_ids)
    if not development_actions or not confirmation_actions:
        raise RuntimeError("Frozen split has no v2.2 teacher-forced action states")

    screen_ids = {
        row["id"]
        for row in sorted(train_pairs, key=lambda row: stable_rank(args.seed + 1, str(row["id"])))[: args.screen_train_parents]
    }
    recovery_rows: list[dict[str, Any]] = []
    recovery_counts = Counter()
    for pair in train_pairs:
        built = build_recovery_rows(pair)
        recovery_rows.extend(built)
        recovery_counts.update(row["augmentation_type"] for row in built)

    balanced_actions = list(read_jsonl(args.balanced_train_actions))
    full_train = [dict(row, calibration_source="balanced_on_policy") for row in balanced_actions]
    full_train.extend(dict(row, calibration_source="state_recovery") for row in recovery_rows)
    screen_train = [row for row in full_train if row["parent_id"] in screen_ids]
    if not screen_train or not full_train:
        raise RuntimeError("Empty v2.3 training manifest")
    if len({row["id"] for row in full_train}) != len(full_train):
        raise RuntimeError("Duplicate action IDs in v2.3 full training manifest")

    outputs = {
        "train_actions_full": (args.output_root / "manifests/train_actions_full.jsonl", full_train),
        "train_actions_screen": (args.output_root / "manifests/train_actions_screen.jsonl", screen_train),
        "development_actions": (args.output_root / "manifests/development_actions.jsonl", development_actions),
        "confirmation_actions": (args.output_root / "manifests/confirmation_actions.jsonl", confirmation_actions),
        "development_pairs": (args.output_root / "manifests/development_pairs.jsonl", development_pairs),
        "confirmation_pairs": (args.output_root / "manifests/confirmation_pairs.jsonl", confirmation_pairs),
        "recovery_actions": (args.output_root / "augmentation/state_recovery_actions_train.jsonl", recovery_rows),
    }
    for _, (path, rows) in outputs.items():
        write_jsonl(path, rows)

    report = {
        "created_for": "v2.3 state-distribution recovery augmentation",
        "scientific_status": "pseudo_gloss_state_recovery_not_gold_asl",
        "seed": args.seed,
        "source_train_parent_count": len(train_pairs),
        "source_validation_parent_count": len(validation_pairs),
        "development_parent_count": len(development_pairs),
        "confirmation_parent_count": len(confirmation_pairs),
        "development_parent_count_with_teacher_forced_actions": len(
            development_ids - set(missing_development_action_ids)
        ),
        "confirmation_parent_count_with_teacher_forced_actions": len(
            confirmation_ids - set(missing_confirmation_action_ids)
        ),
        "missing_development_teacher_forced_action_ids": missing_development_action_ids,
        "missing_confirmation_teacher_forced_action_ids": missing_confirmation_action_ids,
        "screen_train_parent_count": len(screen_ids),
        "balanced_on_policy_action_count": len(balanced_actions),
        "recovery_action_count": len(recovery_rows),
        "recovery_type_counts": dict(recovery_counts),
        "full_train_action_count": len(full_train),
        "screen_train_action_count": len(screen_train),
        "full_action_class_counts": dict(Counter(action_class(row) for row in full_train)),
        "selection_protocol": {
            "development": "checkpoint selection and the single screen gate only",
            "confirmation": "frozen before v2.3 training; never used for checkpoint or hyperparameter selection",
            "locked_500": "not accessed by this preparation or screen stage",
            "clerc_epee": "not accessed by this preparation or screen stage",
        },
        "method": {
            "on_policy_component": "immutable v2.2.1 WAIT-balanced training states",
            "off_policy_component": [
                "full source with empty committed history",
                "full source with mid-sequence lag",
                "full source with one-token lag",
                "same-prefix burst catch-up with multiple available glosses",
            ],
            "source_end_rule": "when source is COMPLETE and gloss remains, target action is WRITE, never WAIT",
        },
        "sha256_inputs": {
            "causal_pairs": sha256(args.causal_pairs),
            "evaluation_pairs": sha256(args.evaluation_pairs),
            "balanced_train_actions": sha256(args.balanced_train_actions),
            "validation_actions": sha256(args.validation_actions),
        },
        "sha256_outputs": {name: sha256(path) for name, (path, _) in outputs.items()},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol_path = args.output_root / "reports/frozen_method_protocol.json"
    protocol_path.write_text(
        json.dumps(
            {
                "frozen_before_training": True,
                "seed": args.seed,
                "screen_gate": {
                    "valid_end_completion_rate_minimum": 0.99,
                    "mean_invalid_action_ratio_maximum": 0.01,
                    "maximum_actions_reached_rate_maximum": 0.01,
                    "token_f1_vs_v2_2_minimum_difference": -0.005,
                    "sequence_similarity_vs_v2_2_minimum_difference": -0.005,
                    "laal_vs_v2_2_maximum_increase_source_tokens": 0.20,
                    "require_one_recovery_improvement": True,
                },
                "checkpoint_selection": "eligible checkpoint with highest development token F1; tie-break by sequence similarity then lower LAAL",
                "no_tuning_sets": ["v2_3_confirmation_250", "locked_500", "CLERC_Epee_401"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
