#!/usr/bin/env python3
"""Shared frozen CLERC Epee rollout utilities."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path, expected_count: int) -> list[dict[str, Any]]:
    sources = read_jsonl(path)
    ids = [str(row["id"]) for row in sources]
    if len(sources) != expected_count or len(set(ids)) != expected_count:
        raise ValueError(
            f"Expected {expected_count} unique Epee sources, found "
            f"{len(sources)} rows and {len(set(ids))} IDs"
        )
    required = {
        "id",
        "phrase_id",
        "evaluation_subset",
        "source_text_en",
        "source_tokens",
        "references",
        "median_content_reference_length",
    }
    for source in sources:
        missing = required - set(source)
        if missing:
            raise ValueError(f"{source.get('id')} missing manifest fields: {sorted(missing)}")
        if not source["source_tokens"] or not source["references"]:
            raise ValueError(f"{source['id']} has empty source or references")
    return sources


def shared_max_actions(source: dict[str, Any]) -> int:
    max_reference_length = max(
        [len(ref["content_tokens"]) for ref in source["references"]] or [1]
    )
    return len(source["source_tokens"]) + max(30, 4 * max_reference_length + 10)


def length_adaptive_average_lagging(
    write_positions: list[int], source_count: int, reference_count: int
) -> float | None:
    if not write_positions or source_count <= 0 or reference_count <= 0:
        return None
    ratio = max(reference_count, len(write_positions)) / source_count
    tau = next(
        (
            index
            for index, position in enumerate(write_positions, 1)
            if position >= source_count
        ),
        len(write_positions),
    )
    return round(
        mean(
            write_positions[index - 1] - (index - 1) / ratio
            for index in range(1, tau + 1)
        ),
        8,
    )


def average_proportion(write_positions: list[int], source_count: int) -> float | None:
    if not write_positions or source_count <= 0:
        return None
    return round(sum(write_positions) / (source_count * len(write_positions)), 8)


def initialize_external_state(source: dict[str, Any], *, wait_k: int = 1) -> dict[str, Any]:
    source_tokens = list(source["source_tokens"])
    return {
        "id": str(source["id"]),
        "phrase_id": int(source["phrase_id"]),
        "evaluation_subset": str(source["evaluation_subset"]),
        "source_text_en": str(source["source_text_en"]),
        "source_tokens": source_tokens,
        "reference_count": int(source.get("reference_count", len(source["references"]))),
        "median_reference_length": int(source["median_content_reference_length"]),
        "wait_k": int(wait_k),
        "prefix_length": min(int(wait_k), len(source_tokens)),
        "committed": [],
        "write_positions": [],
        "events": [],
        "done": False,
        "termination_reason": None,
        "max_actions": shared_max_actions(source),
    }


def finalize_external_state(
    state: dict[str, Any],
    *,
    stage: str,
    system_family: str,
    training_seed: int,
    adapter_dir: Path,
) -> dict[str, Any]:
    event_types = Counter(event["predicted_action_class"] for event in state["events"])
    source_count = len(state["source_tokens"])
    prediction = list(state["committed"])
    termination = state["termination_reason"]
    return {
        "stage": stage,
        "system_family": system_family,
        "training_seed": training_seed,
        "adapter_dir": str(adapter_dir),
        "inference_dtype": "float16",
        "decoding": "greedy",
        "id": state["id"],
        "phrase_id": state["phrase_id"],
        "evaluation_subset": state["evaluation_subset"],
        "source_text_en": state["source_text_en"],
        "source_tokens": state["source_tokens"],
        "reference_count": state["reference_count"],
        "median_content_reference_length": state["median_reference_length"],
        "wait_k": state.get("wait_k"),
        "predicted_gloss": " ".join(prediction),
        "predicted_gloss_tokens": prediction,
        "predicted_gloss_token_count": len(prediction),
        "write_source_positions": list(state["write_positions"]),
        "termination_reason": termination,
        "completed_after_full_source": termination == "end_after_full_source",
        "premature_end": termination == "premature_end",
        "action_count": len(state["events"]),
        "max_action_count": state["max_actions"],
        "action_class_counts": dict(event_types),
        "invalid_action_ratio": round(
            event_types["INVALID"] / max(1, len(state["events"])), 8
        ),
        "length_adaptive_average_lagging_source_tokens": (
            length_adaptive_average_lagging(
                state["write_positions"], source_count, state["median_reference_length"]
            )
        ),
        "average_proportion": average_proportion(state["write_positions"], source_count),
        "rollout_events": state["events"],
        "target_is_external_human_reference": True,
        "reference_exposed_to_model": False,
        "rollout_history_uses_reference_gloss": False,
    }


def summarize_rollouts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finite_laal = [
        float(row["length_adaptive_average_lagging_source_tokens"])
        for row in rows
        if row["length_adaptive_average_lagging_source_tokens"] is not None
    ]
    termination = Counter(str(row["termination_reason"]) for row in rows)
    return {
        "source_count": len(rows),
        "termination_reason_counts": dict(termination),
        "valid_completion_rate": (
            sum(bool(row["completed_after_full_source"]) for row in rows) / max(1, len(rows))
        ),
        "no_output_rate": (
            sum(not row["predicted_gloss_tokens"] for row in rows) / max(1, len(rows))
        ),
        "finite_laal_count": len(finite_laal),
        "mean_laal_source_tokens": mean(finite_laal) if finite_laal else None,
        "mean_invalid_action_ratio": mean(
            float(row["invalid_action_ratio"]) for row in rows
        ),
        "mean_action_count": mean(float(row["action_count"]) for row in rows),
        "mean_predicted_gloss_token_count": mean(
            float(row["predicted_gloss_token_count"]) for row in rows
        ),
    }
