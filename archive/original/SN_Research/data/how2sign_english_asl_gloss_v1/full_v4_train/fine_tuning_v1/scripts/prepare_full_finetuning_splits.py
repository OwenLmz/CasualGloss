#!/usr/bin/env python3
"""Build deterministic source-video-disjoint full fine-tuning splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{namespace}|{value}".encode("utf-8")).hexdigest()


class DisjointSet:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def select_components(
    components: list[dict[str, Any]], target_count: int, seed: int, namespace: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(
        components,
        key=lambda component: stable_key(seed, namespace, component["component_id"]),
    )
    selected: list[dict[str, Any]] = []
    count = 0
    for component in ordered:
        if count >= target_count:
            break
        selected.append(component)
        count += len(component["request_ids"])
    selected_ids = {component["component_id"] for component in selected}
    remaining = [
        component
        for component in components
        if component["component_id"] not in selected_ids
    ]
    return selected, remaining


def sampled_eval_rows(
    rows: list[dict[str, Any]], count: int, seed: int, namespace: str
) -> list[dict[str, Any]]:
    flagged = [row for row in rows if row["automatic_review_flagged"]]
    unflagged = [row for row in rows if not row["automatic_review_flagged"]]
    flagged_target = round(count * len(flagged) / max(1, len(rows)))
    ordered_flagged = sorted(
        flagged, key=lambda row: stable_key(seed, namespace + "-flagged", row["id"])
    )
    ordered_unflagged = sorted(
        unflagged,
        key=lambda row: stable_key(seed, namespace + "-unflagged", row["id"]),
    )
    selected = ordered_flagged[:flagged_target] + ordered_unflagged[
        : count - flagged_target
    ]
    return sorted(
        selected, key=lambda row: stable_key(seed, namespace + "-order", row["id"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unique-predictions-jsonl", type=Path, required=True)
    parser.add_argument("--all-records-jsonl", type=Path, required=True)
    parser.add_argument("--audit-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--generation-eval-count", type=int, default=500)
    args = parser.parse_args()

    unique_rows = read_jsonl(args.unique_predictions_jsonl)
    all_records = read_jsonl(args.all_records_jsonl)
    audit_by_id = {
        str(row["request_id"]): row for row in read_jsonl(args.audit_jsonl)
    }
    unique_by_id = {str(row["request_id"]): row for row in unique_rows}
    if len(unique_rows) != len(unique_by_id):
        raise SystemExit("Duplicate request_id values in unique predictions")
    if set(unique_by_id) != set(audit_by_id):
        raise SystemExit("Unique predictions and audit request IDs differ")

    videos_by_request: dict[str, set[str]] = defaultdict(set)
    requests_by_video: dict[str, set[str]] = defaultdict(set)
    for record in all_records:
        request_id = str(record["request_id"])
        video_id = str(record["how2sign_video_id"])
        if request_id not in unique_by_id:
            raise SystemExit(f"Unknown request_id in all records: {request_id}")
        videos_by_request[request_id].add(video_id)
        requests_by_video[video_id].add(request_id)
    if set(videos_by_request) != set(unique_by_id):
        raise SystemExit("Some unique requests have no official source-video mapping")

    dsu = DisjointSet(list(unique_by_id))
    for request_ids in requests_by_video.values():
        ordered = sorted(request_ids)
        for request_id in ordered[1:]:
            dsu.union(ordered[0], request_id)

    component_request_ids: dict[str, list[str]] = defaultdict(list)
    for request_id in unique_by_id:
        component_request_ids[dsu.find(request_id)].append(request_id)
    components: list[dict[str, Any]] = []
    for request_ids in component_request_ids.values():
        request_ids = sorted(request_ids)
        video_ids = sorted(
            set().union(*(videos_by_request[request_id] for request_id in request_ids))
        )
        component_id = hashlib.sha256("|".join(request_ids).encode("utf-8")).hexdigest()
        components.append(
            {
                "component_id": component_id,
                "request_ids": request_ids,
                "video_ids": video_ids,
            }
        )

    total_count = len(unique_rows)
    test_target = round(total_count * args.test_ratio)
    validation_target = round(total_count * args.validation_ratio)
    test_components, remaining = select_components(
        components, test_target, args.seed, "test-components"
    )
    validation_components, train_components = select_components(
        remaining, validation_target, args.seed, "validation-components"
    )
    components_by_split = {
        "train": train_components,
        "validation": validation_components,
        "test": test_components,
    }

    request_split: dict[str, str] = {}
    for split, split_components in components_by_split.items():
        for component in split_components:
            for request_id in component["request_ids"]:
                if request_id in request_split:
                    raise SystemExit(f"Request assigned twice: {request_id}")
                request_split[request_id] = split
    if set(request_split) != set(unique_by_id):
        raise SystemExit("Split assignment does not cover all unique requests")

    split_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request_id, row in unique_by_id.items():
        audit = audit_by_id[request_id]
        enriched = {
            **row,
            "id": request_id,
            "split": request_split[request_id],
            "how2sign_video_ids": sorted(videos_by_request[request_id]),
            "automatic_review_flags": list(audit.get("automatic_review_flags", [])),
            "automatic_review_flagged": bool(audit.get("automatic_review_flags", [])),
            "automatic_review_priority": audit.get("automatic_review_priority"),
            "transparent_lexical_coverage_proxy": audit.get(
                "transparent_lexical_coverage_proxy"
            ),
        }
        split_rows[request_split[request_id]].append(enriched)
    for split in split_rows:
        split_rows[split] = sorted(
            split_rows[split],
            key=lambda row: stable_key(args.seed, split + "-row-order", row["id"]),
        )

    output_paths: dict[str, Path] = {}
    for split in ("train", "validation", "test"):
        path = args.output_dir / f"{split}.jsonl"
        write_jsonl(path, split_rows[split])
        output_paths[split] = path
    validation_generation = sampled_eval_rows(
        split_rows["validation"],
        min(args.generation_eval_count, len(split_rows["validation"])),
        args.seed,
        "validation-generation",
    )
    test_generation = sampled_eval_rows(
        split_rows["test"],
        min(args.generation_eval_count, len(split_rows["test"])),
        args.seed,
        "test-generation",
    )
    for name, rows in (
        ("validation_generation_eval", validation_generation),
        ("test_generation_eval", test_generation),
    ):
        path = args.output_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        output_paths[name] = path

    request_sets = {
        split: {row["id"] for row in split_rows[split]}
        for split in ("train", "validation", "test")
    }
    video_sets = {
        split: set().union(*(set(row["how2sign_video_ids"]) for row in split_rows[split]))
        for split in ("train", "validation", "test")
    }
    request_disjoint = not (
        request_sets["train"] & request_sets["validation"]
        or request_sets["train"] & request_sets["test"]
        or request_sets["validation"] & request_sets["test"]
    )
    video_disjoint = not (
        video_sets["train"] & video_sets["validation"]
        or video_sets["train"] & video_sets["test"]
        or video_sets["validation"] & video_sets["test"]
    )
    status_counts = {
        split: dict(Counter(row["generation_status"] for row in split_rows[split]))
        for split in ("train", "validation", "test")
    }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": "connected How2Sign source-video-group-disjoint split",
        "seed": args.seed,
        "source": {
            "unique_predictions_jsonl": str(args.unique_predictions_jsonl),
            "unique_predictions_sha256": file_sha256(args.unique_predictions_jsonl),
            "all_records_jsonl": str(args.all_records_jsonl),
            "all_records_sha256": file_sha256(args.all_records_jsonl),
            "audit_jsonl": str(args.audit_jsonl),
            "audit_sha256": file_sha256(args.audit_jsonl),
        },
        "total_unique_requests": total_count,
        "component_count": len(components),
        "largest_component_request_count": max(
            len(component["request_ids"]) for component in components
        ),
        "largest_component_video_count": max(
            len(component["video_ids"]) for component in components
        ),
        "counts": {split: len(split_rows[split]) for split in split_rows},
        "video_counts": {split: len(video_sets[split]) for split in video_sets},
        "flagged_counts": {
            split: sum(row["automatic_review_flagged"] for row in split_rows[split])
            for split in split_rows
        },
        "fallback_counts": {
            split: sum(
                row["generation_status"] == "completed_with_rule_based_fallback"
                for row in split_rows[split]
            )
            for split in split_rows
        },
        "generation_status_counts": status_counts,
        "request_splits_are_disjoint": request_disjoint,
        "source_video_splits_are_disjoint": video_disjoint,
        "all_targets_marked_non_gold": all(
            row.get("pseudo_gloss_is_gold") is False for row in unique_rows
        ),
        "generation_eval_counts": {
            "validation": len(validation_generation),
            "test": len(test_generation),
        },
        "output_paths": {name: str(path) for name, path in output_paths.items()},
        "output_sha256": {
            name: file_sha256(path) for name, path in output_paths.items()
        },
    }
    report["pipeline_ok"] = all(
        [
            sum(report["counts"].values()) == total_count,
            request_disjoint,
            video_disjoint,
            report["all_targets_marked_non_gold"],
            len(validation_generation) == min(args.generation_eval_count, len(split_rows["validation"])),
            len(test_generation) == min(args.generation_eval_count, len(split_rows["test"])),
        ]
    )
    report["important_limitation"] = (
        "The split prevents request/video leakage but the targets remain generated "
        "pseudo glosses, not qualified ASL annotations."
    )
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pipeline_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
