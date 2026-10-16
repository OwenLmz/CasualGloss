#!/usr/bin/env python3
"""Select a deterministic, proportionally stratified 500-parent pilot."""

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
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def stratum(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("candidate_selection_bucket", "unknown")),
            str(row.get("candidate_source_length_bin", "unknown")),
        ]
    )


def allocate_quotas(groups: dict[str, list[dict[str, Any]]], total: int) -> dict[str, int]:
    population = sum(len(rows) for rows in groups.values())
    exact = {key: len(rows) * total / population for key, rows in groups.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = total - sum(quotas.values())
    order = sorted(groups, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    source_path = Path(config["source_manifest"])
    rows = read_jsonl(source_path)
    count = int(config["pilot_parent_count"])
    seed = int(config["selection_seed"])
    shard_count = int(config["shard_count"])
    if len(rows) < count:
        raise ValueError(f"Only {len(rows)} source rows are available for a {count}-row pilot")
    if any(row.get("split") not in (None, "train") for row in rows):
        raise ValueError("The pilot source must contain training parents only")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(row)
    quotas = allocate_quotas(groups, count)
    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda row: stable_key(seed, str(row["id"])))
        selected.extend(ordered[: quotas[key]])
    selected.sort(key=lambda row: stable_key(seed, str(row["id"])))

    prepared = []
    for index, row in enumerate(selected):
        value = dict(row)
        value.update(
            {
                "semantic_v2_pilot_index": index,
                "semantic_v2_selection_seed": seed,
                "semantic_v2_selection_stratum": stratum(row),
                "semantic_v2_shard_id": index % shard_count,
            }
        )
        prepared.append(value)
    write_jsonl(args.output, prepared)
    args.shard_dir.mkdir(parents=True, exist_ok=True)
    shard_counts = {}
    for shard_id in range(shard_count):
        shard_rows = [row for row in prepared if row["semantic_v2_shard_id"] == shard_id]
        write_jsonl(args.shard_dir / f"pilot_parents_shard_{shard_id:02d}.jsonl", shard_rows)
        shard_counts[f"{shard_id:02d}"] = len(shard_rows)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(source_path),
        "source_parent_count": len(rows),
        "selected_parent_count": len(prepared),
        "selection_seed": seed,
        "selection_method": "proportional_stratification_then_seeded_sha256_order",
        "stratum_population_counts": dict(Counter(stratum(row) for row in rows)),
        "stratum_selected_counts": dict(Counter(stratum(row) for row in prepared)),
        "shard_counts": shard_counts,
        "unique_parent_count": len({row["id"] for row in prepared}),
        "all_from_train_split": all(row.get("split") in (None, "train") for row in prepared),
        "ok": len(prepared) == count and len({row["id"] for row in prepared}) == count,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
