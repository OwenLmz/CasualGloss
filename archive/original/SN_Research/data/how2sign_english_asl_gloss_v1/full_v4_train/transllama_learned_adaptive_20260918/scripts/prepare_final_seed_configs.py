#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from transllama_data import write_json


FINAL_SEEDS = (20260829, 20260904, 20260905)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    selection_path = root / "runs" / "lr_screen_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection.get("selected")
    if not selected:
        raise SystemExit("learning-rate selection is incomplete")
    passing = {
        candidate["name"]: candidate
        for candidate in selection["candidates"]
        if candidate["passed_sanity_gate"]
    }
    if selected not in passing:
        raise SystemExit("selected candidate did not pass the frozen sanity gate")

    source_config_path = root / "configs" / f"{selected}.json"
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    manifest = {
        "selection_file": str(selection_path),
        "selection_rule": selection["selection_rule"],
        "selected_candidate": selected,
        "selected_learning_rate": source_config["learning_rate"],
        "locked_test_accessed": False,
        "seeds": list(FINAL_SEEDS),
        "configs": [],
    }
    for seed in FINAL_SEEDS:
        name = f"final_seed_{seed}"
        config = dict(source_config)
        config["experiment_name"] = f"transllama_style_{name}"
        config["seed"] = seed
        output_path = root / "configs" / f"{name}.json"
        write_json(output_path, config)
        manifest["configs"].append(str(output_path))
    write_json(root / "runs" / "final_seed_config_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
