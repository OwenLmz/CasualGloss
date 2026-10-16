#!/usr/bin/env python3
"""Write a compact machine-readable and Markdown summary for v2.3."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else None


def digest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"exists": True, "size_bytes": path.stat().st_size, "sha256": value}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    selection = read_json(args.root / "reports/validation_policy_selection.json")
    test = read_json(args.root / "reports/test_selected.json")
    comparison = read_json(args.root / "reports/locked500_v1_vs_v2_3_comparison.json")
    decision = read_json(args.root / "reports/locked500_v2_3_decision.json")
    core_paths = [
        args.root / "reports/finetune1_prefix_predictions_validation.json",
        args.root / "reports/validation_policy_selection.json",
        args.root / "reports/finetune1_prefix_predictions_test.json",
        args.root / "reports/test_selected.json",
        args.root / "reports/locked500_v1_vs_v2_3_comparison.json",
        args.root / "reports/locked500_v2_3_decision.json",
        args.root / "scripts/generate_finetune1_prefix_predictions.py",
        args.root / "scripts/evaluate_local_agreement_policy.py",
        args.root / "scripts/select_validation_local_agreement_policy.py",
    ]
    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "v2_3_local_agreement_against_pseudo_gloss_not_gold_asl",
        "validation_policy_selection": selection,
        "locked_test_metrics": test,
        "locked_test_comparison": comparison,
        "locked_test_decision": decision,
        "files": {
            str(path.relative_to(args.root)): digest(path) for path in core_paths
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Local-Agreement Causal Alignment v2.3 Summary",
        "",
        "All quality references are Qwen-generated pseudo glosses, not expert gold ASL.",
        "",
    ]
    if selection:
        selected = selection["selected"]
        lines.extend(
            [
                "## Validation-only policy selection",
                "",
                f"- Status: `{selection['selection_status']}`",
                f"- Stability window: {selected['stability_window']}",
                f"- Minimum prefix tokens: {selected['minimum_prefix_tokens']}",
                f"- Sequence similarity: {selected['sequence_similarity']:.4f}",
                f"- Token F1: {selected['token_f1']:.4f}",
                f"- LAAL: {selected['laal']:.4f} source tokens",
                f"- Irreversible-conflict ratio: {selected['conflict_ratio']:.4f}",
                "",
            ]
        )
    if test:
        lines.extend(
            [
                "## Locked 500-parent test",
                "",
                f"- Sequence similarity: {test['pseudo_label_sequence_similarity']:.4f}",
                f"- Token F1: {test['pseudo_label_token_f1']:.4f}",
                f"- Exact match: {test['pseudo_label_sequence_exact_match']:.4f}",
                f"- LAAL: {test['mean_length_adaptive_average_lagging_source_tokens']:.4f} source tokens",
                f"- Conflict ratio: {test['irreversible_conflict_parent_ratio']:.4f}",
                "",
            ]
        )
    if decision:
        lines.extend(
            [
                "## Predeclared decision",
                "",
                f"- Selected candidate: `{decision.get('selected_candidate')}`",
                f"- Fallback: `{decision.get('fallback')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "This experiment measures latency and imitation of pseudo-gloss targets. It does not establish ASL linguistic correctness.",
            "",
        ]
    )
    args.output_markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    main()
