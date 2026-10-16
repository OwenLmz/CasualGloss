#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from transllama_data import read_jsonl
from transllama_metrics import summarize_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--system", required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.rows)
    for row in rows:
        predicted_length = int(row["predicted_gloss_token_count"])
        target_length = int(row["target_gloss_token_count"])
        row.setdefault(
            "relative_length_error",
            abs(predicted_length - target_length) / max(1, target_length),
        )
        row.setdefault("over_generated", predicted_length > target_length)
    observed = summarize_rows(rows)
    expected = json.loads(args.expected.read_text(encoding="utf-8"))["systems"][args.system]
    comparisons = {
        "token_f1": (observed["mean_pseudo_gloss_token_f1"], expected["token_f1"]),
        "sequence_similarity": (
            observed["mean_pseudo_gloss_sequence_similarity"],
            expected["sequence_similarity"],
        ),
        "laal": (
            observed["mean_length_adaptive_average_lagging_source_tokens"],
            expected["laal"],
        ),
        "completion": (observed["completion_rate"], expected["completion"]),
        "length_ratio": (observed["length_ratio"], expected["length_ratio"]),
        **{
            f"bleu_{order}": (observed[f"bleu_{order}"], expected[f"bleu_{order}"])
            for order in range(1, 5)
        },
    }
    differences = {
        key: {"observed": pair[0], "expected": pair[1], "absolute_error": abs(pair[0] - pair[1])}
        for key, pair in comparisons.items()
    }
    print(json.dumps(differences, ensure_ascii=False, indent=2))
    if not all(
        math.isclose(observed_value, expected_value, rel_tol=0.0, abs_tol=1e-8)
        for observed_value, expected_value in comparisons.values()
    ):
        raise SystemExit("metric parity check failed")


if __name__ == "__main__":
    main()
