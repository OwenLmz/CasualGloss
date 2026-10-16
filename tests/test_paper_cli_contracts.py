"""Small synthetic CLI runs; no model weights or experimental data required."""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CLERC = ROOT / "evaluation" / "clerc"


def write_fixture_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class PaperCliContractTests(unittest.TestCase):
    def test_clerc_adaptive_runs_without_unreported_lexical_input(self):
        with patch.object(sys, "path", [str(CLERC), *sys.path]):
            comparison = importlib.import_module("compare_epee_adaptive")
        with tempfile.TemporaryDirectory(prefix="causalgloss_clerc_cli_") as temp:
            directory = Path(temp)
            manifest = directory / "sources.jsonl"
            sources = [
                {
                    "id": f"synthetic-{index}",
                    "phrase_id": index,
                    "evaluation_subset": "synthetic_test_fixture",
                    "source_text_en": "synthetic test fixture",
                    "source_tokens": ["synthetic", "test", "fixture"],
                    "references": [{"strict_tokens": ["A", "B"], "content_tokens": ["A", "B"]}],
                    "median_content_reference_length": 2,
                }
                for index in range(2)
            ]
            write_fixture_jsonl(manifest, sources)
            output = directory / "comparison"
            argv = [
                "compare_epee_adaptive.py", "--manifest", str(manifest),
                "--expected-count", "2", "--k-values", "5", "--confirmatory-k", "5",
                "--bootstrap-repetitions", "32", "--output-dir", str(output),
            ]
            for seed in ("20260829", "20260904", "20260905"):
                causal_path = directory / f"causal-{seed}.jsonl"
                learned_dir = directory / f"learned-{seed}"
                def rows(tokens, laal):
                    return [
                        {
                            "id": source["id"], "predicted_gloss_tokens": tokens,
                            "completed_after_full_source": True,
                            "termination_reason": "end_after_full_source",
                            "invalid_action_ratio": 0.0, "action_count": 4,
                            "length_adaptive_average_lagging_source_tokens": laal,
                        }
                        for source in sources
                    ]
                write_fixture_jsonl(causal_path, rows(["A", "B"], 1.0))
                write_fixture_jsonl(learned_dir / "k5" / "sentence_rollouts.jsonl", rows(["A", "C"], 3.0))
                argv.extend(["--causal", f"{seed}={causal_path}", "--learned", f"{seed}={learned_dir}"])
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                comparison.main()
            report = json.loads((output / "comparison_report.json").read_text(encoding="utf-8"))
            self.assertEqual(set(report["systems"]), {"causalgloss", "transllama_style"})
            self.assertNotIn("lexical", report["input_hashes"])
            self.assertNotIn("lexical_comparisons", report)
            quality = report["confirmatory"]["quality_causal_minus_learned"]
            self.assertEqual(quality["mean_difference"], 0.5)
            self.assertEqual(quality["ci95"], [0.5, 0.5])
            self.assertEqual(quality["matched_seed_count"], 3)
            self.assertEqual(quality["paired_source_count_by_seed"], [2, 2, 2])
            self.assertEqual(report["confirmatory"]["laal_causal_minus_learned"]["mean_difference"], -2.0)
            scores = [json.loads(line) for line in (output / "per_source_scores.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(scores), 12)
            self.assertEqual({row["system"] for row in scores}, {"causalgloss", "transllama_style"})

    def test_fixed_comparison_shared_scoring_api_is_preserved(self):
        with patch.object(sys, "path", [str(CLERC), *sys.path]):
            adaptive = importlib.import_module("compare_epee_adaptive")
            fixed = importlib.import_module("compare_fixed_waitk_epee")
        for name in ("load_exact_rollout", "paired_source_bootstrap", "parse_named_path", "percentile_interval", "score_rollout", "summarize_seeded"):
            with self.subTest(function=name):
                self.assertIs(getattr(fixed, name), getattr(adaptive, name))

    def test_obsolete_inference_only_bleu_workflow_is_absent(self):
        self.assertFalse((ROOT / "evaluation/analysis/evaluate_locked500_bleu.py").exists())


if __name__ == "__main__":
    unittest.main()
