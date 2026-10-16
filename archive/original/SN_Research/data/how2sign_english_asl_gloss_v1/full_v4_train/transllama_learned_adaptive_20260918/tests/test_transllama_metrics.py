from __future__ import annotations

import unittest

from transllama_metrics import corpus_bleu, length_adaptive_average_lagging, token_overlap


class TransLLaMAMetricTests(unittest.TestCase):
    def test_token_overlap_counts_duplicates(self):
        precision, recall, f1 = token_overlap(["A", "A", "B"], ["A", "B", "B"])
        self.assertAlmostEqual(precision, 2 / 3)
        self.assertAlmostEqual(recall, 2 / 3)
        self.assertAlmostEqual(f1, 2 / 3)

    def test_laal_matches_locked_paper_formula(self):
        value = length_adaptive_average_lagging([2, 3, 5], 5, 3)
        self.assertAlmostEqual(value, (2 + (3 - 5 / 3) + (5 - 10 / 3)) / 3)

    def test_perfect_bleu_is_100(self):
        rows = [
            {
                "predicted_pseudo_asl_gloss_tokens": ["A", "B", "C", "D"],
                "target_pseudo_asl_gloss_tokens": ["A", "B", "C", "D"],
            }
        ]
        self.assertAlmostEqual(corpus_bleu(rows)["bleu_4"], 100.0)


if __name__ == "__main__":
    unittest.main()
