from __future__ import annotations

import unittest

from transllama_data import WAIT_SYMBOL, align_pair, fix_alignment, make_prefix_example


class FakeAligner:
    def __init__(self, edges):
        self.edges = edges

    def get_word_aligns(self, source, target):
        return {"itermax": list(self.edges)}


class TransLLaMADataTests(unittest.TestCase):
    def test_manifest_tokens_are_preserved(self):
        row = align_pair(
            parent_id="x",
            split="train",
            source_text="Let's test.",
            target_text="LET'S TEST",
            source_words=["Let", "'s", "test", "."],
            target_words=["LET'S", "TEST"],
            aligner=FakeAligner([(0, 0), (2, 1)]),
            input_line_index=0,
        )
        self.assertEqual(row["target_words"], ["LET'S", "TEST"])
        self.assertEqual(row["tokenization"], "frozen_shared_manifest_tokens")

    def test_fix_alignment_keeps_latest_edges(self):
        edges = [(0, 0), (1, 0), (1, 1), (2, 2)]
        self.assertEqual(fix_alignment(edges), [(1, 1), (2, 2)])

    def test_wait_insertion_makes_late_alignment_causal(self):
        row = align_pair(
            parent_id="x",
            split="train",
            source_text="one two three four",
            target_text="THREE FOUR",
            aligner=FakeAligner([(2, 0), (3, 1)]),
            input_line_index=0,
        )
        self.assertEqual(row["aligned_target_words"], [WAIT_SYMBOL, WAIT_SYMBOL, "THREE", "FOUR"])
        self.assertEqual(row["inserted_wait_count"], 2)

    def test_concise_target_gets_explicit_tail_waits(self):
        row = align_pair(
            parent_id="x",
            split="train",
            source_text="one two three four",
            target_text="ONE TWO",
            aligner=FakeAligner([(0, 0), (1, 1)]),
            input_line_index=0,
        )
        self.assertEqual(row["aligned_target_words"], ["ONE", "TWO", WAIT_SYMBOL, WAIT_SYMBOL])
        self.assertEqual(row["target_tail_wait_count"], 2)

    def test_long_target_pads_only_source(self):
        row = align_pair(
            parent_id="x",
            split="train",
            source_text="one two",
            target_text="ONE TWO EXTRA",
            aligner=FakeAligner([(0, 0), (1, 1)]),
            input_line_index=0,
        )
        self.assertEqual(row["aligned_source_words"], ["one", "two", WAIT_SYMBOL])
        self.assertEqual(row["source_tail_filler_count"], 1)

    def test_prefix_keeps_only_final_wait(self):
        row = {
            "id": "x",
            "split": "train",
            "source_words": ["one", "two", "three", "four"],
            "target_words": ["THREE", "FOUR"],
            "aligned_source_words": ["one", "two", "three", "four"],
            "aligned_target_words": [WAIT_SYMBOL, WAIT_SYMBOL, "THREE", "FOUR"],
        }
        first = make_prefix_example(row, 1)
        third = make_prefix_example(row, 3)
        final = make_prefix_example(row, 4)
        self.assertEqual(first["assistant_target_text"], WAIT_SYMBOL)
        self.assertEqual(third["assistant_target_text"], "THREE")
        self.assertEqual(final["assistant_target_text"], "THREE FOUR")


if __name__ == "__main__":
    unittest.main()
