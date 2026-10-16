from __future__ import annotations

import unittest

from evaluate_transllama import (
    apply_action,
    check_for_loop,
    native_context_window,
    parse_generated_action,
)


class FakeTokenizer:
    eos_token_id = 99

    def convert_ids_to_tokens(self, ids):
        pieces = {
            20: "@@",
            21: "Ġ@@",
            30: "GO",
            31: "ĠDOWN",
            32: "D",
            33: "OWN",
            34: "EITHER",
            35: "/",
            36: "OR",
            37: "WE",
            38: "'RE",
            39: "ĠGOING",
            40: "START",
            41: "-OFF",
            42: "ĠNEXT",
        }
        return [pieces.get(value, "?") for value in ids]

    def decode(
        self,
        ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        mapping = {
            (20,): "@@",
            (21,): " @@",
            (30,): "GO",
            (30, 31): "GO DOWN",
            (32, 33): "DOWN",
            (32, 33, 21): "DOWN @@",
            (34, 35, 36): "EITHER/OR",
            (37,): "WE",
            (37, 38): "WE'RE",
            (40,): "START",
            (40, 41): "START-OFF",
            (50,): "AAAAAA",
            (99,): "<|im_end|>",
        }
        if skip_special_tokens and tuple(ids) == (99,):
            return ""
        return mapping.get(tuple(ids), "")


class FakeModel:
    class Config:
        max_position_embeddings = 32768

    config = Config()


class TransLLaMAEvaluatorTests(unittest.TestCase):
    def test_official_loop_detector_rejects_repeated_characters(self):
        self.assertTrue(check_for_loop("AAAAAA", max_repeated_chars=6))

    def test_official_loop_detector_allows_ordinary_text(self):
        self.assertFalse(check_for_loop("GO DOWN NEXT", max_repeated_chars=6))

    def test_continue_loop_terminates_without_advancing_source(self):
        state = {
            "source_tokens": ["ONE", "TWO"],
            "prefix_length": 1,
            "pending_token_ids": [],
            "target_token_ids": [],
            "events": [],
            "committed": [],
            "write_positions": [],
            "done": False,
            "termination_reason": None,
            "max_actions": 20,
        }
        apply_action(
            state,
            "<CONTINUE>",
            "CONTINUE",
            "AAAAAA",
            [50],
            [50],
            False,
            FakeTokenizer(),
        )
        self.assertTrue(state["done"])
        self.assertEqual(state["termination_reason"], "loop_detected")
        self.assertEqual(state["prefix_length"], 1)
        self.assertEqual(state["events"][0]["predicted_action_class"], "LOOP")

    def test_inference_uses_native_model_context_window(self):
        tokenizer = FakeTokenizer()
        tokenizer.model_max_length = 131072
        self.assertEqual(native_context_window(FakeModel(), tokenizer), 32768)

    def test_wait_at_start(self):
        action, kind, *_ = parse_generated_action(FakeTokenizer(), [20], {20, 21})
        self.assertEqual((action, kind), ("@@", "WAIT"))

    def test_native_eos_is_end(self):
        action, kind, *_ = parse_generated_action(FakeTokenizer(), [99], {20, 21})
        self.assertEqual((action, kind), ("<END>", "END"))

    def test_only_first_complete_gloss_is_committed(self):
        action, kind, *_ = parse_generated_action(FakeTokenizer(), [30, 31], {20, 21})
        self.assertEqual((action, kind), ("GO", "WRITE"))

    def test_multi_piece_gloss_is_buffered_before_wait(self):
        action, kind, _, accepted, pending, finish = parse_generated_action(
            FakeTokenizer(), [32, 33, 21], {20, 21}
        )
        self.assertEqual((action, kind), ("@@", "WAIT"))
        self.assertEqual(accepted, [32, 33])
        self.assertEqual(pending, [32, 33])
        self.assertFalse(finish)

    def test_slash_gloss_is_valid(self):
        action, kind, *_ = parse_generated_action(
            FakeTokenizer(), [34, 35, 36, 31], {20, 21}
        )
        self.assertEqual((action, kind), ("EITHER/OR", "WRITE"))

    def test_apostrophe_continuation_completes_buffered_word(self):
        action, kind, _, accepted, pending, finish = parse_generated_action(
            FakeTokenizer(), [38, 39], {20, 21}, pending_ids=[37]
        )
        self.assertEqual((action, kind), ("WE'RE", "WRITE"))
        self.assertEqual(accepted, [38])
        self.assertEqual(pending, [])
        self.assertFalse(finish)

    def test_hyphen_continuation_completes_buffered_word(self):
        action, kind, _, accepted, pending, finish = parse_generated_action(
            FakeTokenizer(), [41, 42], {20, 21}, pending_ids=[40]
        )
        self.assertEqual((action, kind), ("START-OFF", "WRITE"))
        self.assertEqual(accepted, [41])
        self.assertEqual(pending, [])
        self.assertFalse(finish)

    def test_eos_emits_pending_word_and_finishes(self):
        action, kind, _, accepted, pending, finish = parse_generated_action(
            FakeTokenizer(), [99], {20, 21}, pending_ids=[34, 35, 36]
        )
        self.assertEqual((action, kind), ("EITHER/OR", "WRITE"))
        self.assertEqual(accepted, [])
        self.assertEqual(pending, [])
        self.assertTrue(finish)


if __name__ == "__main__":
    unittest.main()
