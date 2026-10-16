from __future__ import annotations

import unittest

from transllama_training import (
    deterministic_prefix_length,
    encode_prefix_example,
    wait_token_ids,
)


class FakeTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.assert_prompt_args = (tokenize, add_generation_prompt)
        return [10, 11, 12]

    def encode(self, text, add_special_tokens=False):
        mapping = {"@@": [20], " @@": [21], "ONE": [30], "ONE @@": [30, 21]}
        return mapping[text]


class TransLLaMATrainingTests(unittest.TestCase):
    def test_prefix_sampling_is_deterministic_and_in_range(self):
        first = deterministic_prefix_length("parent", 20, 7, 2)
        second = deterministic_prefix_length("parent", 20, 7, 2)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 1)
        self.assertLessEqual(first, 20)

    def test_incomplete_prefix_masks_prompt_and_has_no_eos(self):
        tokenizer = FakeTokenizer()
        encoded = encode_prefix_example(
            {
                "id": "x__prefix_0001",
                "parent_id": "x",
                "source_prefix_length": 1,
                "source_complete": False,
                "source_prefix_text": "one",
                "assistant_target_text": "ONE @@",
            },
            tokenizer,
            "system",
            32,
        )
        self.assertEqual(encoded["input_ids"], [10, 11, 12, 30, 21])
        self.assertEqual(encoded["labels"], [-100, -100, -100, 30, 21])
        self.assertNotIn(tokenizer.eos_token_id, encoded["labels"])

    def test_complete_prefix_adds_native_eos(self):
        tokenizer = FakeTokenizer()
        encoded = encode_prefix_example(
            {
                "id": "x__prefix_0002",
                "parent_id": "x",
                "source_prefix_length": 2,
                "source_complete": True,
                "source_prefix_text": "one two",
                "assistant_target_text": "ONE",
            },
            tokenizer,
            "system",
            32,
        )
        self.assertEqual(encoded["labels"], [-100, -100, -100, 30, 99])

    def test_wait_ids_cover_sentence_initial_and_space_prefixed_forms(self):
        self.assertEqual(wait_token_ids(FakeTokenizer()), {20, 21})


if __name__ == "__main__":
    unittest.main()
