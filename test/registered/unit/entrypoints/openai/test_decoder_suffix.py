"""Suffix publication and bounded decoder-prefix regressions."""

import re
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.openai.realtime.decoder_suffix import (
    DecoderSuffixState,
    SuffixUpdate,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDecoderSuffixState(CustomTestCase):
    def test_agreement_final_pending_and_repeated_speech(self):
        state = DecoderSuffixState("one", pending="two three four")
        update = state.reconcile("two three five", is_last=False, holdback_units=1)
        self.assertEqual(update, SuffixUpdate(delta="two", pending="three five"))
        state.apply(update)
        self.assertEqual(state.emitted_text, "one two")
        state.apply(state.reconcile("", is_last=True, holdback_units=1))
        self.assertEqual(state.emitted_text, "one two three five")
        state.apply(state.reconcile("one two", is_last=True, holdback_units=1))
        self.assertEqual(state.emitted_text, "one two three five one two")
        self.assertEqual(state.flush(), "")

    def test_confirmed_holdback_preserves_word_extensions_and_repeated_speech(self):
        tokenizer = SimpleNamespace(
            encode=lambda text, **kwargs: list(text),
            decode=lambda tokens, **kwargs: "".join(tokens),
        )
        for continuation, expected in (("pet", "carpet"), (" pet", "car pet")):
            with self.subTest(continuation=continuation):
                state = DecoderSuffixState("", pending="car")
                state.apply(state.reconcile("car", is_last=False, holdback_units=1))
                self.assertEqual(state.emitted_text, "")
                self.assertEqual(state.confirmed_pending, "car")
                self.assertEqual(state.bounded_prefix(tokenizer, 10), "car")
                self.assertEqual(state.bounded_prefix(tokenizer, 2), "")
                state.apply(
                    state.reconcile(continuation, is_last=False, holdback_units=1)
                )
                self.assertEqual(state.confirmed_pending, "car")
                self.assertEqual(state.pending, expected)
                state.apply(
                    state.reconcile(continuation, is_last=False, holdback_units=1)
                )
                state.flush()
                self.assertEqual(state.emitted_text, expected)
                self.assertEqual(state.confirmed_pending_chars, 0)
        state = DecoderSuffixState("", pending="very very")
        state.apply(state.reconcile("very very", is_last=False, holdback_units=1))
        state.flush()
        self.assertEqual(state.emitted_text, "very very")

    def test_punctuation_and_spaces_match_the_published_delta(self):
        state = DecoderSuffixState("Hello")
        update = state.reconcile(",  world", is_last=True, holdback_units=1)
        self.assertEqual(update.delta, ", world")
        state.apply(update)
        self.assertEqual(state.emitted_text, "Hello" + update.delta)
        state.pending = " ,  again"
        self.assertEqual(state.flush(), ", again")
        self.assertEqual(state.emitted_text, "Hello, world, again")
        self.assertEqual(state.flush(), "")

    def test_revised_holdback_is_not_confirmed_by_normalized_agreement(self):
        for revised in ("Yes.", "yes?", "yesterday"):
            with self.subTest(revised=revised):
                state = DecoderSuffixState("", pending="yes.")
                state.apply(state.reconcile(revised, is_last=False, holdback_units=1))
                self.assertEqual(state.confirmed_pending, "")
                self.assertEqual(state.pending, revised)
                state.apply(state.reconcile(revised, is_last=False, holdback_units=1))
                self.assertEqual(state.confirmed_pending, revised)

    def test_prefix_keeps_whole_units_and_rechecks_the_token_budget(self):
        char_tokenizer = SimpleNamespace(
            encode=lambda text, **kwargs: list(text),
            decode=lambda tokens, **kwargs: "".join(tokens),
        )
        space_tokenizer = SimpleNamespace(
            encode=lambda text, **kwargs: re.findall(r" \S+|\S", text),
            decode=lambda tokens, **kwargs: "".join(tokens),
        )
        for source, budget, tokenizer, expected in (
            ("hello world", 7, char_tokenizer, "world"),
            ("hello world", 3, char_tokenizer, ""),
            ("earlier text caution", 1, space_tokenizer, ""),
            ("earlier text caution one two", 3, space_tokenizer, "two"),
            ("previous 你好世界", 2, space_tokenizer, "世界"),
        ):
            with self.subTest(source=source, budget=budget):
                state = DecoderSuffixState(source)
                self.assertEqual(state.bounded_prefix(tokenizer, budget), expected)
                self.assertEqual(state.emitted_text, source)


if __name__ == "__main__":
    unittest.main()
