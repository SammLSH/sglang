"""Unit tests for windowed realtime ASR suffix reconciliation."""

import re
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.openai.realtime.decoder_suffix import (
    DecoderSuffixState,
    SuffixUpdate,
    trim_prefix_echo,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_WORD_TOKENIZER = SimpleNamespace(
    encode=lambda text, add_special_tokens=False: text.split(),
    decode=lambda token_ids, **kwargs: " ".join(token_ids),
)
_CHAR_TOKENIZER = SimpleNamespace(
    encode=lambda text, add_special_tokens=False: list(text),
    decode=lambda token_ids, **kwargs: "".join(token_ids),
)
# Like a BPE vocabulary, a leading space can make a word one token while
# the same word at the start of the string takes multiple tokens.
_SPACE_WORD_TOKENIZER = SimpleNamespace(
    encode=lambda text, add_special_tokens=False: re.findall(r" \S+|\S", text),
    decode=lambda token_ids, **kwargs: "".join(token_ids),
)


class TestDecoderSuffixState(CustomTestCase):
    def test_only_a_complete_prefix_replay_is_trimmed(self):
        self.assertEqual(
            trim_prefix_echo("one two three four", "one two"), "three four"
        )
        self.assertEqual(trim_prefix_echo("One, two three", "one two"), "three")
        self.assertEqual(trim_prefix_echo("two three", "one two"), "two three")
        self.assertEqual(trim_prefix_echo("one", "one two"), "one")
        self.assertEqual(trim_prefix_echo("one two", ""), "one two")
        self.assertEqual(trim_prefix_echo("你好世界今天", "你好"), "世界今天")
        self.assertEqual(trim_prefix_echo("one two", "one two"), "")

    def test_reconcile_publishes_agreed_units_and_holds_the_rest(self):
        # The first decode after a handoff is held entirely.
        state = DecoderSuffixState(emitted_text="one")
        update = state.reconcile("two three", is_last=False, holdback_units=1)
        self.assertEqual(update, SuffixUpdate(delta="", pending="two three"))

        # Local Agreement publishes the agreed units minus the holdback.
        state = DecoderSuffixState(emitted_text="one", pending="two three four")
        update = state.reconcile("two three five six", is_last=False, holdback_units=1)
        self.assertEqual((update.delta, update.pending), ("two", "three five six"))
        update = state.reconcile("Two, three four.", is_last=False, holdback_units=0)
        self.assertEqual((update.delta, update.pending), ("Two, three four.", ""))
        update = state.reconcile("nine ten", is_last=False, holdback_units=1)
        self.assertEqual(update, SuffixUpdate(delta="", pending="nine ten"))

        # No-space scripts agree per character.
        state = DecoderSuffixState(emitted_text="", pending="你好世")
        update = state.reconcile("你好世界今", is_last=False, holdback_units=1)
        self.assertEqual((update.delta, update.pending), ("你好", "世界今"))

        # An echoed decoder prefix is removed before agreement.
        state = DecoderSuffixState(emitted_text="one two", pending="three")
        update = state.reconcile(
            "one two three four",
            is_last=False,
            holdback_units=0,
            decoder_prefix="one two",
        )
        self.assertEqual((update.delta, update.pending), ("three", "four"))

        # The final decode supersedes pending text but never loses it.
        state = DecoderSuffixState(emitted_text="one", pending="two three")
        self.assertEqual(
            state.reconcile("two four", is_last=True, holdback_units=1),
            SuffixUpdate(delta="two four", pending=""),
        )
        self.assertEqual(
            state.reconcile("", is_last=True, holdback_units=1),
            SuffixUpdate(delta="two three", pending=""),
        )

        # An empty intermediate continuation keeps pending and says so.
        state = DecoderSuffixState(emitted_text="one", pending="two")
        self.assertEqual(
            state.reconcile("", is_last=False, holdback_units=1),
            SuffixUpdate(delta="", pending="two", empty_continuation=True),
        )

    def test_apply_and_flush(self):
        state = DecoderSuffixState(emitted_text="one")
        state.apply(SuffixUpdate(delta="two", pending="three"))
        self.assertEqual(state.emitted_text, "one two")
        self.assertEqual(state.latest_text, "one two three")
        self.assertEqual(state.flush(), "three")
        self.assertEqual(state.emitted_text, "one two three")
        self.assertEqual(state.pending, "")
        self.assertEqual(state.flush(), "")

        chinese = DecoderSuffixState(emitted_text="你好")
        chinese.apply(SuffixUpdate(delta="世界", pending="今"))
        self.assertEqual(chinese.latest_text, "你好世界今")

    def test_bounded_prefix_is_the_recent_tail_on_a_unit_boundary(self):
        state = DecoderSuffixState(emitted_text="one two three four")
        self.assertEqual(state.bounded_prefix(_WORD_TOKENIZER, 3), "two three four")
        self.assertEqual(
            state.bounded_prefix(_WORD_TOKENIZER, 10), "one two three four"
        )
        self.assertEqual(
            DecoderSuffixState(emitted_text="").bounded_prefix(_WORD_TOKENIZER, 3), ""
        )
        self.assertEqual(state.bounded_prefix(_WORD_TOKENIZER, 0), "")

        # Token-level slicing drops a partial leading unit.
        state = DecoderSuffixState(emitted_text="hello world")
        self.assertEqual(state.bounded_prefix(_CHAR_TOKENIZER, 7), "world")
        self.assertEqual(state.bounded_prefix(_CHAR_TOKENIZER, 3), "")
        self.assertEqual(state.bounded_prefix(_CHAR_TOKENIZER, 5), "world")

        # No-space scripts slice on characters.
        self.assertEqual(
            DecoderSuffixState(emitted_text="你好世界").bounded_prefix(
                _CHAR_TOKENIZER, 2
            ),
            "世界",
        )
        self.assertEqual(
            DecoderSuffixState(emitted_text="hello你好").bounded_prefix(
                _CHAR_TOKENIZER, 3
            ),
            "你好",
        )

    def test_bounded_prefix_rechecks_tokens_after_removing_leading_space(self):
        for source, budget, expected in (
            ("earlier text caution", 1, ""),
            ("earlier text caution one", 4, "one"),
            ("earlier text caution one two", 3, "two"),
            ("previous 你好", 1, "好"),
            ("previous 你好世界", 2, "世界"),
        ):
            with self.subTest(source=source, budget=budget):
                state = DecoderSuffixState(emitted_text=source)
                prefix = state.bounded_prefix(_SPACE_WORD_TOKENIZER, budget)
                self.assertEqual(prefix, expected)
                self.assertLessEqual(len(_SPACE_WORD_TOKENIZER.encode(prefix)), budget)
                self.assertEqual(state.emitted_text, source)


if __name__ == "__main__":
    unittest.main()
