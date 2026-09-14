"""Suffix publication and bounded decoder-prefix regressions."""

import re
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.openai.realtime.audio_transcription.transcription_suffix import (
    SuffixUpdate,
    TranscriptionSuffixState,
)
from sglang.srt.entrypoints.openai.streaming_asr import join_text
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _pending_state(update):
    return TranscriptionSuffixState(
        pending=update.pending, confirmed_pending_chars=update.confirmed_pending_chars
    )


class TestTranscriptionSuffixState(CustomTestCase):
    def test_agreement_and_final_pending(self):
        state = TranscriptionSuffixState(pending="two three four")
        update = state.reconcile("two three five", is_last=False, holdback_units=1)
        self.assertEqual(update, SuffixUpdate(delta="two", pending="three five"))
        self.assertEqual(state.pending, "two three four")
        state = _pending_state(update)
        update = state.reconcile("", is_last=True, holdback_units=1)
        self.assertEqual(update, SuffixUpdate(delta="three five", pending=""))
        state = _pending_state(update)
        self.assertEqual(state.flush(), "")

    def test_confirmed_holdback_preserves_word_extensions_and_repeated_speech(self):
        tokenizer = SimpleNamespace(
            encode=lambda text, **kwargs: list(text),
            decode=lambda tokens, **kwargs: "".join(tokens),
        )
        for continuation, expected in (("pet", "carpet"), (" pet", "car pet")):
            with self.subTest(continuation=continuation):
                state = TranscriptionSuffixState(pending="car")
                state = _pending_state(
                    state.reconcile("car", is_last=False, holdback_units=1)
                )
                self.assertEqual(state.confirmed_pending, "car")
                self.assertEqual(
                    state.bounded_prefix(tokenizer, 10, emitted_text=""), "car"
                )
                self.assertEqual(
                    state.bounded_prefix(tokenizer, 2, emitted_text=""), ""
                )
                state = _pending_state(
                    state.reconcile(continuation, is_last=False, holdback_units=1)
                )
                self.assertEqual(state.confirmed_pending, "car")
                self.assertEqual(state.pending, expected)
                update = state.reconcile(continuation, is_last=False, holdback_units=1)
                state = _pending_state(update)
                self.assertEqual(join_text(update.delta, state.flush()), expected)
                self.assertEqual(state.confirmed_pending_chars, 0)
        state = TranscriptionSuffixState(pending="very very")
        update = state.reconcile("very very", is_last=False, holdback_units=1)
        self.assertEqual(update.delta, "very")
        state = _pending_state(update)
        self.assertEqual(state.flush(), "very")

    def test_final_delta_and_flush_preserve_punctuation(self):
        update = TranscriptionSuffixState().reconcile(
            ",  world", is_last=True, holdback_units=1
        )
        self.assertEqual(update.delta, ", world")
        state = TranscriptionSuffixState(pending=" ,  again")
        self.assertEqual(state.flush(), ", again")
        self.assertEqual(state.pending, "")
        self.assertEqual(state.flush(), "")

    def test_revised_holdback_is_not_confirmed_by_normalized_agreement(self):
        for revised in ("Yes.", "yes?", "yesterday"):
            with self.subTest(revised=revised):
                state = TranscriptionSuffixState(pending="yes.")
                state = _pending_state(
                    state.reconcile(revised, is_last=False, holdback_units=1)
                )
                self.assertEqual(state.confirmed_pending, "")
                self.assertEqual(state.pending, revised)
                state = _pending_state(
                    state.reconcile(revised, is_last=False, holdback_units=1)
                )
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
                state = TranscriptionSuffixState()
                self.assertEqual(
                    state.bounded_prefix(tokenizer, budget, emitted_text=source),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
