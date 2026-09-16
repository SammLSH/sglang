import re
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.openai.realtime.audio_transcription.transcription_suffix import (
    TranscriptionSuffixState,
    TranscriptionSuffixUpdate,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def make_suffix_state(update: TranscriptionSuffixUpdate) -> TranscriptionSuffixState:
    return TranscriptionSuffixState(
        pending_text=update.pending_text,
        confirmed_pending_text_chars=update.confirmed_pending_text_chars,
    )


class TestTranscriptionSuffixState(CustomTestCase):
    def test_agreement_and_final_pending(self) -> None:
        for pending_text, candidate, delta, tail in (
            ("two three four", "two three five", "two", " three five"),
            ("你好API接口", "你好API结果", "你好", "API结果"),
        ):
            with self.subTest(candidate=candidate):
                state = TranscriptionSuffixState(pending_text=pending_text)
                update = state.prepare_transcript_update(
                    candidate, is_last=False, holdback_units=1
                )
                self.assertEqual(
                    update, TranscriptionSuffixUpdate(delta=delta, pending_text=tail)
                )
                self.assertEqual(update.delta + update.pending_text, candidate)
                self.assertEqual(state.pending_text, pending_text)
                state = make_suffix_state(update)
                update = state.prepare_transcript_update(
                    "", is_last=True, holdback_units=1
                )
                self.assertEqual(
                    update, TranscriptionSuffixUpdate(delta=tail, pending_text="")
                )
                self.assertEqual(
                    make_suffix_state(update).flush_pending_transcript(), ""
                )

        # A revised opening must not force every later request to decode it again.
        state = TranscriptionSuffixState(pending_text="Sibyl has ridden and driven.")
        candidate = "Sybil has ridden and driven past its medieval gateway many."
        update = state.prepare_transcript_update(
            candidate, is_last=False, holdback_units=1, unfixed_units=5
        )
        self.assertEqual(update.delta, "Sybil has ridden and driven")
        self.assertEqual(update.pending_text, " past its medieval gateway many.")
        self.assertEqual(update.delta + update.pending_text, candidate)
        self.assertEqual(state.pending_text, "Sibyl has ridden and driven.")

    def test_confirmed_holdback_preserves_word_extensions_and_repeated_speech(
        self,
    ) -> None:
        tokenizer = SimpleNamespace(
            encode=lambda text, **kwargs: list(text),
            decode=lambda tokens, **kwargs: "".join(tokens),
        )
        for continuation, expected in (("pet", "carpet"), (" pet", "car pet")):
            with self.subTest(continuation=continuation):
                state = TranscriptionSuffixState(pending_text="car")
                state = make_suffix_state(
                    state.prepare_transcript_update(
                        "car", is_last=False, holdback_units=1
                    )
                )
                self.assertEqual(state.confirmed_pending_text, "car")
                self.assertEqual(
                    state.build_decoder_prefix(tokenizer, 10, emitted_text=""), "car"
                )
                self.assertEqual(
                    state.build_decoder_prefix(tokenizer, 2, emitted_text=""), ""
                )
                state = make_suffix_state(
                    state.prepare_transcript_update(
                        continuation, is_last=False, holdback_units=1
                    )
                )
                self.assertEqual(state.confirmed_pending_text, "car")
                self.assertEqual(state.pending_text, expected)
                update = state.prepare_transcript_update(
                    continuation, is_last=False, holdback_units=1
                )
                state = make_suffix_state(update)
                self.assertEqual(
                    update.delta + state.flush_pending_transcript(), expected
                )
                self.assertEqual(state.confirmed_pending_text_chars, 0)
        state = TranscriptionSuffixState(pending_text="very very")
        update = state.prepare_transcript_update(
            "very very", is_last=False, holdback_units=1
        )
        self.assertEqual(update.delta, "very")
        state = make_suffix_state(update)
        self.assertEqual(state.flush_pending_transcript(), " very")

    def test_final_delta_and_flush_preserve_punctuation(self) -> None:
        update = TranscriptionSuffixState().prepare_transcript_update(
            ",  world", is_last=True, holdback_units=1
        )
        self.assertEqual(update.delta, ",  world")
        state = TranscriptionSuffixState(pending_text=" ,  again")
        self.assertEqual(state.flush_pending_transcript(), " ,  again")
        self.assertEqual(state.pending_text, "")
        self.assertEqual(state.flush_pending_transcript(), "")

    def test_revised_holdback_is_not_confirmed_by_normalized_agreement(self) -> None:
        for revised in ("Yes.", "yes?", "yesterday"):
            with self.subTest(revised=revised):
                state = TranscriptionSuffixState(pending_text="yes.")
                state = make_suffix_state(
                    state.prepare_transcript_update(
                        revised, is_last=False, holdback_units=1
                    )
                )
                self.assertEqual(state.confirmed_pending_text, "")
                self.assertEqual(state.pending_text, revised)
                state = make_suffix_state(
                    state.prepare_transcript_update(
                        revised, is_last=False, holdback_units=1
                    )
                )
                self.assertEqual(state.confirmed_pending_text, revised)

    def test_prefix_keeps_whole_units_and_rechecks_the_token_budget(self) -> None:
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
                    state.build_decoder_prefix(tokenizer, budget, emitted_text=source),
                    expected,
                )
        state = TranscriptionSuffixState(
            pending_text="API接口", confirmed_pending_text_chars=3
        )
        self.assertEqual(
            state.build_decoder_prefix(char_tokenizer, 20, emitted_text="你好"),
            "你好API",
        )
        for published, pending_text in (
            ("earlier " * 10_000, "last word"),
            ("earlier ", "confirmed " * 10_000 + "last word"),
        ):
            with self.subTest(confirmed_chars=len(pending_text)):
                state = TranscriptionSuffixState(
                    pending_text=pending_text,
                    confirmed_pending_text_chars=len(pending_text),
                )
                self.assertEqual(
                    state.build_decoder_prefix(
                        char_tokenizer, 9, emitted_text=published
                    ),
                    "last word",
                )


if __name__ == "__main__":
    unittest.main()
