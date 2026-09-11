"""Focused transcript publication and backend iterator regressions."""

# ruff: noqa: E402 -- CPU kernel stubs must precede runtime imports.

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.openai.streaming_asr import (
    ASRBackendAborted,
    StreamingASRState,
    apply_cumulative_transcript,
    generate_asr_transcript,
    join_text,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _publish(state, delta):
    state.emitted_text = join_text(state.emitted_text, delta)
    return delta


class TestStreamingASRState(CustomTestCase):
    def test_holdback_and_revision_without_prefix(self):
        state = StreamingASRState(2.0, 2, 1)
        self.assertEqual(_publish(state, state.update("one two three")), "one two")
        self.assertEqual(_publish(state, state.update("one two three four")), "three")
        self.assertEqual(state.get_prefix_text(), "one two three")
        self.assertEqual(state.finalize(), "four")
        state = StreamingASRState(2.0, 3, 1)
        self.assertEqual(
            [
                _publish(state, state.update(text))
                for text in ("a tail", "b tail", "b tail")
            ],
            ["a", "b", ""],
        )
        self.assertEqual(state.finalize(), "tail")
        state = StreamingASRState(2.0, 2, 5)
        self.assertEqual(state.update("今天北京天气晴朗"), "")
        self.assertEqual(state.update("今天上海天气晴朗"), "")
        self.assertEqual(state.finalize(), "今天上海天气晴朗")

    def test_punctuation_remainder_does_not_repeat_the_published_word(self):
        state = StreamingASRState(2.0, 2, 1)
        self.assertEqual(_publish(state, state.update("Hello pending")), "Hello")
        delta = apply_cumulative_transcript(state, "Hello,  world", is_last=True)
        self.assertEqual(delta, ", world")
        self.assertEqual(state.emitted_text, "Hello")
        _publish(state, delta)
        self.assertEqual(state.emitted_text, "Hello" + delta)
        self.assertEqual(state.finalize(), "")

    def test_repetition_after_prefix_advances_and_an_empty_continuation(self):
        state = StreamingASRState(2.0, 2, 5)
        phrase = "one two three four five six"
        deltas, suffixes = [], []
        for count in range(1, 7):
            reference = " ".join([phrase] * min(count, 4))
            prefix = state.get_prefix_text()
            self.assertTrue(reference.startswith(prefix))
            suffix = "" if count == 5 else reference[len(prefix) :]
            suffixes.append(suffix)
            deltas.append(
                _publish(
                    state,
                    apply_cumulative_transcript(
                        state, prefix + suffix, is_last=count == 6
                    ),
                )
            )
        self.assertEqual(suffixes[2], suffixes[3])
        self.assertEqual(state.emitted_text, reference)
        self.assertEqual(" ".join(filter(None, deltas)), reference)


class TestASRBackendContract(CustomTestCase):
    def test_abort_normalization_and_iterator_cleanup(self):
        async def run():
            closed = []

            async def responses(request, raw_request):
                try:
                    if isinstance(frame, Exception):
                        raise frame
                    yield frame
                    self.fail("non-stream read continued after its response")
                finally:
                    closed.append(True)

            adapter = SimpleNamespace(
                prompt_template="PROMPT:",
                postprocess_text=lambda text: text,
                postprocess_streaming_text=lambda text, **kwargs: text,
            )
            manager = SimpleNamespace(generate_request=responses)
            frame = {"text": "hello", "meta_info": {"finish_reason": {"type": "stop"}}}
            result = await generate_asr_transcript(manager, adapter, bytes(4), {})
            self.assertEqual(
                (result.text, result.finish_reason, closed), ("hello", "stop", [True])
            )
            callback_error = HTTPException(503, "callback failed")
            abort = {
                "meta_info": {
                    "finish_reason": {
                        "type": "abort",
                        "status_code": 503,
                        "message": "backend failed",
                    }
                }
            }
            for frame, callback, expected in (
                (HTTPException(503, "backend failed"), None, ASRBackendAborted),
                (abort, AsyncMock(), ASRBackendAborted),
                (
                    {"text": "hello"},
                    AsyncMock(side_effect=callback_error),
                    HTTPException,
                ),
            ):
                closed.clear()
                with self.assertRaises(expected) as caught:
                    await generate_asr_transcript(
                        manager, adapter, bytes(4), {}, on_update=callback
                    )
                self.assertEqual(closed, [True])
                if expected is ASRBackendAborted:
                    self.assertTrue(caught.exception.retryable)
                else:
                    self.assertIs(caught.exception, callback_error)

        with patch(
            "sglang.srt.entrypoints.openai.streaming_asr.get_serving",
            return_value=SimpleNamespace(incremental_streaming_output=False),
        ):
            asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
