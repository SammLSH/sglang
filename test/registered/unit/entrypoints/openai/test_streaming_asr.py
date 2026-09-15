# ruff: noqa: E402 -- CPU kernel stubs must precede runtime imports.

import asyncio
import unittest
from types import SimpleNamespace
from typing import AsyncIterator, Optional
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException, Request
from pydantic import JsonValue

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.openai.streaming_asr import (
    StreamingASRState,
    TranscriptionBackendAborted,
    apply_cumulative_transcript,
    generate_transcript,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCumulativeTranscriptState(CustomTestCase):
    def test_holdback_and_revision_without_prefix(self) -> None:
        state = StreamingASRState(2.0, 2, 1)
        emitted_text = ""
        delta = state.update("one two three", emitted_text=emitted_text)
        self.assertEqual(delta, "one two")
        emitted_text += delta
        delta = state.update("one two three four", emitted_text=emitted_text)
        self.assertEqual(delta, " three")
        emitted_text += delta
        self.assertEqual(
            state.get_prefix_text(emitted_text=emitted_text), "one two three"
        )
        self.assertEqual(state.finalize(emitted_text=emitted_text), " four")
        state = StreamingASRState(2.0, 3, 1)
        emitted_text, deltas = "", []
        for text in ("a tail", "b tail", "b tail"):
            delta = state.update(text, emitted_text=emitted_text)
            deltas.append(delta)
            emitted_text += delta
        self.assertEqual(deltas, ["a", " b", ""])
        self.assertEqual(state.finalize(emitted_text=emitted_text), " tail")
        for text in ("今天北京天气晴朗", "现在使用API接口继续输出"):
            with self.subTest(text=text):
                state = StreamingASRState(2.0, 2, 5)
                emitted_text = state.update(text, emitted_text="")
                self.assertTrue(emitted_text)
                extended = text + "然后继续"
                emitted_text += state.update(extended, emitted_text=emitted_text)
                self.assertEqual(
                    state.get_prefix_text(emitted_text=emitted_text), emitted_text
                )
                emitted_text += state.finalize(emitted_text=emitted_text)
                self.assertEqual(emitted_text, extended)

    def test_punctuation_remainder_does_not_repeat_the_published_word(self) -> None:
        state = StreamingASRState(2.0, 2, 1)
        delta = state.update("Hello pending", emitted_text="")
        self.assertEqual(delta, "Hello")
        emitted_text = delta
        delta = apply_cumulative_transcript(
            state, "Hello,  world", is_last=True, emitted_text=emitted_text
        )
        self.assertEqual(delta, ",  world")
        emitted_text += delta
        self.assertEqual(state.finalize(emitted_text=emitted_text), "")

        # Exact appends can extend the final published word without repeating it.
        state = StreamingASRState(2.0, 2, 1)
        state.full_transcript = "one twofold three"
        self.assertEqual(
            state.unpublished_text(emitted_text="one two"),
            "fold three",
        )

    def test_repetition_after_prefix_advances_and_an_empty_continuation(self) -> None:
        state = StreamingASRState(2.0, 2, 5)
        emitted_text = ""
        phrase = "one two three four five six"
        deltas, suffixes = [], []
        for count in range(1, 7):
            reference = " ".join([phrase] * min(count, 4))
            prefix = state.get_prefix_text(emitted_text=emitted_text)
            self.assertTrue(reference.startswith(prefix))
            suffix = "" if count == 5 else reference[len(prefix) :]
            suffixes.append(suffix)
            delta = apply_cumulative_transcript(
                state, prefix + suffix, is_last=count == 6, emitted_text=emitted_text
            )
            deltas.append(delta)
            emitted_text += delta
        self.assertEqual(suffixes[2], suffixes[3])
        self.assertEqual(emitted_text, reference)
        self.assertEqual("".join(deltas), reference)


class TestTranscriptionBackendContract(CustomTestCase):
    def test_abort_normalization_and_iterator_cleanup(self) -> None:
        async def _run() -> None:
            closed = []

            async def _responses(
                request: GenerateReqInput, raw_request: Optional[Request]
            ) -> AsyncIterator[dict[str, JsonValue]]:
                try:
                    if isinstance(frame, Exception):
                        raise frame
                    else:
                        yield frame
                    self.fail("non-stream read continued after its response")
                finally:
                    closed.append(True)

            adapter = SimpleNamespace(
                prompt_template="PROMPT:",
                postprocess_text=lambda text: text,
                postprocess_streaming_text=lambda text, **kwargs: text,
            )
            manager = SimpleNamespace(generate_request=_responses)
            frame = {"text": "hello", "meta_info": {"finish_reason": {"type": "stop"}}}
            result = await generate_transcript(manager, adapter, bytes(4), {})
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
                (
                    HTTPException(503, "backend failed"),
                    None,
                    TranscriptionBackendAborted,
                ),
                (abort, AsyncMock(), TranscriptionBackendAborted),
                (
                    {"text": "hello"},
                    AsyncMock(side_effect=callback_error),
                    HTTPException,
                ),
            ):
                closed.clear()
                with self.assertRaises(expected) as caught:
                    await generate_transcript(
                        manager, adapter, bytes(4), {}, on_update=callback
                    )
                self.assertEqual(closed, [True])
                if expected is TranscriptionBackendAborted:
                    self.assertTrue(caught.exception.retryable)
                else:
                    self.assertIs(caught.exception, callback_error)

        with patch(
            "sglang.srt.entrypoints.openai.streaming_asr.get_serving",
            return_value=SimpleNamespace(incremental_streaming_output=False),
        ):
            asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
