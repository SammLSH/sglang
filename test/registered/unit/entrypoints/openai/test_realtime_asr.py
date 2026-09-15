# ruff: noqa: E402 -- CPU kernel stubs must precede runtime imports.

import asyncio
import base64
import json
import unittest
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from typing import AsyncIterator, Awaitable, Optional, TypeVar
from unittest.mock import AsyncMock, Mock, patch

from fastapi import Request
from pydantic import JsonValue
from starlette.types import Message

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import numpy as np

from sglang.srt.entrypoints.openai.realtime.audio_transcription.transcription_state import (
    WindowedState,
)
from sglang.srt.entrypoints.openai.realtime.audio_transcription.transcription_suffix import (
    TranscriptionSuffixState,
)
from sglang.srt.entrypoints.openai.realtime.audio_transcription.windowed_transcription import (
    ResolvedEncoderWindowPolicy,
)
from sglang.srt.entrypoints.openai.realtime.handler import handle_realtime_transcription
from sglang.srt.entrypoints.openai.realtime.session import RealtimeConnection
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    RealtimeEncoderWindowPolicy,
    TranscriptionAdapter,
)
from sglang.srt.entrypoints.openai.transcription_adapters.qwen3_asr import (
    Qwen3ASRAdapter,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.multimodal.encoder_window import (
    EncoderWindowConfig,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.utils import get_or_create_event_loop
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_override

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

Script = list[str | tuple[str, dict[str, JsonValue]]]
Result = TypeVar("Result")


class ScriptedBackend:
    """One list of snapshots per request; tuples supply terminal reasons."""

    def __init__(self, scripts: list[Script], *, blocked: bool = False) -> None:
        self.scripts = list(scripts)
        self.requests = []
        self.served_model_name = "qwen3-asr"
        self.tokenizer = SimpleNamespace(
            encode=lambda text, **kwargs: text.split(),
            decode=lambda tokens, **kwargs: " ".join(tokens),
        )
        self.blocked = blocked
        self.started = asyncio.Event()
        self.cleaned = asyncio.Event()

    async def generate_request(
        self,
        request: GenerateReqInput,
        raw_request: Optional[Request] = None,
        **kwargs: object,
    ) -> AsyncIterator[dict[str, JsonValue]]:
        self.requests.append(request)
        if self.blocked:
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                self.cleaned.set()
            return
        else:
            script = self.scripts.pop(0)
        for index, frame in enumerate(script):
            if isinstance(frame, tuple):
                text, reason = frame
            else:
                text = frame
                reason = {"type": "stop"} if index == len(script) - 1 else None
            yield {"text": text, "meta_info": {"finish_reason": reason}}


def run_async(coro: Awaitable[Result]) -> Result:
    return get_or_create_event_loop().run_until_complete(coro)


def window_policy(threshold: float) -> ResolvedEncoderWindowPolicy:
    # At 1 Hz, an 8-s encoder window is eight samples / sixteen PCM16 bytes.
    return ResolvedEncoderWindowPolicy(
        config=EncoderWindowConfig(
            sample_rate=1,
            window_samples=8,
            window_tokens=8,
            leading_context_samples=2,
        ),
        policy=RealtimeEncoderWindowPolicy(
            min_audio_sec=threshold,
            max_audio_context_windows=6,
            decoder_prefix_max_tokens=3,
            decoder_prefix_holdback_units=1,
        ),
    )


def make_connection(
    scripts: list[Script],
    *,
    window: bool = False,
    streaming: bool = False,
    threshold: float = 0,
    blocked: bool = False,
    adapter: Optional[TranscriptionAdapter] = None,
) -> tuple[ScriptedBackend, RealtimeConnection]:
    get_context().override("realtime_asr_test", enable_asr_decoder_streaming=streaming)
    manager = ScriptedBackend(scripts, blocked=blocked)
    adapter = adapter or SimpleNamespace(
        prompt_template="PROMPT:",
        model_sample_rate=1,
        supports_chunked_streaming=True,
        postprocess_text=lambda text: text,
        postprocess_streaming_text=lambda text, continuation=False: text,
        chunked_streaming_config={
            "chunk_size_sec": 2.0,
            "unfixed_chunk_num": 2,
            "unfixed_token_num": 1,
        },
    )
    connection = RealtimeConnection(
        AsyncMock(),
        manager,
        adapter,
        server_args=Mock(),
        encoder_window=window_policy(threshold) if window else None,
    )
    connection.config.configured = True
    connection.config.input_sample_rate = adapter.model_sample_rate
    connection.config.language = "en"
    connection.config.sampling_params = {"max_new_tokens": 256}
    return manager, connection


def run_step(connection: RealtimeConnection) -> str:
    collect = AsyncMock()
    run_async(
        connection.transcription_processor.process(
            connection.transcription_state,
            is_last=False,
            sampling_params=connection.config.sampling_params,
            on_transcript_delta=collect,
        )
    )
    return "".join(call.args[0] for call in collect.await_args_list)


def append_audio(connection: RealtimeConnection, pcm: bytes) -> bool:
    return run_async(
        connection._on_input_audio_buffer_append(
            SimpleNamespace(audio=base64.b64encode(pcm).decode())
        )
    )


def activate_window(
    connection: RealtimeConnection, pending: str = " four five six"
) -> None:
    connection.transcription_state.emitted_text = "one two three"
    connection.transcription_state.mode_state = WindowedState(
        suffix=TranscriptionSuffixState(pending=pending)
    )


def events_of_type(
    connection: RealtimeConnection, suffix: str
) -> list[dict[str, JsonValue]]:
    events = [
        json.loads(call.args[0])
        for call in connection.websocket.send_text.call_args_list
    ]
    return [event for event in events if event["type"].endswith(suffix)]


def published_transcript(connection: RealtimeConnection) -> tuple[str, list[str]]:
    return (
        "".join(event["delta"] for event in events_of_type(connection, ".delta")),
        [event["transcript"] for event in events_of_type(connection, ".completed")],
    )


class TestRealtimeASR(CustomTestCase):
    def setUp(self) -> None:
        super().setUp()
        enter_override(
            self,
            get_context().override_server_args(
                asr_max_buffer_seconds=120,
                enable_asr_decoder_streaming=False,
                incremental_streaming_output=False,
            ),
        )

    def test_final_failure_never_emits_completed(self) -> None:
        _, connection = make_connection([[""], [""]], window=True)
        activate_window(connection)
        connection.transcription_state.audio.append_pcm(bytes(4))
        run_step(connection)
        connection.transcription_state.audio.append_pcm(bytes(2))
        with self.assertLogs(level="ERROR"):
            run_async(connection._on_input_audio_buffer_commit(SimpleNamespace()))
        self.assertEqual(len(events_of_type(connection, ".failed")), 1)
        self.assertFalse(events_of_type(connection, ".completed"))
        self.assertFalse(connection.transcription_state.has_audio)

    def test_handoff_then_rolling_requests_preserve_context_and_compact_pcm(
        self,
    ) -> None:
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                self.check_handoff_and_rolling(streaming)

    def check_handoff_and_rolling(self, streaming: bool) -> None:
        manager, connection = make_connection(
            [
                ["one two three four"],
                [" four five six"],
                [" four five six seven"],
                [" six seven eight nine"],
                [" seven eight nine ten"],
            ],
            window=True,
            streaming=streaming,
            threshold=60,
        )
        state = connection.transcription_state
        state.audio.append_pcm(np.arange(58, dtype=np.int16).tobytes())
        state.audio.last_attempted_offset_bytes = 116
        state.audio.last_processed_offset_bytes = 116

        # Body starts exclude the two samples supplied as leading STFT context.
        for end, start in ((60, 0), (62, 0), (64, 0), (66, 8)):
            with self.subTest(end=end):
                self.assertFalse(
                    append_audio(
                        connection, np.arange(end - 2, end, dtype=np.int16).tobytes()
                    )
                )
                request = manager.requests[-1]
                context = 2 if start else 0
                np.testing.assert_array_equal(
                    request.audio_data,
                    np.arange(start - context, end, dtype=np.float32) / 32768.0,
                )
                # The handoff's provisional suffix needs another decode before
                # its audio is confirmed and the window can advance.
                self.assertEqual(
                    state.audio.last_processed_offset_bytes,
                    (60 if end == 62 else end) * 2,
                )
                self.assertEqual(state.audio.base_offset_bytes, (start - context) * 2)
                self.assertEqual(state.encoder_window_active, end > 60)
                expected_kwargs = (
                    {"encoder_window": {"leading_context_samples": context}}
                    if end > 60
                    else None
                )
                self.assertEqual(request.mm_processor_kwargs, expected_kwargs)

        self.assertEqual(manager.requests[1].text, "PROMPT:one two three")
        self.assertFalse(
            append_audio(connection, np.array([66], dtype=np.int16).tobytes())
        )
        run_async(connection._on_input_audio_buffer_commit(SimpleNamespace()))
        expected = "one two three four five six seven eight nine ten"
        self.assertEqual(published_transcript(connection), (expected, [expected]))
        self.assertEqual(len(manager.requests), 5)
        self.assertFalse(connection.transcription_state.has_audio)

    def test_stable_holdback_survives_silence_and_commit(self) -> None:
        spoken = "yes."
        # Two agreeing candidates confirm the holdback; later decodes are silent.
        manager, connection = make_connection(
            [[spoken], [spoken]] + [[""]] * 39, window=True
        )
        for _ in range(40):
            self.assertFalse(append_audio(connection, bytes(4)))
        state = connection.transcription_state
        self.assertGreaterEqual(state.audio.last_processed_offset_bytes, 156)
        self.assertLessEqual(len(state.audio.data), 144)
        self.assertEqual(state.mode_state.suffix.confirmed_pending, spoken)
        self.assertEqual(manager.requests[-1].text, "PROMPT:" + spoken)
        self.assertFalse(events_of_type(connection, ".delta"))

        # Commit still decodes a real sub-chunk tail and publishes the holdback.
        self.assertFalse(append_audio(connection, bytes(2)))
        requests_before_commit = len(manager.requests)
        run_async(connection._on_input_audio_buffer_commit(SimpleNamespace()))
        self.assertEqual(len(manager.requests), requests_before_commit + 1)
        self.assertEqual(published_transcript(connection), (spoken, [spoken]))

    def test_publication_uses_one_snapshot_through_preview_final_and_flush(
        self,
    ) -> None:
        scripts = [
            ["one two three four"],
            ["one two three four"],
            [" five six seven", " five six seven eight"],
            [" five six seven eight nine", " five six seven eight nine ten"],
        ]
        manager, connection = make_connection(scripts, window=True, streaming=True)
        for _ in scripts:
            self.assertFalse(append_audio(connection, bytes(4)))
            self.assertEqual(
                connection.transcription_state.emitted_text,
                published_transcript(connection)[0],
            )
        self.assertEqual(manager.requests[-1].text, "PROMPT:two three four")
        run_async(connection._on_input_audio_buffer_commit(SimpleNamespace()))
        self.assertEqual(len(manager.requests), len(scripts))
        expected = "one two three four five six seven eight nine ten"
        self.assertEqual(published_transcript(connection), (expected, [expected]))

        # Cumulative CJK previews and their final append retain mixed-script
        # boundaries through the same publication and flush path.
        expected = "现在使用API接口继续输出"
        _, connection = make_connection([["现在使用API接口", expected]], streaming=True)
        self.assertFalse(append_audio(connection, bytes(4)))
        self.assertEqual(
            events_of_type(connection, ".delta")[0]["delta"], "现在使用API"
        )
        run_async(connection._on_input_audio_buffer_commit(SimpleNamespace()))
        self.assertEqual(published_transcript(connection), (expected, [expected]))

        scripts = [["你好API接口"], ["你好API接口"], ["继续输出"], ["继续输出"]]
        manager, connection = make_connection(scripts, window=True, streaming=True)
        for _ in scripts:
            self.assertFalse(append_audio(connection, bytes(4)))
        self.assertEqual(manager.requests[-1].text, "PROMPT:你好API接口")
        run_async(connection._on_input_audio_buffer_commit(SimpleNamespace()))
        expected = "你好API接口继续输出"
        self.assertEqual(published_transcript(connection), (expected, [expected]))

    def test_failed_send_records_only_successfully_published_text(self) -> None:
        _, connection = make_connection(
            [["one two three", "one two three four"]], streaming=True
        )
        mode_before = deepcopy(connection.transcription_state.mode_state)
        delivered = []

        async def _send(text: str) -> None:
            event = json.loads(text)
            if event["type"].endswith(".delta"):
                if delivered:
                    raise RuntimeError("peer disconnected during publication")
                else:
                    delivered.append(event["delta"])
            else:
                return

        connection.websocket.send_text.side_effect = _send
        with self.assertLogs(level="ERROR"):
            self.assertTrue(append_audio(connection, bytes(4)))
        state = connection.transcription_state
        self.assertEqual(state.emitted_text, "one")
        self.assertEqual(state.emitted_text, "".join(delivered))
        self.assertEqual(state.mode_state, mode_before)
        self.assertEqual(state.audio.last_processed_offset_bytes, 0)
        connection.websocket.close.assert_awaited_once_with(code=1011)

    def test_qwen_prefix_fragments_publish_before_decode_finishes(self) -> None:
        async def _run(streaming: bool, incremental: bool) -> None:
            manager, connection = make_connection(
                [], streaming=streaming, adapter=Qwen3ASRAdapter()
            )
            get_context().override(
                "realtime_asr_test", incremental_streaming_output=incremental
            )
            ready, finish, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()
            fragments = (
                "lang",
                "uage English<as",
                "r_text>one two three four five six seven",
                " eight",
            )

            async def _generate(
                request: GenerateReqInput,
                raw_request: Optional[Request] = None,
                **kwargs: object,
            ) -> AsyncIterator[dict[str, JsonValue]]:
                text = ""
                try:
                    for fragment in fragments[:-1]:
                        text += fragment
                        if request.stream:
                            yield {
                                "text": fragment if incremental else text,
                                "meta_info": {"finish_reason": None},
                            }
                        else:
                            continue
                    ready.set()
                    await finish.wait()
                    yield {
                        "text": fragments[-1]
                        if request.stream and incremental
                        else text + fragments[-1],
                        "meta_info": {"finish_reason": {"type": "stop"}},
                    }
                finally:
                    cleaned.set()

            manager.generate_request = _generate
            task = asyncio.create_task(
                connection._on_input_audio_buffer_append(
                    SimpleNamespace(
                        audio=base64.b64encode(
                            bytes(connection.transcription_processor.chunk_size_bytes)
                        ).decode()
                    )
                )
            )
            try:
                await asyncio.wait_for(ready.wait(), 2)
                self.assertFalse(task.done())
                self.assertEqual(
                    published_transcript(connection)[0], "one" if streaming else ""
                )
                finish.set()
                self.assertFalse(await asyncio.wait_for(task, 2))
                await connection._on_input_audio_buffer_commit(SimpleNamespace())
                expected = "one two three four five six seven eight"
                self.assertEqual(
                    published_transcript(connection), (expected, [expected])
                )
                self.assertTrue(cleaned.is_set())
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        for streaming in (False, True):
            for incremental in (False, True):
                with self.subTest(streaming=streaming, incremental=incremental):
                    run_async(_run(streaming, incremental))

    def test_item_audio_limit_still_applies_after_window_fallback(self) -> None:
        with get_context().override_server_args(asr_max_buffer_seconds=6):
            for window, scripts, sizes in (
                (False, [["one two"]], [6]),
                (True, [[], [], ["one two"]], [2, 4, 6]),
            ):
                with self.subTest(window=window):
                    manager, connection = make_connection(scripts, window=window)
                    # Ordinary cumulative mode consumes the append once. With
                    # windowing, two failed handoffs precede cumulative fallback.
                    self.assertFalse(append_audio(connection, bytes(12)))
                    self.assertEqual(
                        [len(r.audio_data) for r in manager.requests], sizes
                    )
                    self.assertIsNone(manager.requests[-1].mm_processor_kwargs)
                    self.assertTrue(append_audio(connection, bytes(2)))
                    connection.websocket.close.assert_awaited_once_with(code=1009)
                    self.assertEqual(
                        connection.transcription_state.audio.received_bytes, 12
                    )

    def test_truncation_retains_audio_and_recovery_preserves_repeated_speech(
        self,
    ) -> None:
        manager, connection = make_connection(
            [
                [("truncated words", {"type": "length"})],
                [" one two three four", " one two three four five six"],
            ],
            window=True,
            streaming=True,
        )
        activate_window(connection, pending=" one two three four five")
        state = connection.transcription_state
        original_pcm = np.array([1, 2], dtype=np.int16).tobytes()
        state.audio.append_pcm(original_pcm)
        with self.assertLogs(level="WARNING"):
            self.assertEqual(run_step(connection), "")

        self.assertEqual(state.audio.last_attempted_offset_bytes, 4)
        self.assertEqual(state.audio.last_processed_offset_bytes, 0)
        self.assertEqual(bytes(state.audio.data), original_pcm)
        self.assertEqual(state.emitted_text, "one two three")
        self.assertEqual(state.mode_state.suffix.pending, " one two three four five")

        state.audio.append_pcm(np.array([3, 4], dtype=np.int16).tobytes())
        delta = run_step(connection)
        np.testing.assert_array_equal(
            manager.requests[-1].audio_data,
            np.array([1, 2, 3, 4], dtype=np.float32) / 32768.0,
        )
        self.assertEqual(state.audio.last_processed_offset_bytes, 8)
        self.assertEqual(state.mode_state.consecutive_failures, 0)
        self.assertEqual(manager.requests[-1].text, "PROMPT:one two three")
        self.assertEqual(state.emitted_text, "one two three one two three four")
        self.assertEqual(delta, " one two three four")

        # Once a preview is published, a truncated response cannot be retried.
        _, connection = make_connection(
            [
                [
                    " four five six seven",
                    (" four five six seven eight", {"type": "length"}),
                ]
            ],
            window=True,
            streaming=True,
        )
        activate_window(connection)
        state = connection.transcription_state
        accepted = state.mode_state
        state.audio.append_pcm(original_pcm)
        with self.assertRaisesRegex(RuntimeError, "max_new_tokens"):
            run_step(connection)
        self.assertEqual(state.emitted_text, "one two three four five")
        self.assertIs(state.mode_state, accepted)
        self.assertEqual(state.audio.last_processed_offset_bytes, 0)
        self.assertEqual(bytes(state.audio.data), original_pcm)
        self.assertFalse(events_of_type(connection, ".completed"))

    def test_connection_exit_waits_for_backend_cleanup_and_releases_session(
        self,
    ) -> None:
        async def _run(exit_kind: str) -> None:
            manager, connection = make_connection([], blocked=True)
            incoming = asyncio.Queue()

            async def _receive() -> Message:
                message = await incoming.get()
                if isinstance(message, RuntimeError):
                    raise message
                else:
                    return message

            connection.websocket.receive.side_effect = _receive
            incoming.put_nowait(
                {
                    "type": "websocket.receive",
                    "text": json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(bytes(4)).decode(),
                        }
                    ),
                }
            )
            semaphore = asyncio.Semaphore(1)
            with (
                patch(
                    "sglang.srt.entrypoints.openai.realtime.handler.RealtimeConnection",
                    return_value=connection,
                ),
                patch(
                    "sglang.srt.entrypoints.openai.realtime.session.MAX_PENDING_INPUT_BYTES",
                    1024,
                ),
            ):
                task = asyncio.create_task(
                    handle_realtime_transcription(
                        connection.websocket,
                        manager,
                        connection.adapter,
                        connection.server_args,
                        semaphore,
                    )
                )
                try:
                    await asyncio.wait_for(manager.started.wait(), timeout=2)
                    self.assertTrue(semaphore.locked())
                    if exit_kind == "overflow":
                        incoming.put_nowait(
                            {"type": "websocket.receive", "text": "x" * 1025}
                        )
                    elif exit_kind == "runtime_error":
                        incoming.put_nowait(RuntimeError("ASGI receive failed"))
                    else:
                        incoming.put_nowait({"type": "websocket.disconnect"})
                    await asyncio.wait_for(task, timeout=2)
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            self.assertTrue(manager.cleaned.is_set())
            self.assertFalse(semaphore.locked())
            self.assertEqual(connection.pending_input_bytes, 0)
            self.assertTrue(connection.incoming_messages.empty())
            self.assertFalse(events_of_type(connection, ".completed"))
            errors = events_of_type(connection, "error")
            if exit_kind == "overflow":
                self.assertEqual(errors[-1]["error"]["code"], "buffer_overflow")
                connection.websocket.close.assert_any_await(code=1009)
            elif exit_kind == "runtime_error":
                self.assertEqual(errors[-1]["error"]["code"], "inference_failed")
            else:
                self.assertEqual(errors, [])

        for exit_kind in ("disconnect", "overflow", "runtime_error"):
            with (
                self.subTest(exit_kind=exit_kind),
                (
                    self.assertLogs(level="ERROR")
                    if exit_kind == "runtime_error"
                    else nullcontext()
                ),
            ):
                run_async(_run(exit_kind))


if __name__ == "__main__":
    unittest.main()
