"""Realtime window handoff, transcript publication, and failure recovery."""

# ruff: noqa: E402 -- CPU kernel stubs must precede runtime imports.

import asyncio
import base64
import json
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

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
)
from sglang.srt.multimodal.encoder_window import (
    EncoderWindowConfig,
    encoder_window_kwargs,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.utils import get_or_create_event_loop
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_override

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Backend:
    """One list of snapshots per request; tuples supply terminal reasons."""

    def __init__(self, scripts, *, blocked=False):
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

    async def generate_request(self, request, raw_request=None, **kwargs):
        self.requests.append(request)
        if self.blocked:
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                self.cleaned.set()
            return
        script = self.scripts.pop(0)
        for index, frame in enumerate(script):
            if isinstance(frame, tuple):
                text, reason = frame
            else:
                text = frame
                reason = {"type": "stop"} if index == len(script) - 1 else None
            yield {"text": text, "meta_info": {"finish_reason": reason}}


def _run(coro):
    return get_or_create_event_loop().run_until_complete(coro)


def _policy(threshold):
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


def _connection(scripts, *, window=False, streaming=False, threshold=0, blocked=False):
    get_context().override("realtime_asr_test", enable_asr_decoder_streaming=streaming)
    manager = _Backend(scripts, blocked=blocked)
    adapter = SimpleNamespace(
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
        encoder_window=_policy(threshold) if window else None,
    )
    connection.config.configured = True
    connection.config.input_sample_rate = adapter.model_sample_rate
    connection.config.language = "en"
    connection.config.sampling_params = {"max_new_tokens": 256}
    return manager, connection


def _step(connection):
    collect = AsyncMock()
    _run(
        connection.transcription_processor.process(
            connection.transcription_state,
            is_last=False,
            sampling_params=connection.config.sampling_params,
            on_transcript_delta=collect,
        )
    )
    return "".join(call.args[0] for call in collect.await_args_list)


def _append(connection, pcm):
    return _run(
        connection._on_input_audio_buffer_append(
            SimpleNamespace(audio=base64.b64encode(pcm).decode())
        )
    )


def _activate(connection, pending="four five six"):
    connection.transcription_state.emitted_text = "one two three"
    connection.transcription_state.mode_state = WindowedState(
        suffix=TranscriptionSuffixState(pending=pending)
    )


def _events(connection, suffix):
    events = [
        json.loads(call.args[0])
        for call in connection.websocket.send_text.call_args_list
    ]
    return [event for event in events if event["type"].endswith(suffix)]


def _transcript(connection):
    return (
        "".join(event["delta"] for event in _events(connection, ".delta")),
        [event["transcript"] for event in _events(connection, ".completed")],
    )


class TestRealtimeASR(CustomTestCase):
    def setUp(self):
        super().setUp()
        enter_override(
            self,
            get_context().override_server_args(
                asr_max_buffer_seconds=120,
                enable_asr_decoder_streaming=False,
                incremental_streaming_output=False,
            ),
        )

    def test_final_failure_never_emits_completed(self):
        _, connection = _connection([[""], [""]], window=True)
        _activate(connection)
        connection.transcription_state.audio.append_pcm(bytes(4))
        _step(connection)
        connection.transcription_state.audio.append_pcm(bytes(2))
        with self.assertLogs(level="ERROR"):
            _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))
        self.assertEqual(len(_events(connection, ".failed")), 1)
        self.assertFalse(_events(connection, ".completed"))
        self.assertFalse(connection.transcription_state.has_audio)

    def test_handoff_then_rolling_requests_preserve_context_and_compact_pcm(self):
        manager, connection = _connection(
            [
                ["one two three four"],
                ["four five six"],
                ["four five six seven"],
                ["six seven eight nine"],
            ],
            window=True,
            threshold=60,
        )
        state = connection.transcription_state
        state.audio.append_pcm(np.arange(58, dtype=np.int16).tobytes())
        state.audio.last_attempted_offset_bytes = 116
        state.audio.last_processed_offset_bytes = 116

        # Body starts exclude the two samples supplied as leading STFT context.
        for end, start in ((60, 0), (62, 0), (64, 0), (66, 8)):
            with self.subTest(end=end):
                state.audio.append_pcm(
                    np.arange(end - 2, end, dtype=np.int16).tobytes()
                )
                _step(connection)
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
                    encoder_window_kwargs(leading_context_samples=context)
                    if end > 60
                    else None
                )
                self.assertEqual(request.mm_processor_kwargs, expected_kwargs)

        self.assertEqual(manager.requests[1].text, "PROMPT:one two three")

    def test_stable_holdback_survives_silence_and_commit(self):
        spoken = "yes."
        # Two agreeing candidates confirm the holdback; later decodes are silent.
        manager, connection = _connection(
            [[spoken], [spoken]] + [[""]] * 39, window=True
        )
        for _ in range(40):
            self.assertFalse(_append(connection, bytes(4)))
        state = connection.transcription_state
        self.assertGreaterEqual(state.audio.last_processed_offset_bytes, 156)
        self.assertLessEqual(len(state.audio.data), 144)
        self.assertEqual(state.mode_state.suffix.confirmed_pending, spoken)
        self.assertEqual(manager.requests[-1].text, "PROMPT:" + spoken)
        self.assertFalse(_events(connection, ".delta"))

        # Commit still decodes a real sub-chunk tail and publishes the holdback.
        self.assertFalse(_append(connection, bytes(2)))
        requests_before_commit = len(manager.requests)
        _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))
        self.assertEqual(len(manager.requests), requests_before_commit + 1)
        self.assertEqual(_transcript(connection), (spoken, [spoken]))

    def test_publication_uses_one_snapshot_through_preview_final_and_flush(self):
        scripts = [
            ["one two three four"],
            ["one two three four"],
            [" five six seven", " five six seven eight"],
            [" five six seven eight nine", " five six seven eight nine ten"],
        ]
        manager, connection = _connection(scripts, window=True, streaming=True)
        for _ in scripts:
            self.assertFalse(_append(connection, bytes(4)))
            self.assertEqual(
                connection.transcription_state.emitted_text, _transcript(connection)[0]
            )
        self.assertEqual(manager.requests[-1].text, "PROMPT:two three four")
        _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))
        self.assertEqual(len(manager.requests), len(scripts))
        expected = "one two three four five six seven eight nine ten"
        self.assertEqual(_transcript(connection), (expected, [expected]))

        # Cumulative CJK previews and their final append retain mixed-script
        # boundaries through the same publication and flush path.
        expected = "现在使用API接口继续输出"
        _, connection = _connection([["现在使用API接口", expected]], streaming=True)
        self.assertFalse(_append(connection, bytes(4)))
        self.assertEqual(_events(connection, ".delta")[0]["delta"], "现在使用API")
        _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))
        self.assertEqual(_transcript(connection), (expected, [expected]))

    def test_failed_send_records_only_successfully_published_text(self):
        _, connection = _connection(
            [["one two three", "one two three four"]], streaming=True
        )
        mode_before = deepcopy(connection.transcription_state.mode_state)
        delivered = []

        async def send(text):
            event = json.loads(text)
            if event["type"].endswith(".delta"):
                if delivered:
                    raise RuntimeError("peer disconnected during publication")
                delivered.append(event["delta"])

        connection.websocket.send_text.side_effect = send
        with self.assertLogs(level="ERROR"):
            self.assertTrue(_append(connection, bytes(4)))
        state = connection.transcription_state
        self.assertEqual(state.emitted_text, "one")
        self.assertEqual(state.emitted_text, "".join(delivered))
        self.assertEqual(state.mode_state, mode_before)
        self.assertEqual(state.audio.last_processed_offset_bytes, 0)
        connection.websocket.close.assert_awaited_once_with(code=1011)

    def test_item_audio_limit_still_applies_after_window_fallback(self):
        with get_context().override_server_args(asr_max_buffer_seconds=6):
            for window, scripts, sizes in (
                (False, [["one two"]], [6]),
                (True, [[], [], ["one two"]], [2, 4, 6]),
            ):
                with self.subTest(window=window):
                    manager, connection = _connection(scripts, window=window)
                    # Ordinary cumulative mode consumes the append once. With
                    # windowing, two failed handoffs precede cumulative fallback.
                    self.assertFalse(_append(connection, bytes(12)))
                    self.assertEqual(
                        [len(r.audio_data) for r in manager.requests], sizes
                    )
                    self.assertIsNone(manager.requests[-1].mm_processor_kwargs)
                    self.assertTrue(_append(connection, bytes(2)))
                    connection.websocket.close.assert_awaited_once_with(code=1009)
                    self.assertEqual(
                        connection.transcription_state.audio.received_bytes, 12
                    )

    def test_truncation_retains_audio_and_recovery_preserves_repeated_speech(self):
        manager, connection = _connection(
            [
                [("truncated words", {"type": "length"})],
                ["one two three four", "one two three four five six"],
            ],
            window=True,
            streaming=True,
        )
        _activate(connection, pending="one two three four five")
        state = connection.transcription_state
        original_pcm = np.array([1, 2], dtype=np.int16).tobytes()
        state.audio.append_pcm(original_pcm)
        with self.assertLogs(level="WARNING"):
            self.assertEqual(_step(connection), "")

        self.assertEqual(state.audio.last_attempted_offset_bytes, 4)
        self.assertEqual(state.audio.last_processed_offset_bytes, 0)
        self.assertEqual(bytes(state.audio.data), original_pcm)
        self.assertEqual(state.emitted_text, "one two three")
        self.assertEqual(state.mode_state.suffix.pending, "one two three four five")

        state.audio.append_pcm(np.array([3, 4], dtype=np.int16).tobytes())
        delta = _step(connection)
        np.testing.assert_array_equal(
            manager.requests[-1].audio_data,
            np.array([1, 2, 3, 4], dtype=np.float32) / 32768.0,
        )
        self.assertEqual(state.audio.last_processed_offset_bytes, 8)
        self.assertEqual(state.mode_state.consecutive_failures, 0)
        self.assertEqual(manager.requests[-1].text, "PROMPT:one two three")
        self.assertEqual(state.emitted_text, "one two three one two three four")
        self.assertEqual(delta, " one two three four")

    def test_connection_exit_waits_for_backend_cleanup_and_releases_session(self):
        async def run():
            manager, connection = _connection([], blocked=True)
            incoming = asyncio.Queue()
            connection.websocket.receive.side_effect = incoming.get
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
            with patch(
                "sglang.srt.entrypoints.openai.realtime.handler.RealtimeConnection",
                return_value=connection,
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
                    incoming.put_nowait({"type": "websocket.disconnect"})
                    await asyncio.wait_for(task, timeout=2)
                finally:
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            self.assertTrue(manager.cleaned.is_set())
            self.assertFalse(semaphore.locked())

        _run(run())


if __name__ == "__main__":
    unittest.main()
