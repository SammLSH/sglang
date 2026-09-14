"""Core window-continuation, decoder-streaming, and disconnect contracts."""

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
from fastapi import WebSocket

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
from sglang.srt.entrypoints.openai.realtime.session import RealtimeASRSession
from sglang.srt.entrypoints.openai.realtime.transport import WebSocketRealtimeTransport
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    RealtimeEncoderWindowPolicy,
)
from sglang.srt.entrypoints.openai.transcription_adapters.qwen3_asr import (
    Qwen3ASRAdapter,
)
from sglang.srt.multimodal.encoder_window import (
    AudioEncoderWindowConfig,
    build_audio_window_processor_kwargs,
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
        config=AudioEncoderWindowConfig(
            sample_rate=1,
            hop_length=1,
            window_frames=8,
            window_samples=8,
            window_tokens=8,
            min_tail_samples=4,
            context_samples=2,
        ),
        policy=RealtimeEncoderWindowPolicy(
            min_audio_sec=threshold,
            max_audio_context_windows=6,
            decoder_prefix_max_tokens=3,
            decoder_prefix_holdback_units=1,
        ),
    )


def _connection(
    scripts, *, window=False, streaming=False, threshold=0, blocked=False, adapter=None
):
    get_context().override("realtime_asr_test", enable_asr_decoder_streaming=streaming)
    manager = _Backend(scripts, blocked=blocked)
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
    connection = RealtimeASRSession(
        Mock(receive=AsyncMock(), send=AsyncMock(), finish=AsyncMock()),
        manager,
        adapter,
        encoder_window=_policy(threshold) if window else None,
    )
    connection.config.configured = True
    connection.config.input_sample_rate = adapter.model_sample_rate
    connection.config.language = "en"
    connection.config.sampling_params = {"max_new_tokens": 256}
    return manager, connection


def _step(connection, *, is_last=False, published=None):
    deltas = []

    async def collect(text):
        deltas.append(text)
        if published is not None:
            published.append(text)

    _run(
        connection.transcription_processor.process(
            connection.transcription_state,
            is_last=is_last,
            sampling_params=connection.config.sampling_params,
            on_transcript_delta=collect,
        )
    )
    return "".join(deltas)


def _activate(connection, pending="four five six"):
    connection.transcription_state.emitted_text = "one two three"
    connection.transcription_state.mode_state = WindowedState(
        suffix=TranscriptionSuffixState(pending=pending)
    )


def _events(connection, suffix):
    return [
        call.args[0]
        for call in connection.transport.send.call_args_list
        if call.args[0].type.endswith(suffix)
    ]


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

    def test_windowing_requires_opt_in_threshold_and_nonfinal_update(self):
        for enabled, final, threshold, language, expected_window in (
            (False, False, 0, "en", False),
            (True, True, 0, "en", False),
            (True, False, 2, "en", False),
            (True, False, 0, "en", True),
            (True, False, 0, None, True),
            (True, False, 0, "zh", True),
        ):
            with self.subTest(
                enabled=enabled, language=language, final=final, threshold=threshold
            ):
                manager, connection = _connection(
                    [["one two"]], window=enabled, threshold=threshold
                )
                connection.config.language = language
                connection.transcription_state.audio.append_pcm(bytes(4))
                self.assertTrue(_run(connection._run_inference(is_last=final)))
                self.assertEqual(
                    connection.transcription_state.encoder_window_active,
                    expected_window,
                )
                self.assertEqual(
                    bool(manager.requests[0].mm_processor_kwargs), expected_window
                )

    def test_cumulative_truncation_keeps_text_prefix_and_audio_unchanged(self):
        for final in (False, True):
            with self.subTest(final=final):
                manager, connection = _connection(
                    [["one two three"], [("repeated " * 10, {"type": "length"})]]
                )
                state = connection.transcription_state
                state.mode_state.transcript.unfixed_chunk_num = 1
                state.audio.append_pcm(bytes(4))
                _step(connection)
                before = vars(state.mode_state.transcript).copy()
                state.audio.append_pcm(bytes(2))
                published = []
                with self.assertRaisesRegex(RuntimeError, "max_new_tokens"):
                    _step(connection, is_last=final, published=published)
                self.assertEqual(published, [])
                self.assertEqual(vars(state.mode_state.transcript), before)
                self.assertEqual(manager.requests[-1].text, "PROMPT:one two")
                self.assertFalse(manager.requests[-1].stream)
                self.assertEqual(state.audio.last_attempted_offset_bytes, 4)
                self.assertEqual(state.audio.last_processed_offset_bytes, 4)
                self.assertEqual(bytes(state.audio.data), bytes(6))

    def test_punctuation_handoff_and_real_tail_match_completed_text(self):
        manager, connection = _connection(
            [["Hello there"], ["Hello,  world today"], [", today!"]]
        )
        # Two complete chunks, then one real sample handled by commit.
        for size in (4, 4, 2):
            _run(
                connection._on_input_audio_buffer_append(
                    SimpleNamespace(audio=base64.b64encode(bytes(size)).decode())
                )
            )
        _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))
        text = "".join(e.delta for e in _events(connection, ".delta"))
        self.assertEqual(text, "Hello, world, today!")
        self.assertEqual(_events(connection, ".completed")[0].transcript, text)
        self.assertEqual(len(manager.requests[-1].audio_data), 5)
        self.assertTrue(all(not request.stream for request in manager.requests))

    def test_final_failure_never_emits_completed(self):
        for window, final_frame in (
            (False, ("repeated words", {"type": "length"})),
            (True, ""),
        ):
            with self.subTest(window=window):
                manager, connection = _connection([[""], [final_frame]], window=window)
                if window:
                    _activate(connection)
                connection.transcription_state.audio.append_pcm(bytes(4))
                _step(connection)
                connection.transcription_state.audio.append_pcm(bytes(2))
                with self.assertLogs(level="ERROR"):
                    _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))
                self.assertEqual(len(manager.requests[-1].audio_data), 3)
                self.assertEqual(len(_events(connection, ".failed")), 1)
                self.assertFalse(_events(connection, ".completed"))
                self.assertFalse(connection.transcription_state.has_audio)

    def test_handoff_then_rolling_requests_preserve_context_and_compact_pcm(self):
        manager, connection = _connection(
            [
                [text]
                for text in (
                    "one two three four",
                    "four five six",
                    "four five six seven",
                    "six seven eight nine",
                )
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
                self.assertEqual(state.audio.received_bytes, end * 2)
                # The handoff's provisional suffix needs another decode before
                # its audio is confirmed and the window can advance.
                self.assertEqual(
                    state.audio.last_processed_offset_bytes,
                    (60 if end == 62 else end) * 2,
                )
                self.assertEqual(state.audio.base_offset_bytes, (start - context) * 2)
                self.assertEqual(state.encoder_window_active, end > 60)
                expected_kwargs = (
                    build_audio_window_processor_kwargs(leading_context_samples=context)
                    if end > 60
                    else None
                )
                self.assertEqual(request.mm_processor_kwargs, expected_kwargs)
                self.assertLessEqual(
                    len(request.text.removeprefix("PROMPT:").split()), 3
                )

        self.assertEqual(manager.requests[1].text, "PROMPT:one two three")

    def test_stable_holdback_survives_long_silence_resume_commit_and_clear(self):
        for ending in ("commit", "resume", "clear"):
            with (
                self.subTest(ending=ending),
                get_context().override_server_args(asr_max_buffer_seconds=600),
            ):
                manager, connection = _connection([], window=True, threshold=60)
                spoken = "yes."

                async def generate(request, raw_request=None, **kwargs):
                    manager.requests.append(request)
                    prefix = request.text.removeprefix("PROMPT:")
                    self.assertTrue(spoken.startswith(prefix), (spoken, prefix))
                    yield {
                        "text": spoken[len(prefix) :],
                        "meta_info": {"finish_reason": {"type": "stop"}},
                    }

                manager.generate_request = generate

                def append(seconds=2):
                    # The fixture uses 1 Hz PCM; exercise normal append pacing.
                    stop = _run(
                        connection._on_input_audio_buffer_append(
                            SimpleNamespace(
                                audio=base64.b64encode(bytes(seconds * 2)).decode()
                            )
                        )
                    )
                    self.assertFalse(
                        stop,
                        (
                            connection.transcription_state.audio.received_bytes,
                            _events(connection, "error"),
                        ),
                    )

                # Cross the 60-s handoff and multiple 8-s window boundaries.
                for _ in range(40):
                    append()
                state = connection.transcription_state
                self.assertTrue(state.encoder_window_active)
                self.assertGreaterEqual(state.audio.last_processed_offset_bytes, 156)
                self.assertLessEqual(len(state.audio.data), 144)
                self.assertEqual(state.mode_state.suffix.confirmed_pending, "yes.")
                self.assertFalse(_events(connection, ".delta"))
                if ending == "resume":
                    spoken = "yes. we continue."
                    for _ in range(3):
                        append()
                elif ending == "clear":
                    _run(connection._on_input_audio_buffer_clear(SimpleNamespace()))
                    self.assertFalse(
                        connection.transcription_state.encoder_window_active
                    )
                    self.assertFalse(connection.transcription_state.has_audio)
                    spoken = "new item."
                # A real sub-chunk tail must be decoded at commit, even in silence.
                append(1)
                requests_before_commit = len(manager.requests)
                _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))
                self.assertEqual(len(manager.requests), requests_before_commit + 1)
                self.assertEqual(
                    _events(connection, ".completed")[0].transcript, spoken
                )
                self.assertEqual(
                    "".join(e.delta for e in _events(connection, ".delta")), spoken
                )
                self.assertFalse(connection.transcription_state.has_audio)
                self.assertFalse(connection.transcription_state.encoder_window_active)

    def test_unconfirmed_tail_retains_audio_despite_empty_or_changing_results(self):
        for empty in (False, True):
            with (
                self.subTest(empty=empty),
                get_context().override_server_args(asr_max_buffer_seconds=600),
            ):
                scripts = (
                    [["yes."]] * 30
                    + [["maybe."]]
                    + [["" if empty else f"revision{i}."] for i in range(5)]
                )
                manager, connection = _connection(scripts, window=True, threshold=60)
                event = SimpleNamespace(audio=base64.b64encode(bytes(4)).decode())
                for _ in range(35):
                    self.assertFalse(
                        _run(connection._on_input_audio_buffer_append(event))
                    )
                state = connection.transcription_state
                self.assertEqual(state.audio.last_processed_offset_bytes, 120)
                self.assertEqual(state.audio.base_offset_bytes, 0)
                self.assertEqual(state.mode_state.suffix.confirmed_pending_chars, 0)
                with self.assertLogs(level="ERROR"):
                    self.assertTrue(
                        _run(connection._on_input_audio_buffer_append(event))
                    )
                self.assertEqual(len(manager.requests), 35)
                self.assertFalse(_events(connection, ".completed"))

    def test_decoder_streaming_preserves_mixed_script_text(self):
        first = "价格是100元数量200件总额30000元税率10%预计3天交付"
        final = first + "完成"
        with get_context().override_server_args(incremental_streaming_output=True):
            manager, connection = _connection(
                [["language Chinese<asr_text>" + first, "完成"]],
                streaming=True,
                adapter=Qwen3ASRAdapter(),
            )
            connection.config.language = "zh"
            _run(
                connection._on_input_audio_buffer_append(
                    SimpleNamespace(audio=base64.b64encode(bytes(64000)).decode())
                )
            )
            _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))
            self.assertEqual(_events(connection, ".completed")[0].transcript, final)
            self.assertEqual(
                "".join(e.delta for e in _events(connection, ".delta")), final
            )
            self.assertEqual(len(manager.requests), 1)

    def test_publication_uses_one_snapshot_through_preview_final_and_flush(self):
        for window, scripts, expected_prefix, expected in (
            (
                False,
                [["one two"], ["one two"], [" two three four", " two three four five"]],
                "one",
                "one two three four five",
            ),
            (
                True,
                [
                    ["one two three four"],
                    ["one two three four"],
                    [" five six seven", " five six seven eight"],
                    [" five six seven eight nine", " five six seven eight nine ten"],
                ],
                "two three four",
                "one two three four five six seven eight nine ten",
            ),
            (
                False,
                [["two three four", "twofold three four"]],
                "",
                "twofold three four",
            ),
        ):
            with self.subTest(window=window, expected=expected):
                manager, connection = _connection(
                    scripts, window=window, streaming=True
                )
                for _ in scripts:
                    event = SimpleNamespace(audio=base64.b64encode(bytes(4)).decode())
                    self.assertFalse(
                        _run(connection._on_input_audio_buffer_append(event))
                    )
                    self.assertEqual(
                        connection.transcription_state.emitted_text,
                        "".join(e.delta for e in _events(connection, ".delta")),
                    )
                self.assertEqual(manager.requests[-1].text, "PROMPT:" + expected_prefix)
                _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))
                self.assertEqual(len(manager.requests), len(scripts))
                self.assertEqual(
                    "".join(e.delta for e in _events(connection, ".delta")), expected
                )
                self.assertEqual(
                    _events(connection, ".completed")[0].transcript, expected
                )

    def test_failed_send_records_only_successfully_published_text(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                _, connection = _connection(
                    [["one two three", "one two three four"]], streaming=streaming
                )
                delivered = []

                async def send(event):
                    if event.type.endswith(".delta"):
                        if len(delivered) == int(streaming):
                            raise ConnectionError(
                                "peer disconnected during publication"
                            )
                        delivered.append(event.delta)

                connection.transport.send.side_effect = send
                event = SimpleNamespace(audio=base64.b64encode(bytes(4)).decode())
                with self.assertLogs(level="ERROR"):
                    self.assertTrue(
                        _run(connection._on_input_audio_buffer_append(event))
                    )
                self.assertEqual(
                    connection.transcription_state.emitted_text,
                    "one" if streaming else "",
                )
                self.assertEqual(
                    connection.transcription_state.emitted_text, "".join(delivered)
                )
                self.assertEqual(
                    connection.transcription_state.mode_state.transcript.full_transcript,
                    "",
                )
                self.assertEqual(
                    connection.transcription_state.audio.last_processed_offset_bytes, 0
                )
                connection.transport.finish.assert_awaited_once_with("server_error")

    def test_failed_preview_preserves_sent_text_but_not_candidate_or_audio(self):
        for window, frames in (
            (True, ["four five six", "four five six seven", "four five five seven"]),
            (False, ["two three four", "too three four"]),
            (True, ["four five six", ("four five six seven", {"type": "length"})]),
            (True, ["four five six", ("", {"type": "abort", "status_code": 500})]),
        ):
            with self.subTest(window=window, frames=frames):
                _, connection = _connection([frames], window=window, streaming=True)
                if window:
                    _activate(connection)
                state = connection.transcription_state
                mode_before = deepcopy(state.mode_state)
                emitted_before = state.emitted_text
                state.audio.append_pcm(bytes(4))
                published = []
                with self.assertRaises(RuntimeError):
                    _step(connection, published=published)
                self.assertTrue(published)
                self.assertEqual(
                    state.emitted_text, emitted_before + "".join(published)
                )
                self.assertEqual(state.mode_state, mode_before)
                self.assertEqual(state.audio.last_attempted_offset_bytes, 0)
                self.assertEqual(state.audio.last_processed_offset_bytes, 0)
                self.assertEqual(state.audio.base_offset_bytes, 0)
                self.assertEqual(bytes(state.audio.data), bytes(4))

    def test_item_audio_limit_still_applies_after_window_fallback(self):
        with get_context().override_server_args(asr_max_buffer_seconds=6):
            manager, connection = _connection([[], [], ["one two"]], window=True)
            # Both handoff attempts fail; the same append resumes cumulatively.
            event = SimpleNamespace(audio=base64.b64encode(bytes(12)).decode())
            self.assertFalse(_run(connection._on_input_audio_buffer_append(event)))
            self.assertEqual(len(manager.requests), 3)
            self.assertIsNone(manager.requests[-1].mm_processor_kwargs)
            event.audio = base64.b64encode(bytes(2)).decode()
            self.assertTrue(_run(connection._on_input_audio_buffer_append(event)))
            connection.transport.finish.assert_awaited_once_with("buffer_overflow")
            self.assertEqual(connection.transcription_state.audio.received_bytes, 12)

    def test_window_request_requires_prefix_when_text_was_published(self):
        for active in (False, True):
            with self.subTest(active=active):
                manager, connection = _connection([], window=True)
                # The token budget fits only a fragment of the final word.
                manager.tokenizer = SimpleNamespace(
                    encode=lambda text, **kwargs: [1, 2, 3, 4],
                    decode=lambda tokens, **kwargs: "ization",
                )
                state = connection.transcription_state
                state.emitted_text = "internationalization"
                if active:
                    state.mode_state = WindowedState(suffix=TranscriptionSuffixState())
                state.audio.append_pcm(bytes(4))
                with self.assertRaisesRegex(RuntimeError, "prefix budget"):
                    _step(connection)
                self.assertFalse(manager.requests)
                self.assertEqual(state.audio.last_attempted_offset_bytes, 0)
                self.assertEqual(state.audio.last_processed_offset_bytes, 0)
                self.assertEqual(state.audio.base_offset_bytes, 0)

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
        published = []
        with self.assertLogs(level="WARNING"):
            self.assertEqual(_step(connection, published=published), "")

        self.assertEqual(published, [])
        self.assertEqual(state.audio.last_attempted_offset_bytes, 4)
        self.assertEqual(state.audio.last_processed_offset_bytes, 0)
        self.assertEqual(bytes(state.audio.data), original_pcm)
        self.assertEqual(state.emitted_text, "one two three")
        self.assertEqual(state.mode_state.suffix.pending, "one two three four five")

        state.audio.append_pcm(np.array([3, 4], dtype=np.int16).tobytes())
        _step(connection, published=published)
        np.testing.assert_array_equal(
            manager.requests[-1].audio_data,
            np.array([1, 2, 3, 4], dtype=np.float32) / 32768.0,
        )
        self.assertEqual(state.audio.last_processed_offset_bytes, 8)
        self.assertEqual(state.mode_state.consecutive_failures, 0)
        self.assertFalse(state.has_new_audio)
        self.assertEqual(manager.requests[-1].text, "PROMPT:one two three")
        self.assertEqual(state.emitted_text, "one two three one two three four")
        self.assertEqual("".join(published), " one two three four")

    def test_invalid_payload_clears_previous_client_event_id(self):
        _, connection = _connection([])
        cases = [
            (
                {"type": "unknown", "event_id": "first"},
                ("unknown_event", "first", None),
            ),
            (ValueError("Invalid JSON"), ("invalid_payload", None, None)),
            (
                {"type": "input_audio_buffer.append", "event_id": "third"},
                ("invalid_value", "third", "audio"),
            ),
            ({}, ("invalid_value", None, "type")),
            (
                {"type": [], "event_id": "bad-type"},
                ("invalid_value", "bad-type", "type"),
            ),
            (
                {"type": "input_audio_buffer.clear", "event_id": []},
                ("invalid_value", None, "event_id"),
            ),
        ]
        connection.transport.receive.side_effect = [raw for raw, _ in cases] + [
            {"type": "input_audio_buffer.clear", "event_id": None},
            {"type": "input_audio_buffer.clear", "event_id": ""},
            {"type": "input_audio_buffer.clear"},
            None,
        ]
        _run(connection.run())
        errors = [event.error for event in _events(connection, "error")]
        self.assertEqual(
            [(error.code, error.event_id, error.param) for error in errors],
            [expected for _, expected in cases],
        )
        self.assertEqual(len(_events(connection, ".cleared")), 3)
        connection.transport.finish.assert_not_awaited()

    def test_connection_exit_waits_for_backend_cleanup_and_releases_session(self):
        async def run(exit_reason):
            manager, connection = _connection([], blocked=True)
            incoming = asyncio.Queue()
            incoming.put_nowait({"type": "websocket.connect"})
            sent = AsyncMock()
            websocket = WebSocket(
                {"type": "websocket", "path": "/v1/realtime", "headers": []},
                incoming.get,
                sent,
            )

            def bind_transport(transport, *args, **kwargs):
                self.assertEqual(transport._max_pending_bytes, 11_520_000)
                transport._max_pending_bytes = 1024
                connection.transport = transport
                return connection

            semaphore = asyncio.Semaphore(1)
            with patch(
                "sglang.srt.entrypoints.openai.realtime.handler.RealtimeASRSession",
                side_effect=bind_transport,
            ):
                task = asyncio.create_task(
                    handle_realtime_transcription(
                        WebSocketRealtimeTransport(websocket),
                        manager,
                        connection.adapter,
                        semaphore,
                    )
                )
                try:
                    incoming.put_nowait(
                        {
                            "type": "websocket.receive",
                            "text": json.dumps(
                                {
                                    "type": "input_audio_buffer.append",
                                    "event_id": "active-append",
                                    "audio": base64.b64encode(bytes(4)).decode(),
                                }
                            ),
                        }
                    )
                    await asyncio.wait_for(manager.started.wait(), timeout=2)
                    self.assertTrue(semaphore.locked())
                    incoming.put_nowait({"type": "websocket.receive", "text": "{}"})
                    await asyncio.sleep(0)
                    self.assertFalse(connection.transport._incoming_messages.empty())
                    if exit_reason == "cancel":
                        task.cancel()
                    else:
                        incoming.put_nowait(
                            {"type": "websocket.disconnect", "code": 1000}
                            if exit_reason == "disconnect"
                            else {"type": "websocket.receive", "text": "x" * 1024}
                        )
                    result = await asyncio.wait_for(
                        asyncio.gather(task, return_exceptions=True), timeout=2
                    )
                    if exit_reason == "cancel":
                        self.assertIsInstance(result[0], asyncio.CancelledError)
                    else:
                        self.assertEqual(result, [None])
                    self.assertTrue(manager.cleaned.is_set())
                    self.assertFalse(semaphore.locked())
                    self.assertTrue(connection.transport._incoming_messages.empty())
                    self.assertEqual(connection.transport._pending_input_bytes, 0)
                    if exit_reason == "overflow":
                        messages = [call.args[0] for call in sent.call_args_list]
                        error = json.loads(messages[-2]["text"])["error"]
                        self.assertEqual(
                            (error["code"], error["event_id"]),
                            ("buffer_overflow", None),
                        )
                        self.assertEqual(messages[-1]["code"], 1009)
                finally:
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

        for exit_reason, item_limit in (
            ("disconnect", 60),
            ("overflow", 600),
            ("cancel", 3600),
        ):
            with (
                self.subTest(exit_reason=exit_reason),
                get_context().override_server_args(asr_max_buffer_seconds=item_limit),
            ):
                _run(run(exit_reason))


if __name__ == "__main__":
    unittest.main()
