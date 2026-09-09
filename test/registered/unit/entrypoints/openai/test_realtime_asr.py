"""Focused CPU coverage for realtime ASR request and transcript contracts."""

import asyncio
import base64
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, create_autospec, patch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that pull in sgl_kernel

import msgspec
import numpy as np
from fastapi import WebSocket

from sglang.srt.entrypoints.openai.realtime.asr_processor import (
    RealtimeASRProcessor,
)
from sglang.srt.entrypoints.openai.realtime.audio_buffer import AudioBuffer
from sglang.srt.entrypoints.openai.realtime.decoder_suffix import DecoderSuffixState
from sglang.srt.entrypoints.openai.realtime.encoder_window_policy import (
    ResolvedEncoderWindowPolicy,
    resolve_realtime_encoder_window_policy,
)
from sglang.srt.entrypoints.openai.realtime.handler import (
    handle_realtime_transcription,
)
from sglang.srt.entrypoints.openai.realtime.protocol import (
    AudioTranscription,
    SessionUpdateEvent,
    TranscriptionSessionAudio,
    TranscriptionSessionAudioInput,
    TranscriptionSessionConfig,
)
from sglang.srt.entrypoints.openai.realtime.session import RealtimeConnection
from sglang.srt.entrypoints.openai.serving_transcription import (
    OpenAIServingTranscription,
)
from sglang.srt.entrypoints.openai.streaming_asr import (
    StreamingASRState,
    generate_asr_transcript,
    process_asr_chunk,
)
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    RealtimeEncoderWindowPolicy,
)
from sglang.srt.entrypoints.openai.transcription_adapters.qwen3_asr import (
    Qwen3ASRAdapter,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.multimodal.encoder_window import (
    EncoderWindowCapability,
    EncoderWindowConfig,
    encoder_window_kwargs,
)
from sglang.srt.runtime_context import get_context, get_server_args, get_serving
from sglang.srt.utils import get_or_create_event_loop
from sglang.srt.utils.common import load_audio
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_override

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _RuntimeConfigTestCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        enter_override(
            self,
            get_context().override_server_args(
                asr_max_buffer_seconds=60,
                enable_asr_decoder_streaming=False,
                incremental_streaming_output=False,
            ),
        )


def _adapter(*, chunk_size_sec=2.0, unfixed_token_num=1):
    return SimpleNamespace(
        prompt_template="PROMPT:",
        model_sample_rate=1,
        supports_chunked_streaming=True,
        postprocess_text=lambda text: text,
        postprocess_streaming_text=lambda text, continuation=False: text,
        build_sampling_params=lambda request: {"language": request.language},
        chunked_streaming_config={
            "chunk_size_sec": chunk_size_sec,
            "unfixed_chunk_num": 2,
            "unfixed_token_num": unfixed_token_num,
        },
    )


class _MockTokenizerManager:
    """Scripted backend: one transcript (or None for no response) per request."""

    def __init__(self, transcripts=None, *, fail=False, finish_reasons=None):
        self._transcripts = (
            list(transcripts) if isinstance(transcripts, list) else [transcripts]
        )
        self._finish_reasons = (
            list(finish_reasons)
            if finish_reasons is not None
            else ["stop"] * len(self._transcripts)
        )
        self._fail = fail
        self.served_model_name = "qwen3-asr"
        self.requests = []
        self.tokenizer = SimpleNamespace(
            encode=lambda text, add_special_tokens=False: text.split(),
            decode=lambda token_ids, **kwargs: " ".join(token_ids),
        )

    def generate_request(self, adapted_request, raw_request=None, **kwargs):
        self.requests.append(adapted_request)
        transcript = self._transcripts.pop(0)

        async def generate():
            if self._fail:
                raise ValueError("synthetic failure")
            if transcript is not None:
                finish_reason = self._finish_reasons.pop(0)
                if not isinstance(finish_reason, dict):
                    finish_reason = {"type": finish_reason}
                response = {
                    "text": transcript,
                    "meta_info": {"finish_reason": finish_reason},
                }
                # Match the real stream/non-stream abort contract instead of
                # yielding a frame the manager would turn into an exception.
                await TokenizerManager._handle_abort_finish_reason(
                    SimpleNamespace(rid_to_state={}, enable_lora=False),
                    response,
                    SimpleNamespace(obj=adapted_request),
                    is_stream=adapted_request.stream,
                )
                yield response

        return generate()


def _run(coro):
    return get_or_create_event_loop().run_until_complete(coro)


def _server_args(max_buffer_seconds=60, *, decoder_streaming=False):
    get_context().override(
        "realtime_asr_test",
        asr_max_buffer_seconds=max_buffer_seconds,
        enable_asr_decoder_streaming=decoder_streaming,
    )
    return get_server_args()


# 1 Hz PCM16: 2 bytes per second; 2 s chunks are 4 bytes; a window is 32 bytes.
_WINDOW_CONFIG = EncoderWindowConfig(
    sample_rate=1,
    hop_length=2,
    n_fft=4,
    window_frames=8,
    window_samples=16,
    window_tokens=8,
    min_tail_samples=4,
    context_samples=4,
)


def _policy(min_audio_sec=0.0, *, dp_size=1, languages=("en",)):
    return ResolvedEncoderWindowPolicy(
        config=_WINDOW_CONFIG,
        policy=RealtimeEncoderWindowPolicy(
            min_audio_sec=min_audio_sec,
            max_audio_context_windows=2,
            supported_languages=languages,
            decoder_prefix_max_tokens=3,
            decoder_prefix_holdback_units=1,
        ),
        dp_size=dp_size,
    )


def _capable_processor(config=_WINDOW_CONFIG):
    """A processor that satisfies EncoderWindowCapability with resolved geometry."""
    processor = create_autospec(EncoderWindowCapability, instance=True)
    processor.encoder_window_config.return_value = config
    return processor


def _processor(
    transcripts, *, encoder_window=None, session_id="sess_test", **adapter_kwargs
):
    manager = _MockTokenizerManager(transcripts)
    processor = RealtimeASRProcessor(
        manager,
        _adapter(**adapter_kwargs),
        _server_args(120),
        encoder_window=encoder_window,
        session_id=session_id,
    )
    return manager, processor, processor.create_state()


def _process(processor, state, *, is_last=False):
    return _run(
        processor.process(state, is_last=is_last, language="en", sampling_params={})
    )


def _connection(
    transcripts,
    finish_reasons=None,
    *,
    manager=None,
    max_buffer_seconds=120,
    encoder_window=None,
):
    manager = manager or _MockTokenizerManager(
        transcripts, finish_reasons=finish_reasons
    )
    connection = RealtimeConnection(
        Mock(send_text=AsyncMock(), close=AsyncMock()),
        manager,
        _adapter(),
        _server_args(max_buffer_seconds),
        encoder_window=encoder_window,
    )
    connection.config.configured = True
    connection.config.input_sample_rate = 1
    connection.config.language = "en"
    connection.config.sampling_params = {"max_new_tokens": 256}
    connection._send = AsyncMock()
    return manager, connection


def _append_seconds(connection, seconds):
    return _run(
        connection._on_input_audio_buffer_append(
            SimpleNamespace(audio=base64.b64encode(bytes(seconds * 2)).decode())
        )
    )


def _sent_errors(connection):
    """Error envelopes go straight to the socket, not through _send."""
    errors = []
    for call in connection.websocket.send_text.call_args_list:
        payload = json.loads(call.args[0])
        if payload.get("type") == "error":
            errors.append(payload["error"])
    return errors


def _sent_events(connection, suffix):
    return [
        call.args[0]
        for call in connection._send.call_args_list
        if call.args[0].type.endswith(suffix)
    ]


class TestAudioBuffer(CustomTestCase):
    def test_offsets_stay_absolute_across_compaction(self):
        buffer = AudioBuffer()
        buffer.append_pcm(bytes(range(12)))

        self.assertEqual(buffer.received_bytes, 12)
        self.assertEqual(buffer.snapshot(2, 6), bytes([2, 3, 4, 5]))

        buffer.discard_before(4)
        self.assertEqual(buffer.base_offset_bytes, 4)
        self.assertEqual(buffer.received_bytes, 12)
        self.assertEqual(buffer.snapshot(4, 12), bytes(range(4, 12)))
        buffer.append_pcm(bytes([12, 13]))
        self.assertEqual(buffer.snapshot(10, 14), bytes([10, 11, 12, 13]))

        with self.assertRaisesRegex(ValueError, "outside resident"):
            buffer.snapshot(2, 6)
        with self.assertRaisesRegex(ValueError, "sample-aligned"):
            buffer.discard_before(5)
        with self.assertRaisesRegex(ValueError, "outside resident"):
            buffer.discard_before(2)
        with self.assertRaisesRegex(ValueError, "outside resident"):
            buffer.discard_before(16)
        buffer.discard_before(4)  # no-op at the resident start
        self.assertEqual(buffer.base_offset_bytes, 4)

    def test_load_audio_passes_decoded_samples_through(self):
        mono = np.arange(4, dtype=np.float32)
        self.assertIs(load_audio(mono), mono)
        stereo = np.array([[0.0, 2.0], [1.0, 3.0]], dtype=np.float32)
        np.testing.assert_array_equal(load_audio(stereo), np.array([1.0, 2.0]))
        self.assertIs(load_audio(stereo, mono=False), stereo)


class TestRealtimeASRProcessor(_RuntimeConfigTestCase):
    def test_rejects_invalid_chunk_sizes(self):
        manager = _MockTokenizerManager("text")
        for chunk_size in (0.0, -1.0, float("inf"), 1e-20, 0.5):
            with self.subTest(chunk_size=chunk_size), self.assertRaises(ValueError):
                RealtimeASRProcessor(
                    manager, _adapter(chunk_size_sec=chunk_size), _server_args()
                )

    def test_chunk_cadence_advances_cursors(self):
        manager, processor, state = _processor(["one", "one two", "one two three"])
        self.assertEqual(processor.chunk_size_bytes, 4)
        self.assertEqual(processor.max_item_bytes("en"), 240)

        state.audio.append_pcm(bytes(6))
        self.assertTrue(processor.is_chunk_ready(state))
        _process(processor, state)
        self.assertEqual(manager.requests[-1].audio_data.size, 2)
        self.assertEqual(manager.requests[-1].text, "PROMPT:")
        self.assertFalse(manager.requests[-1].stream)
        self.assertEqual(state.audio.last_attempted_offset_bytes, 4)
        self.assertEqual(state.audio.last_processed_offset_bytes, 4)
        self.assertFalse(processor.is_chunk_ready(state))

        state.audio.append_pcm(bytes(2))
        self.assertTrue(processor.is_chunk_ready(state))
        _process(processor, state)
        self.assertEqual(manager.requests[-1].audio_data.size, 4)

        state.audio.append_pcm(bytes(2))
        _process(processor, state, is_last=True)
        self.assertEqual(manager.requests[-1].audio_data.size, 5)
        self.assertEqual(state.audio.last_processed_offset_bytes, 10)

    def test_pcm_samples_match_soundfile_normalization(self):
        manager, processor, state = _processor("one")
        state.audio.append_pcm(np.array([-32768, 16384], dtype=np.int16).tobytes())
        _process(processor, state)
        np.testing.assert_array_equal(
            manager.requests[0].audio_data, np.array([-1.0, 0.5], dtype=np.float32)
        )


class TestRealtimeConnection(_RuntimeConfigTestCase):
    def test_large_append_is_drained_one_chunk_at_a_time(self):
        manager, connection = _connection(["one"] * 45)

        self.assertFalse(_append_seconds(connection, 90))

        sizes = [request.audio_data.size for request in manager.requests]
        self.assertEqual(sizes, list(range(2, 92, 2)))
        self.assertEqual(connection.asr_state.audio.last_attempted_offset_bytes, 180)

    def test_wire_deltas_join_to_completed_transcript(self):
        streaming_manager = _StreamingTokenizerManager(
            [
                ["one", "one two", "one two three"],
                ["one two three four", "one two three four five"],
                ["one two three four five six"],
            ]
        )
        cases = (
            (
                "plain",
                _connection(["one two", "one two three", "one two three four"])[1],
                "one two three four",
            ),
            (
                "streaming",
                _connection(None, manager=streaming_manager)[1],
                "one two three four five six",
            ),
        )
        for name, connection, expected in cases:
            with self.subTest(name):
                connection.asr_processor.decoder_streaming = name == "streaming"
                connection.asr_state.audio.append_pcm(bytes(4))
                self.assertTrue(_run(connection._run_inference(is_last=False)))
                connection.asr_state.audio.append_pcm(bytes(4))
                self.assertTrue(_run(connection._run_inference(is_last=False)))
                connection.asr_state.audio.append_pcm(bytes(2))
                _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))

                deltas = [event.delta for event in _sent_events(connection, ".delta")]
                completed = _sent_events(connection, ".completed")[-1].transcript
                self.assertEqual("".join(deltas), completed)
                self.assertEqual(completed, expected)
        self.assertEqual(
            [request.stream for request in streaming_manager.requests],
            [True, True, False],
        )

    def test_commit_without_new_audio_flushes_held_back_text(self):
        manager, connection = _connection(["one two three"])
        connection.asr_state.audio.append_pcm(bytes(4))
        self.assertTrue(_run(connection._run_inference(is_last=False)))
        self.assertEqual(
            [event.delta for event in _sent_events(connection, ".delta")],
            ["one", " two"],
        )

        _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))

        self.assertEqual(len(manager.requests), 1)
        self.assertEqual(
            _sent_events(connection, ".completed")[-1].transcript, "one two three"
        )

        # The same commit path flushes a windowed item's pending suffix.
        manager, connection = _connection(["x"], encoder_window=_policy())
        _activate(connection.asr_state, emitted="one two")
        connection.asr_state.decoder_suffix.pending = "three"
        connection.item.emitted_deltas.extend(["one", " two"])
        _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))
        self.assertEqual(manager.requests, [])
        self.assertEqual(
            _sent_events(connection, ".completed")[-1].transcript, "one two three"
        )

    def test_empty_response_retries_audio_but_final_empty_fails(self):
        _, connection = _connection([None, "one two three"])
        connection.asr_state.audio.append_pcm(bytes(4))
        self.assertTrue(_run(connection._run_inference(is_last=False)))
        self.assertEqual(connection.asr_state.audio.last_attempted_offset_bytes, 4)
        self.assertEqual(connection.asr_state.audio.last_processed_offset_bytes, 0)
        self.assertTrue(_run(connection._run_inference(is_last=True)))

        _, connection = _connection([None])
        connection.asr_state.audio.append_pcm(bytes(4))
        self.assertFalse(_run(connection._run_inference(is_last=True)))
        connection.websocket.close.assert_not_awaited()
        self.assertTrue(_sent_events(connection, ".failed"))

        _, connection = _connection([None, ""])
        connection.asr_state.audio.append_pcm(bytes(4))
        self.assertTrue(_run(connection._run_inference(is_last=False)))
        _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))
        self.assertTrue(_sent_events(connection, ".completed"))

    def test_truncated_cumulative_commit_uses_clean_final_result(self):
        for final_reason, completed in (("length", False), ("stop", True)):
            with self.subTest(final_reason=final_reason):
                manager, connection = _connection(
                    ["one two", "one two three"], ["length", final_reason]
                )
                connection.asr_state.audio.append_pcm(bytes(4))
                self.assertTrue(_run(connection._run_inference(is_last=False)))
                self.assertEqual(
                    connection.asr_state.audio.last_processed_offset_bytes, 0
                )
                _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))
                self.assertEqual(len(manager.requests), 2)
                self.assertEqual(
                    bool(_sent_events(connection, ".completed")), completed
                )
                self.assertEqual(
                    bool(_sent_events(connection, ".failed")), not completed
                )

        for final_text, completed in (("one two three", True), ("", False)):
            with self.subTest(recovery_after_empty=final_text):
                manager, connection = _connection(
                    ["one two", "", final_text], ["length", "stop", "stop"]
                )
                connection.asr_state.audio.append_pcm(bytes(4))
                self.assertTrue(_run(connection._run_inference(is_last=False)))
                connection.asr_state.audio.append_pcm(bytes(4))
                self.assertTrue(_run(connection._run_inference(is_last=False)))
                self.assertEqual(
                    connection.asr_state.audio.last_processed_offset_bytes, 0
                )

                _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))

                self.assertEqual(len(manager.requests), 3)
                self.assertEqual(
                    bool(_sent_events(connection, ".completed")), completed
                )
                self.assertEqual(
                    bool(_sent_events(connection, ".failed")), not completed
                )

    def test_aborted_response_fails_the_step_without_committing_state(self):
        abort = {"type": "abort", "message": "abort", "status_code": 500}
        streaming_manager = _StreamingTokenizerManager([["one two three", abort]])
        for name, connection in (
            ("plain", _connection("partial", [abort])[1]),
            ("streaming", _connection(None, manager=streaming_manager)[1]),
        ):
            with self.subTest(name):
                connection.asr_processor.decoder_streaming = name == "streaming"
                connection.asr_state.audio.append_pcm(bytes(4))
                with self.assertLogs(level="ERROR"):
                    self.assertFalse(_run(connection._run_inference(is_last=False)))
                self.assertEqual(
                    connection.asr_state.audio.last_processed_offset_bytes, 0
                )
                self.assertEqual(connection.asr_state.transcript.full_transcript, "")
                if name == "plain":
                    self.assertFalse(_sent_events(connection, ".delta"))
                connection.websocket.close.assert_awaited_with(code=1011)

    def test_failed_inference_preserves_audio_and_reset_clears_offsets(self):
        manager = _MockTokenizerManager(fail=True)
        _, connection = _connection(None, manager=manager)
        connection.asr_state.audio.append_pcm(bytes(range(12)))
        connection.asr_state.audio.last_attempted_offset_bytes = 8
        connection.asr_state.audio.last_processed_offset_bytes = 8

        with self.assertLogs(level="WARNING"):
            self.assertFalse(_run(connection._run_inference(is_last=False)))
        self.assertEqual(bytes(connection.asr_state.audio.data), bytes(range(12)))
        self.assertEqual(connection.asr_state.audio.last_processed_offset_bytes, 8)
        connection.websocket.close.assert_awaited_with(code=1011)

        connection._reset_inference_state()
        self.assertEqual(connection.asr_state.audio.data, bytearray())
        self.assertEqual(connection.asr_state.audio.received_bytes, 0)
        self.assertEqual(connection.asr_state.audio.last_attempted_offset_bytes, 0)
        self.assertEqual(connection.asr_state.audio.last_processed_offset_bytes, 0)


def _window_kwargs(request):
    return (request.mm_processor_kwargs or {}).get("encoder_window")


def _window_payload(leading_context_samples=0):
    return encoder_window_kwargs(leading_context_samples=leading_context_samples)[
        "encoder_window"
    ]


def _activate(state, emitted="one two three"):
    """Put an item into steady-state windowing."""
    state.decoder_suffix = DecoderSuffixState(emitted_text=emitted)


class TestEncoderWindowPolicy(_RuntimeConfigTestCase):
    def test_policy_validation_and_language_matching(self):
        defaults = {
            "min_audio_sec": 60.0,
            "max_audio_context_windows": 8,
            "supported_languages": ("en", "ZH_cn"),
        }
        policy = RealtimeEncoderWindowPolicy(**defaults)
        self.assertEqual(policy.supported_languages, ("en", "zh"))
        for language, supported in (
            ("en", True),
            ("en-US", True),
            ("zh-CN", True),
            ("fr", False),
            ("auto", False),
            (None, False),
        ):
            self.assertEqual(policy.supports_language(language), supported, language)
        for override in (
            {"min_audio_sec": float("inf")},
            {"min_audio_sec": -1.0},
            {"max_audio_context_windows": 0},
            {"decoder_prefix_max_tokens": 0},
            {"decoder_prefix_holdback_units": -1},
            {"supported_languages": ()},
            {"supported_languages": ("",)},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                RealtimeEncoderWindowPolicy(**(defaults | override))

    def test_resolution_requires_adapter_and_processor_declarations(self):
        adapter = _adapter()
        adapter.realtime_encoder_window_policy = None
        capable = _capable_processor()
        with self.assertLogs(level="WARNING"):
            self.assertIsNone(
                resolve_realtime_encoder_window_policy(
                    adapter=adapter,
                    mm_processor=capable,
                    tokenizer=object(),
                    max_buffer_seconds=120,
                    dp_size=1,
                )
            )
        adapter.realtime_encoder_window_policy = _policy().policy
        with self.assertLogs(level="WARNING"):
            self.assertIsNone(
                resolve_realtime_encoder_window_policy(
                    adapter=adapter,
                    mm_processor=None,
                    tokenizer=object(),
                    max_buffer_seconds=120,
                    dp_size=1,
                )
            )
        # The processor's extractor must run at the adapter's sample rate.
        with self.assertRaisesRegex(ValueError, "sample rate"):
            resolve_realtime_encoder_window_policy(
                adapter=adapter,
                mm_processor=_capable_processor(
                    msgspec.structs.replace(_WINDOW_CONFIG, sample_rate=2)
                ),
                tokenizer=object(),
                max_buffer_seconds=120,
                dp_size=1,
            )
        resolved = resolve_realtime_encoder_window_policy(
            adapter=adapter,
            mm_processor=capable,
            tokenizer=object(),
            max_buffer_seconds=120,
            dp_size=2,
        )
        self.assertEqual(resolved.config, _WINDOW_CONFIG)
        self.assertEqual(resolved.dp_size, 2)
        self.assertEqual(resolved.window_bytes, 32)
        self.assertEqual(resolved.activation_threshold_bytes(4, 2.0), 0)
        self.assertEqual(_policy(60.0).activation_threshold_bytes(4, 2.0), 120)
        self.assertEqual(_policy(61.0).activation_threshold_bytes(4, 2.0), 124)
        overridden = resolve_realtime_encoder_window_policy(
            adapter=adapter,
            mm_processor=capable,
            tokenizer=object(),
            max_buffer_seconds=120,
            dp_size=2,
            min_audio_sec=30.5,
            max_audio_context_windows=4,
            decoder_prefix_max_tokens=128,
            decoder_prefix_holdback_units=0,
        )
        self.assertEqual(overridden.policy.decoder_prefix_max_tokens, 128)
        self.assertEqual(overridden.policy.decoder_prefix_holdback_units, 0)
        self.assertEqual(overridden.policy.min_audio_sec, 30.5)
        self.assertEqual(overridden.policy.max_audio_context_windows, 4)
        self.assertEqual(overridden.activation_threshold_bytes(4, 2.0), 64)
        self.assertEqual(overridden.config, resolved.config)
        self.assertIs(resolved.policy, adapter.realtime_encoder_window_policy)
        self.assertEqual(resolved.policy.min_audio_sec, 0.0)
        self.assertEqual(resolved.policy.max_audio_context_windows, 2)
        self.assertEqual(resolved.policy.decoder_prefix_max_tokens, 3)
        self.assertEqual(resolved.policy.decoder_prefix_holdback_units, 1)

    def test_serving_layer_resolves_the_policy_once(self):
        manager = Mock()
        manager.model_config.hf_config.architectures = [
            "Qwen3ASRForConditionalGeneration"
        ]
        get_context().override(
            "realtime_asr_test",
            incremental_streaming_output=False,
            asr_max_concurrent_sessions=4,
            asr_max_buffer_seconds=1800,
            enable_asr_encoder_window=True,
            enable_asr_decoder_streaming=False,
            asr_encoder_window_min_audio_seconds=None,
            asr_encoder_window_max_context_windows=None,
            asr_decoder_prefix_max_tokens=None,
            asr_decoder_prefix_holdback_units=None,
        )
        manager.server_args = get_server_args()
        manager.mm_processor = _capable_processor(
            msgspec.structs.replace(_WINDOW_CONFIG, sample_rate=16000)
        )
        manager.elastic_worker_count = 2
        manager.served_model_name = "qwen3-asr"
        serving = OpenAIServingTranscription(manager)
        for _ in range(3):
            RealtimeConnection(
                Mock(),
                manager,
                serving._adapter,
                manager.server_args,
                encoder_window=serving._encoder_window,
            )
        self.assertEqual(manager.mm_processor.encoder_window_config.call_count, 1)
        self.assertEqual(serving._encoder_window.dp_size, 2)
        self.assertEqual(serving._encoder_window.policy.min_audio_sec, 60.0)

        for prefix, holdback, expected in (
            (512, None, (512, 1)),
            (None, 0, (192, 0)),
            (None, None, (192, 1)),
            (128, 2, (128, 2)),
        ):
            with self.subTest(prefix=prefix, holdback=holdback):
                get_context().override(
                    "realtime_asr_test",
                    asr_decoder_prefix_max_tokens=prefix,
                    asr_decoder_prefix_holdback_units=holdback,
                )
                policy = OpenAIServingTranscription(manager)._encoder_window.policy
                self.assertEqual(
                    (
                        policy.decoder_prefix_max_tokens,
                        policy.decoder_prefix_holdback_units,
                    ),
                    expected,
                )

        for threshold, windows in ((0.0, None), (None, 4), (30.5, 6), (None, None)):
            with self.subTest(threshold=threshold, windows=windows):
                get_context().override(
                    "realtime_asr_test",
                    asr_encoder_window_min_audio_seconds=threshold,
                    asr_encoder_window_max_context_windows=windows,
                )
                resolved = OpenAIServingTranscription(manager)._encoder_window
                self.assertEqual(
                    resolved.policy.min_audio_sec,
                    threshold if threshold is not None else 60.0,
                )
                self.assertEqual(
                    resolved.policy.max_audio_context_windows,
                    windows if windows is not None else 6,
                )
                self.assertEqual(resolved.config, serving._encoder_window.config)

        get_context().override("realtime_asr_test", enable_asr_encoder_window=False)
        self.assertIsNone(OpenAIServingTranscription(manager)._encoder_window)

    def test_dp_pinning_is_stable_per_session(self):
        def rank(session_id, dp_size=4, encoder_window=True):
            return RealtimeASRProcessor(
                _MockTokenizerManager("x"),
                _adapter(),
                _server_args(),
                encoder_window=_policy(dp_size=dp_size) if encoder_window else None,
                session_id=session_id,
            ).routed_dp_rank

        self.assertIn(rank("sess_a"), range(4))
        self.assertEqual(rank("sess_a"), rank("sess_a"))
        self.assertGreater(len({rank(f"sess_{i}") for i in range(16)}), 1)
        self.assertIsNone(rank("sess_a", dp_size=1))
        self.assertIsNone(rank("sess_a", encoder_window=False))

        manager, processor, state = _processor(
            ["one", "two three"], encoder_window=_policy(dp_size=4)
        )
        state.audio.append_pcm(bytes(4))
        _process(processor, state)
        state.audio.append_pcm(bytes(4))
        _process(processor, state)
        self.assertIsNotNone(processor.routed_dp_rank)
        self.assertEqual(
            [request.routed_dp_rank for request in manager.requests],
            [processor.routed_dp_rank] * 2,
        )


class TestEncoderWindowActivation(_RuntimeConfigTestCase):
    def test_activation_requires_policy_language_and_threshold(self):
        manager, processor, state = _processor(
            "alpha beta", encoder_window=_policy(60.0)
        )
        self.assertEqual(processor.activation_threshold_bytes, 120)
        state.audio.append_pcm(bytes(120))
        _process(processor, state)
        self.assertIsNone(_window_kwargs(manager.requests[0]))

        manager, processor, state = _processor(
            "four five", encoder_window=_policy(60.0)
        )
        state.transcript.emitted_text = "one two three"
        state.audio.append_pcm(bytes(122))
        state.audio.last_attempted_offset_bytes = 120
        state.audio.last_processed_offset_bytes = 120
        _process(processor, state)
        self.assertEqual(_window_kwargs(manager.requests[0]), _window_payload())
        self.assertTrue(state.encoder_window_active)

        for language in (None, "auto", "zh"):
            manager, processor, state = _processor(
                "alpha beta", encoder_window=_policy(60.0)
            )
            state.audio.append_pcm(bytes(122))
            state.audio.last_attempted_offset_bytes = 120
            state.audio.last_processed_offset_bytes = 120
            _run(
                processor.process(
                    state, is_last=False, language=language, sampling_params={}
                )
            )
            self.assertIsNone(_window_kwargs(manager.requests[0]))
            self.assertEqual(processor.max_item_bytes(language), 120)
        self.assertEqual(processor.max_item_bytes("en"), 240)
        self.assertEqual(
            processor.max_item_bytes("en", encoder_window_disabled=True), 120
        )

        manager, processor, state = _processor("alpha beta")
        state.audio.append_pcm(bytes(122))
        _process(processor, state)
        self.assertIsNone(_window_kwargs(manager.requests[0]))
        self.assertEqual(processor.max_item_bytes("zh"), 240)

        # The final decode of an item never switches modes.
        manager, processor, state = _processor("alpha beta", encoder_window=_policy())
        state.audio.append_pcm(bytes(4))
        _process(processor, state, is_last=True)
        self.assertIsNone(_window_kwargs(manager.requests[0]))
        self.assertFalse(state.encoder_window_active)

        # Once active, the language is fixed for the item.
        _, processor, state = _processor("x", encoder_window=_policy())
        _activate(state)
        state.audio.append_pcm(bytes(4))
        with self.assertRaisesRegex(RuntimeError, "language cannot change"):
            _run(
                processor.process(
                    state, is_last=False, language="zh", sampling_params={}
                )
            )

    def test_handoff_then_steady_state(self):
        manager, processor, state = _processor(
            ["four five", "four five six", "five six seven"],
            encoder_window=_policy(),
        )
        state.transcript.emitted_text = "one two three"
        state.transcript.confirmed_text = "one two three"
        state.transcript.full_transcript = "one two three four"

        deltas = []
        for pcm, is_last in ((bytes(4), False), (bytes(4), False), (bytes(2), True)):
            state.audio.append_pcm(pcm)
            deltas.append(_process(processor, state, is_last=is_last))

        self.assertEqual(deltas, ["", "four", "five six seven"])
        self.assertEqual(
            state.decoder_suffix.emitted_text, "one two three four five six seven"
        )
        self.assertEqual(state.decoder_suffix.pending, "")
        # "four" is still pending after the handoff, so the second prefix is
        # unchanged; the final request sees it emitted and slides the window.
        self.assertEqual(
            [r.text for r in manager.requests],
            ["PROMPT:one two three", "PROMPT:one two three", "PROMPT:two three four"],
        )
        self.assertEqual([r.audio_data.size for r in manager.requests], [2, 4, 5])
        self.assertTrue(
            all(_window_kwargs(r) == _window_payload() for r in manager.requests)
        )
        self.assertEqual(state.audio.last_processed_offset_bytes, 10)

    def test_steady_state_start_is_window_aligned_and_compacts(self):
        # (last_processed, window-aligned start, context bytes sent before it)
        for last_processed, expected_start, context in (
            (196, 128, 8),
            (100, 96, 8),
            (0, 0, 0),
        ):
            with self.subTest(last_processed=last_processed):
                manager, processor, state = _processor("x y", encoder_window=_policy())
                _activate(state)
                state.audio.append_pcm(bytes(200))
                state.audio.last_attempted_offset_bytes = 196
                state.audio.last_processed_offset_bytes = last_processed
                _process(processor, state)
                self.assertEqual(
                    manager.requests[0].audio_data.size,
                    (200 - expected_start + context) // 2,
                )
                self.assertEqual(
                    _window_kwargs(manager.requests[0]),
                    _window_payload(leading_context_samples=context // 2),
                )
                # Compaction keeps exactly the context the next request needs.
                self.assertEqual(
                    state.audio.base_offset_bytes, expected_start - context
                )
                self.assertEqual(state.audio.received_bytes, 200)
                self.assertEqual(expected_start % 32, 0)

        # A compacted buffer never asks for audio before its resident start:
        # the first usable window start is the next aligned one past the
        # resident context.
        manager, processor, state = _processor("x y", encoder_window=_policy())
        _activate(state)
        state.audio.append_pcm(bytes(200))
        state.audio.discard_before(160)
        state.audio.last_attempted_offset_bytes = 196
        state.audio.last_processed_offset_bytes = 196
        _process(processor, state)
        self.assertEqual(manager.requests[0].audio_data.size, (200 - 192 + 8) // 2)
        self.assertEqual(state.audio.base_offset_bytes, 184)

    def test_empty_continuation_is_deferred_once(self):
        manager, processor, state = _processor(["", "", "x"], encoder_window=_policy())
        _activate(state)
        state.decoder_suffix.pending = "two"
        state.audio.append_pcm(bytes(4))
        self.assertEqual(_process(processor, state), "")
        self.assertEqual(state.audio.last_processed_offset_bytes, 0)
        self.assertTrue(state.deferred_empty_continuation)
        state.audio.append_pcm(bytes(4))
        self.assertEqual(_process(processor, state), "")
        self.assertEqual(state.audio.last_processed_offset_bytes, 8)
        self.assertEqual(state.decoder_suffix.pending, "two")
        self.assertFalse(state.deferred_empty_continuation)
        self.assertEqual([r.audio_data.size for r in manager.requests], [2, 4])

    def test_truncated_windowed_step_retains_audio(self):
        manager, processor, state = _processor(
            ["a b", "a b c"], encoder_window=_policy()
        )
        manager._finish_reasons = ["length", "stop"]
        _activate(state)
        state.audio.append_pcm(bytes(4))
        with self.assertLogs(level="WARNING"):
            self.assertEqual(_process(processor, state), "")
        self.assertEqual(state.audio.last_processed_offset_bytes, 0)
        self.assertEqual(state.consecutive_window_failures, 1)
        state.audio.append_pcm(bytes(4))
        _process(processor, state)
        self.assertEqual(state.audio.last_processed_offset_bytes, 8)
        self.assertEqual(state.consecutive_window_failures, 0)

    def test_repeated_failures_fall_back_before_activation_and_fail_after(self):
        manager, processor, state = _processor(
            [None, None, "one two"], encoder_window=_policy(2.0)
        )
        state.audio.append_pcm(bytes(6))
        state.audio.last_attempted_offset_bytes = 4
        state.audio.last_processed_offset_bytes = 4
        with self.assertLogs(level="WARNING"):
            self.assertEqual(_process(processor, state), "")
            self.assertFalse(state.encoder_window_disabled)
            state.audio.append_pcm(bytes(2))
            self.assertEqual(_process(processor, state), "")
        self.assertTrue(state.encoder_window_disabled)
        self.assertFalse(state.encoder_window_active)
        self.assertEqual(
            processor.max_item_bytes("en", encoder_window_disabled=True), 4
        )
        state.audio.append_pcm(bytes(2))
        _process(processor, state)
        self.assertIsNone(_window_kwargs(manager.requests[2]))

        abort = {"type": "abort", "message": "abort", "status_code": 500}
        manager, processor, state = _processor(
            ["partial", "partial"], encoder_window=_policy()
        )
        manager._finish_reasons = [abort, abort]
        _activate(state)
        state.audio.append_pcm(bytes(4))
        with self.assertLogs(level="WARNING"):
            self.assertEqual(_process(processor, state), "")
        state.audio.append_pcm(bytes(4))
        with (
            self.assertLogs(level="WARNING"),
            self.assertRaisesRegex(RuntimeError, "failed repeatedly"),
        ):
            _process(processor, state)

    def test_final_windowed_failures_raise(self):
        for transcripts, reasons, message in (
            ([None], ["stop"], "no response"),
            (["x"], ["length"], "max_new_tokens"),
            (
                ["x"],
                [{"type": "abort", "message": "aborted", "status_code": 500}],
                "aborted",
            ),
        ):
            with self.subTest(message):
                manager, processor, state = _processor(
                    transcripts, encoder_window=_policy()
                )
                manager._finish_reasons = reasons
                _activate(state)
                state.audio.append_pcm(bytes(4))
                with self.assertRaisesRegex(RuntimeError, message):
                    _process(processor, state, is_last=True)


class TestSessionGuards(_RuntimeConfigTestCase):
    def _update(self, connection, **transcription):
        event = SessionUpdateEvent(
            type="session.update",
            session=TranscriptionSessionConfig(
                type="transcription",
                audio=TranscriptionSessionAudio(
                    input=TranscriptionSessionAudioInput(
                        transcription=AudioTranscription(**transcription)
                    )
                ),
            ),
        )
        return _run(connection._on_session_update(event))

    def test_language_and_model_are_frozen_while_an_item_is_active(self):
        _, connection = _connection(["x"], encoder_window=_policy())
        connection.asr_state.audio.append_pcm(bytes(2))
        self._update(connection, language="zh")
        self.assertEqual(_sent_errors(connection)[-1]["code"], "invalid_state")
        self.assertEqual(connection.config.language, "en")

        # Repeating the current language is a no-op patch; a new item may change it.
        connection.websocket.send_text.reset_mock()
        self._update(connection, language="en")
        self.assertFalse(_sent_errors(connection))
        connection._reset_inference_state()
        self._update(connection, language="zh")
        self.assertEqual(connection.config.language, "zh")

    def test_large_append_recheck_stops_a_fallen_back_item(self):
        manager = _MockTokenizerManager([None] * 4 + ["one"] * 40)
        _, connection = _connection(
            None, manager=manager, encoder_window=_policy(2.0), max_buffer_seconds=120
        )
        with self.assertLogs(level="WARNING"):
            self.assertTrue(_append_seconds(connection, 90))
        self.assertTrue(connection.asr_state.encoder_window_disabled)
        connection.websocket.close.assert_awaited_with(code=1009)
        self.assertLessEqual(len(manager.requests), 4)


class TestSessionLifecycle(_RuntimeConfigTestCase):
    async def _waiting_session(self, mode):
        started = asyncio.Event()
        cleaned = asyncio.Event()

        class WaitingManager(_MockTokenizerManager):
            async def generate_request(self, request, raw_request=None):
                self.requests.append(request)
                started.set()
                try:
                    await asyncio.Event().wait()
                    yield {}  # async generator; this backend never completes
                finally:
                    await asyncio.sleep(0)
                    cleaned.set()

        manager = WaitingManager("unused")
        _, connection = _connection(None, manager=manager)
        incoming = asyncio.Queue()
        incoming.put_nowait({"type": "websocket.connect"})
        sent = []
        receiving = 0

        async def receive():
            nonlocal receiving
            receiving += 1
            self.assertEqual(receiving, 1, "only one task may receive from ASGI")
            try:
                message = await incoming.get()
                if isinstance(message, Exception):
                    raise message
                return message
            finally:
                receiving -= 1

        async def send(message):
            sent.append(message)

        websocket = WebSocket(
            {"type": "websocket", "path": "/v1/realtime", "headers": []},
            receive,
            send,
        )
        connection.websocket = websocket
        semaphore = asyncio.Semaphore(1)
        before = asyncio.all_tasks()
        with (
            patch(
                "sglang.srt.entrypoints.openai.realtime.handler.RealtimeConnection",
                return_value=connection,
            ),
            # The synthetic adapter runs at 1 Hz rather than a real audio rate.
            patch(
                "sglang.srt.entrypoints.openai.realtime.session.SUPPORTED_INPUT_SAMPLE_RATES",
                (1,),
            ),
        ):
            task = asyncio.create_task(
                handle_realtime_transcription(
                    websocket, manager, connection.adapter, _server_args(), semaphore
                )
            )
            try:
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
                await asyncio.wait_for(started.wait(), timeout=2)
                self.assertTrue(semaphore.locked())
                if mode == "cancel":
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                else:
                    if mode == "overflow":
                        # Many tiny frames must be bounded as well as large audio.
                        for _ in range(32):
                            incoming.put_nowait(
                                {"type": "websocket.receive", "text": ""}
                            )
                    elif mode == "receiver_error":
                        incoming.put_nowait(RuntimeError("synthetic receive failure"))
                    else:
                        # A queued ordinary event must not hide a later disconnect
                        # or mutate the active item while inference is waiting.
                        incoming.put_nowait(
                            {
                                "type": "websocket.receive",
                                "text": '{"type":"input_audio_buffer.clear"}',
                            }
                        )
                        incoming.put_nowait(
                            {"type": "websocket.disconnect", "code": 1000}
                        )
                    await asyncio.wait_for(task, timeout=2)

                self.assertTrue(cleaned.is_set())
                self.assertFalse(semaphore.locked())
                self.assertEqual(receiving, 0)
                self.assertEqual(len(manager.requests), 1)
                self.assertEqual(connection.asr_state.audio.received_bytes, 4)
                self.assertFalse(asyncio.all_tasks() - before)
                if mode == "overflow":
                    self.assertTrue(
                        any(
                            message["type"] == "websocket.close"
                            and message["code"] == 1009
                            for message in sent
                        )
                    )
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    def test_disconnect_cancels_inference_and_releases_session(self):
        _run(self._waiting_session("disconnect"))

    def test_parent_cancellation_waits_for_inference_cleanup(self):
        _run(self._waiting_session("cancel"))

    def test_pending_input_overflow_cancels_inference(self):
        _run(self._waiting_session("overflow"))

    def test_receiver_failure_cancels_inference(self):
        with self.assertLogs(level="ERROR"):
            _run(self._waiting_session("receiver_error"))

    def test_events_remain_serial_and_fatal_dispatch_stops_receiver(self):
        async def run():
            _, connection = _connection(None)
            incoming = asyncio.Queue()
            connection.websocket.receive = incoming.get
            started = asyncio.Event()
            release = asyncio.Event()
            seen = []

            async def dispatch(event):
                seen.append(connection._current_client_event_id)
                if len(seen) == 1:
                    started.set()
                    await release.wait()
                    self.assertEqual(connection._current_client_event_id, "first")
                return len(seen) == 2

            connection._dispatch = dispatch
            before = asyncio.all_tasks()
            task = asyncio.create_task(connection._run_loop())
            try:
                for event_id in ("first", "second"):
                    incoming.put_nowait(
                        {
                            "type": "websocket.receive",
                            "text": json.dumps(
                                {
                                    "type": "input_audio_buffer.clear",
                                    "event_id": event_id,
                                }
                            ),
                        }
                    )
                await asyncio.wait_for(started.wait(), timeout=2)
                self.assertEqual(seen, ["first"])
                release.set()
                await asyncio.wait_for(task, timeout=2)
                self.assertEqual(seen, ["first", "second"])
                self.assertFalse(asyncio.all_tasks() - before)
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        _run(run())

    def test_dispatch_exception_stops_receiver(self):
        async def run():
            _, connection = _connection(None)
            incoming = asyncio.Queue()
            incoming.put_nowait(
                {
                    "type": "websocket.receive",
                    "text": '{"type":"input_audio_buffer.clear"}',
                }
            )
            connection.websocket.receive = incoming.get
            connection._dispatch = AsyncMock(
                side_effect=RuntimeError("dispatch failed")
            )
            before = asyncio.all_tasks()
            with self.assertRaisesRegex(RuntimeError, "dispatch failed"):
                await asyncio.wait_for(connection._run_loop(), timeout=2)
            self.assertFalse(asyncio.all_tasks() - before)

        _run(run())

    def test_pending_budget_accepts_high_sample_rate_audio(self):
        async def run():
            _, connection = _connection(None)
            # One second at the model rate must also accept one second of
            # base64 input at the supported 48 kHz rate, plus its JSON envelope.
            connection.model_sample_rate = 16000
            connection.asr_processor.max_buffer_bytes = 16000 * 2
            connection._dispatch = AsyncMock(return_value=True)
            incoming = asyncio.Queue()
            incoming.put_nowait(
                {
                    "type": "websocket.receive",
                    "text": json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(bytes(48000 * 2)).decode(),
                        }
                    ),
                }
            )
            connection.websocket.receive = incoming.get
            await asyncio.wait_for(connection._run_loop(), timeout=2)
            connection._dispatch.assert_awaited_once()
            connection.websocket.close.assert_not_awaited()

        _run(run())

    def test_ready_event_backlog_does_not_starve_disconnect(self):
        async def run():
            _, connection = _connection(None)
            incoming = asyncio.Queue()
            connection.websocket.receive = incoming.get
            started = asyncio.Event()
            release = asyncio.Event()
            seen = []

            async def dispatch(event):
                seen.append(event)
                if len(seen) == 1:
                    started.set()
                    await release.wait()
                return False

            connection._dispatch = dispatch
            for _ in range(12):
                incoming.put_nowait(
                    {
                        "type": "websocket.receive",
                        "text": '{"type":"input_audio_buffer.clear"}',
                    }
                )
            before = asyncio.all_tasks()
            task = asyncio.create_task(connection._run_loop())
            try:
                async with asyncio.timeout(2):
                    await started.wait()
                    while not incoming.empty():
                        await asyncio.sleep(0)
                    # Wake the consumer first with a full backlog, then wake
                    # the receiver. Short handlers must yield to disconnect.
                    release.set()
                    incoming.put_nowait({"type": "websocket.disconnect", "code": 1000})
                    await task
                self.assertLess(len(seen), 12)
                self.assertFalse(asyncio.all_tasks() - before)
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        _run(run())


class _StreamingTokenizerManager(_MockTokenizerManager):
    """Yields decoder snapshots per request; each entry in ``scripts`` is one
    request's list of chunks. A dict entry is a terminal finish reason."""

    def __init__(self, scripts, *, incremental=False):
        super().__init__("unused")
        self._scripts = [list(script) for script in scripts]
        get_context().override(
            "realtime_asr_test", incremental_streaming_output=incremental
        )

    def generate_request(self, adapted_request, raw_request=None, **kwargs):
        self.requests.append(adapted_request)
        script = self._scripts.pop(0)
        incremental = get_serving().incremental_streaming_output

        async def generate():
            last_text = ""
            for index, chunk in enumerate(script):
                if isinstance(chunk, dict):
                    # A terminal frame repeats the cumulative text unless the
                    # server streams increments, where it carries no new text.
                    yield {
                        "text": "" if incremental else last_text,
                        "meta_info": {"finish_reason": chunk},
                    }
                    return
                last_text = chunk
                last = index == len(script) - 1
                yield {
                    "text": chunk,
                    "meta_info": {"finish_reason": {"type": "stop"} if last else None},
                }

        return generate()


def _streaming_processor(scripts, *, encoder_window=None, incremental=False):
    manager = _StreamingTokenizerManager(scripts, incremental=incremental)
    processor = RealtimeASRProcessor(
        manager,
        _adapter(),
        _server_args(120, decoder_streaming=True),
        encoder_window=encoder_window,
        session_id="sess_stream",
    )
    return manager, processor, processor.create_state()


def _process_streaming(processor, state, *, is_last=False):
    published = []

    async def collect(text):
        published.append(text)

    remainder = _run(
        processor.process(
            state,
            is_last=is_last,
            language="en",
            sampling_params={},
            on_transcript_delta=collect,
        )
    )
    return published, remainder


class TestDecoderStreaming(_RuntimeConfigTestCase):
    def test_backend_snapshots_are_reconstructed_and_model_prefixes_hidden(self):
        async def run(manager, adapter, decoder_prefix=""):
            updates = []

            async def collect(text):
                updates.append(text)

            result = await generate_asr_transcript(
                tokenizer_manager=manager,
                adapter=adapter,
                audio_data=np.zeros(4, dtype=np.float32),
                sampling_params={},
                prompt="PROMPT:",
                decoder_prefix=decoder_prefix,
                on_update=collect,
            )
            return updates, result

        updates, result = _run(
            run(
                _StreamingTokenizerManager(
                    [["hello", "hello world", "hello world again"]]
                ),
                _adapter(),
            )
        )
        self.assertEqual(updates, ["hello", "hello world", "hello world again"])
        self.assertEqual(result.text, "hello world again")

        updates, result = _run(
            run(
                _StreamingTokenizerManager(
                    [["language en<asr", "_text>Hello", " world"]], incremental=True
                ),
                Qwen3ASRAdapter(),
            )
        )
        self.assertEqual(updates, ["Hello", "Hello world"])
        self.assertEqual(result.text, "Hello world")
        self.assertEqual(result.finish_reason, "stop")

        # A prompt that already ends with transcript text is continued without
        # the "language xx<asr_text>" header, so its snapshots stream as they
        # are instead of being held back for a delimiter that never comes.
        manager = _StreamingTokenizerManager([[" world", " world again"]])
        updates, result = _run(run(manager, Qwen3ASRAdapter(), decoder_prefix="Hello"))
        self.assertEqual(manager.requests[0].text, "PROMPT:Hello")
        self.assertEqual(updates, [" world", " world again"])
        self.assertEqual(result.text, " world again")

    def test_flag_off_keeps_requests_unstreamed(self):
        manager, processor, state = _processor(["one two"])
        state.audio.append_pcm(bytes(4))
        published, remainder = _process_streaming(processor, state)
        self.assertEqual((published, remainder), ([], "one"))
        self.assertFalse(manager.requests[0].stream)

        # The final decode of an item is never streamed either.
        manager, processor, state = _streaming_processor([["one two"]])
        state.audio.append_pcm(bytes(4))
        published, remainder = _process_streaming(processor, state, is_last=True)
        self.assertEqual((published, remainder), ([], "one two"))
        self.assertFalse(manager.requests[0].stream)

    def test_cumulative_previews_hold_back_the_last_unit(self):
        manager, processor, state = _streaming_processor(
            [["one", "one two", "one two three", "one two three four"]]
        )
        state.audio.append_pcm(bytes(4))
        published, remainder = _process_streaming(processor, state)
        self.assertTrue(manager.requests[0].stream)
        self.assertEqual(published, ["one", "two"])
        self.assertEqual(remainder, "three")
        self.assertEqual(state.transcript.emitted_text, "one two three")

        # Without streaming the same decode yields the same total delta.
        _, plain_processor, plain_state = _processor(["one two three four"])
        plain_state.audio.append_pcm(bytes(4))
        self.assertEqual(_process(plain_processor, plain_state), "one two three")

    def test_previews_never_retract_and_a_rewrite_is_reconciled_keep_biased(self):
        _, processor, state = _streaming_processor(
            [["one two three", "one four five six"]]
        )
        state.audio.append_pcm(bytes(4))
        with self.assertNoLogs(level="WARNING"):
            published, remainder = _process_streaming(processor, state)
        self.assertEqual(published, ["one", "four"])
        self.assertEqual(remainder, "five")

        _, processor, state = _streaming_processor(
            [["one two three", "one two three x", "one five six"]]
        )
        state.audio.append_pcm(bytes(4))
        with self.assertLogs(level="WARNING"):
            published, remainder = _process_streaming(processor, state)
        self.assertEqual(published, ["one", "two"])
        self.assertEqual(remainder, "five")
        # The state follows the wire, so later decoder prefixes match what
        # the client has seen.
        self.assertEqual(state.transcript.emitted_text, "one two five")

        # Published units must not remain pending; an additional occurrence
        # or text after a revision is still kept, not deduplicated by content.
        for final, pending in (
            ("four five nine seven", "nine seven"),
            ("four five five seven", "five seven"),
            ("four nine five seven", "nine five seven"),
        ):
            with self.subTest(final=final):
                _, processor, state = _streaming_processor(
                    [["four five", "four five six", "four five six seven", final]],
                    encoder_window=_policy(),
                )
                _activate(state)
                state.decoder_suffix.pending = "four five six"
                state.audio.append_pcm(bytes(4))
                with self.assertLogs(level="WARNING"):
                    published, remainder = _process_streaming(processor, state)
                self.assertEqual(published, ["four", "five"])
                self.assertEqual(remainder, "")
                emitted = "one two three four five"
                self.assertEqual(state.decoder_suffix.emitted_text, emitted)
                self.assertEqual(state.decoder_suffix.pending, pending)
                self.assertFalse(state.has_new_audio)
                self.assertEqual(processor.flush_pending_transcript(state), pending)
                self.assertEqual(
                    state.decoder_suffix.emitted_text, f"{emitted} {pending}"
                )
                self.assertEqual(processor.flush_pending_transcript(state), "")

    def test_windowed_previews_wait_out_the_prefix_replay(self):
        manager, processor, state = _streaming_processor(
            [
                [
                    "one",
                    "one two",
                    "one two three four",
                    "one two three four five",
                    "one two three four five six",
                ]
            ],
            encoder_window=_policy(),
        )
        _activate(state)
        state.decoder_suffix.pending = "four five"
        state.audio.append_pcm(bytes(4))
        published, remainder = _process_streaming(processor, state)
        self.assertEqual(manager.requests[0].text, "PROMPT:one two three")
        self.assertEqual(published, ["four"])
        self.assertEqual(remainder, "")
        self.assertEqual(state.decoder_suffix.emitted_text, "one two three four")
        self.assertEqual(state.decoder_suffix.pending, "five six")

    def test_windowed_failures_after_streaming_fail_the_step(self):
        # A terminal-only length response must not publish its first delta
        # before the processor can choose to retry the uncovered audio.
        for incremental in (False, True):
            with self.subTest(terminal_only=True, incremental=incremental):
                manager = _MockTokenizerManager(
                    ["four five six"], finish_reasons=["length"]
                )
                get_context().override(
                    "realtime_asr_test",
                    incremental_streaming_output=incremental,
                )
                processor = RealtimeASRProcessor(
                    manager,
                    _adapter(),
                    _server_args(120, decoder_streaming=True),
                    encoder_window=_policy(),
                )
                state = processor.create_state()
                _activate(state)
                state.decoder_suffix.pending = "four five six"
                state.audio.append_pcm(bytes(4))
                with self.assertLogs(level="WARNING"):
                    self.assertEqual(_process_streaming(processor, state), ([], ""))
                self.assertEqual(state.consecutive_window_failures, 1)
                self.assertEqual(state.audio.last_attempted_offset_bytes, 4)
                self.assertEqual(state.audio.last_processed_offset_bytes, 0)
                self.assertEqual(state.decoder_suffix.emitted_text, "one two three")
                self.assertEqual(state.decoder_suffix.pending, "four five six")

        abort = {"type": "abort", "message": "aborted", "status_code": 500}
        for name, script in (
            ("abort", ["four five", "four five six", "four five six seven", abort]),
            (
                "length",
                [
                    "four five",
                    "four five six",
                    "four five six seven",
                    {"type": "length"},
                ],
            ),
        ):
            with self.subTest(name):
                _, processor, state = _streaming_processor(
                    [script], encoder_window=_policy()
                )
                _activate(state)
                state.decoder_suffix.pending = "four five six"
                state.audio.append_pcm(bytes(4))
                published = []

                async def collect(text):
                    published.append(text)

                # Text already on the wire can be neither retracted nor safely
                # re-decoded: the step fails instead of marking the audio
                # processed (the previous bug) or retrying it (duplicates).
                with self.assertRaises(RuntimeError):
                    _run(
                        processor.process(
                            state,
                            is_last=False,
                            language="en",
                            sampling_params={},
                            on_transcript_delta=collect,
                        )
                    )
                self.assertEqual(published, ["four", "five"])
                self.assertEqual(state.audio.last_processed_offset_bytes, 0)
                self.assertEqual(state.consecutive_window_failures, 0)
                self.assertEqual(state.decoder_suffix.emitted_text, "one two three")
                self.assertEqual(state.decoder_suffix.pending, "four five six")


class TestHttpChunkHelper(_RuntimeConfigTestCase):
    def test_process_asr_chunk_keeps_its_contract(self):
        state = StreamingASRState(
            chunk_size_sec=2.0, unfixed_chunk_num=2, unfixed_token_num=1
        )
        manager = _MockTokenizerManager([None, "one two", "one two three"])

        def run(is_last):
            return _run(
                process_asr_chunk(
                    tokenizer_manager=manager,
                    adapter=_adapter(),
                    state=state,
                    audio_data=b"wav",
                    sampling_params={},
                    is_last=is_last,
                )
            )

        self.assertEqual(run(False), "")
        self.assertEqual(state.chunk_index, 0)
        self.assertEqual(run(False), "one")
        self.assertEqual(run(True), "two three")
        self.assertEqual(manager.requests[-1].audio_data, b"wav")


if __name__ == "__main__":
    unittest.main()
