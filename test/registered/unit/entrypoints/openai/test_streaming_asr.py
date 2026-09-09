"""Transcript rollback and the real TokenizerManager ASR response contract."""

# ruff: noqa: E402 -- CPU kernel stubs must precede runtime imports.

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.openai.realtime.asr_processor import RealtimeASRProcessor
from sglang.srt.entrypoints.openai.realtime.decoder_suffix import DecoderSuffixState
from sglang.srt.entrypoints.openai.realtime.encoder_window_policy import (
    ResolvedEncoderWindowPolicy,
)
from sglang.srt.entrypoints.openai.streaming_asr import (
    ASRBackendAborted,
    StreamingASRState,
    common_unit_prefix,
    generate_asr_transcript,
    join_text,
    join_units,
    needs_space,
    normalize_unit,
    split_units,
)
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    RealtimeEncoderWindowPolicy,
)
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.multimodal.encoder_window import EncoderWindowConfig
from sglang.srt.runtime_context import get_context, get_server_args
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_override

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_ENGLISH_SAMPLES = [
    "",
    "hello",
    "  hello   world  ",
    "Hello, world! It's 5 o'clock (really) - yes.",
    "a b c d e f g",
    "ünïcödé wörds Ärger",
    "한국어 단어 는 띄어쓰기 를 씁니다",
    "hyphen-ated tokens... and ellipses",
]


class _LegacyState:
    """str.split based rollback as shipped before unit-level reconciliation."""

    def __init__(self, unfixed_token_num):
        self.unfixed_token_num = unfixed_token_num
        self.confirmed_text = ""
        self.emitted_text = ""
        self.full_transcript = ""
        self.chunk_index = 0

    def _record_emit(self, delta):
        if delta:
            self.emitted_text = (
                f"{self.emitted_text} {delta}".strip() if self.emitted_text else delta
            )
        return delta

    def update(self, new_transcript):
        old_confirmed = self.confirmed_text
        words = new_transcript.split()
        if len(words) > self.unfixed_token_num:
            self.confirmed_text = " ".join(words[: -self.unfixed_token_num])
        else:
            self.confirmed_text = ""
        self.full_transcript = new_transcript
        self.chunk_index += 1
        if self.confirmed_text.startswith(old_confirmed):
            return self._record_emit(self.confirmed_text[len(old_confirmed) :].strip())
        old_words = old_confirmed.split()
        new_words = self.confirmed_text.split()
        common_count = 0
        for ow, nw in zip(old_words, new_words):
            if ow != nw:
                break
            common_count += 1
        return self._record_emit(" ".join(new_words[common_count:]))

    def finalize(self):
        confirmed_words = self.confirmed_text.split()
        all_words = self.full_transcript.split()
        common_count = 0
        for cw, aw in zip(confirmed_words, all_words):
            if cw != aw:
                break
            common_count += 1
        self.confirmed_text = self.full_transcript
        if common_count == 0 and confirmed_words and all_words:
            return self._record_emit(self.full_transcript)
        return self._record_emit(" ".join(all_words[common_count:]))


def _state(unfixed_token_num=2):
    return StreamingASRState(
        chunk_size_sec=2.0, unfixed_chunk_num=2, unfixed_token_num=unfixed_token_num
    )


class TestSplitAndJoin(CustomTestCase):
    def test_whitespace_scripts_match_str_split_and_join(self):
        for text in _ENGLISH_SAMPLES:
            with self.subTest(text=text):
                self.assertEqual(split_units(text), text.split())
                self.assertEqual(join_units(text.split()), " ".join(text.split()))

    def test_cjk_splits_per_character_and_joins_without_spaces(self):
        self.assertEqual(
            split_units("hello 你好，world"), ["hello", "你", "好", "，", "world"]
        )
        self.assertEqual(split_units("你好世界"), ["你", "好", "世", "界"])
        self.assertEqual(split_units("Python编程"), ["Python", "编", "程"])
        self.assertEqual(
            split_units("こんにちは。"), ["こ", "ん", "に", "ち", "は", "。"]
        )
        # Thai is whitespace-delimited here, as on the pre-existing paths.
        self.assertEqual(split_units("สวัสดี ครับ"), ["สวัสดี", "ครับ"])

        self.assertEqual(join_units(split_units("你好世界")), "你好世界")
        self.assertEqual(join_units(split_units("你好，世界。")), "你好，世界。")
        self.assertEqual(join_units(split_units("สวัสดีครับ")), "สวัสดีครับ")
        self.assertEqual(join_units(split_units("Python编程")), "Python 编程")
        # A Latin word after CJK punctuation keeps the boundary space that the
        # wire delta joiner (needs_space) has always inserted there.
        self.assertEqual(join_units(["hello", "，", "world"]), "hello， world")
        self.assertEqual(join_units(split_units("hello， world")), "hello， world")
        self.assertEqual(join_units(["", "a", "", "b"]), "a b")
        self.assertEqual(join_text("", "x"), "x")
        self.assertEqual(join_text("x", ""), "x")
        self.assertEqual(join_text("one two", "three"), "one two three")
        self.assertEqual(join_text("今", "天"), "今天")

    def test_needs_space_is_unchanged_for_wire_deltas(self):
        self.assertTrue(needs_space("hello", "world"))
        self.assertFalse(needs_space("hello", ","))
        self.assertFalse(needs_space("(", "x"))
        self.assertFalse(needs_space("你", "好"))
        self.assertTrue(needs_space("hello", "你"))
        self.assertFalse(needs_space("", "x"))

    def test_normalize_and_common_prefix(self):
        self.assertEqual(normalize_unit("Hello,"), "hello")
        self.assertEqual(normalize_unit("（Ｈｅｌｌｏ）"), "hello")
        self.assertEqual(normalize_unit("..."), "...")
        self.assertEqual(common_unit_prefix(["a", "b", "c"], ["a", "b", "d"]), 2)
        self.assertEqual(common_unit_prefix(["A,", "b"], ["a", "B."]), 0)
        self.assertEqual(
            common_unit_prefix(["A,", "b"], ["a", "B."], normalized=True), 2
        )


class TestUnitLevelRollback(CustomTestCase):
    def test_english_rollback_is_byte_identical_to_the_legacy_state(self):
        sequences = [
            ["one", "one two", "one two three", "one two three four five"],
            ["one two three", "one two thirty", "one two thirty three four"],
            [
                "hello, world",
                "hello world!",
                "hello world! how are",
                "hello world! how are you",
            ],
            ["a b c d", "x y z", "x y z w"],
            ["", "one", "", "one two three"],
        ]
        for sequence in sequences:
            for unfixed in (1, 2, 5):
                with self.subTest(sequence=sequence, unfixed=unfixed):
                    new, legacy = _state(unfixed), _LegacyState(unfixed)
                    for transcript in sequence[:-1]:
                        self.assertEqual(
                            new.update(transcript), legacy.update(transcript)
                        )
                    new.full_transcript = legacy.full_transcript = sequence[-1]
                    self.assertEqual(new.finalize(), legacy.finalize())
                    self.assertEqual(new.emitted_text, legacy.emitted_text)
                    self.assertEqual(new.confirmed_text, legacy.confirmed_text)

    def test_cjk_rollback_streams_units_every_chunk(self):
        state = _state(unfixed_token_num=2)

        self.assertEqual(state.update("你好世界"), "你好")
        self.assertEqual(state.update("你好世界今天"), "世界")
        self.assertEqual(state.update("你好世界今天天气"), "今天")
        self.assertEqual(state.emitted_text, "你好世界今天")
        # A revision inside the confirmed prefix re-emits from the first
        # changed unit, exactly like the word-level rule does for English.
        self.assertEqual(state.update("你好世间今天天气很好"), "间今天天气")
        state.full_transcript = "你好世间今天天气很好啊"
        self.assertEqual(state.finalize(), "很好啊")
        self.assertEqual(state.emitted_text, "你好世界今天间今天天气很好啊")

        # Latin runs inside CJK text stay whole units.
        state = _state(unfixed_token_num=1)
        self.assertEqual(state.update("我用 Python 编程"), "我用 Python 编")
        self.assertEqual(state.update("我用 Python 编程，很好"), "程，很")
        self.assertEqual(state.get_prefix_text(), "我用 Python 编程，很")


class _ReaderLock:
    def __init__(self):
        self.held = False

    async def __aenter__(self):
        self.held = True

    async def __aexit__(self, *exc):
        self.held = False


class TestASRBackendContract(CustomTestCase):
    """Synthetic scheduler transport; real normalization, state, waiter, and cleanup.

    In particular, non-stream errors go through the manager's actual
    _handle_abort_finish_reason instead of a scripted generator yielding them.
    """

    def _run(self, scenario):
        async def bounded():
            async with asyncio.timeout(5):
                await scenario

        asyncio.run(bounded())

    def setUp(self):
        super().setUp()
        enter_override(
            self,
            get_context().override_server_args(
                asr_max_buffer_seconds=120,
                enable_asr_decoder_streaming=False,
                incremental_streaming_output=False,
                speculative_algorithm=None,
                tokenizer_worker_num=1,
            ),
        )
        self.enterContext(
            patch(
                "sglang.srt.managers.tokenizer_manager.wrap_shm_features",
                side_effect=lambda obj: obj,
            )
        )
        self.adapter = SimpleNamespace(
            prompt_template="PROMPT:",
            model_sample_rate=1,
            postprocess_text=lambda text: text,
            postprocess_streaming_text=lambda text, continuation=False: text,
            chunked_streaming_config={
                "chunk_size_sec": 2.0,
                "unfixed_chunk_num": 2,
                "unfixed_token_num": 1,
            },
        )

    def _manager(self, *, abort_status=None, finished=False, waiting=False):
        tm = TokenizerManager.__new__(TokenizerManager)
        tm.rid_to_state = {}
        tm.encoder_dispatch_ready = {}
        tm.enable_trace = tm.enable_lora = tm.enable_metrics = False
        tm.incremental_streaming_output = False
        tm.disaggregation_mode = "none"
        tm.elastic_worker_count = 1
        tm.auto_create_handle_loop = Mock()
        tm._set_default_priority = Mock()
        tm.request_logger = Mock()
        tm.tokenizer = SimpleNamespace(
            encode=lambda text, **kwargs: text.split(),
            decode=lambda ids, **kwargs: " ".join(ids),
        )
        tm.is_pause = False
        tm.is_pause_cond = asyncio.Condition()
        tm.model_update_lock = SimpleNamespace(reader_lock=_ReaderLock())
        tm._validate_and_resolve_lora = AsyncMock()
        tm.cuda_vmm_feature_transport = Mock()
        tm.cuda_vmm_feature_transport.prepare_for_dispatch_async = AsyncMock(
            return_value=[]
        )
        tm.request_metrics_exporter_manager = Mock()
        tm.request_metrics_exporter_manager.exporter_enabled.return_value = False
        tm.dispatched = asyncio.Event()

        async def tokenize(obj):
            return SimpleNamespace(
                rid=obj.rid,
                mm_inputs=None,
                time_stats=tm.rid_to_state[obj.rid].time_stats,
                wrap_pickle_fields=Mock(),
            )

        tm._tokenize_one_request = tokenize

        def dispatch(obj):
            if isinstance(obj, AbortReq):
                return  # The real scheduler would later acknowledge cleanup.
            tm.dispatched.set()
            state = tm.rid_to_state[obj.rid]

            def respond():
                if abort_status is not None:
                    self._abort(tm, obj.rid, abort_status)
                elif not waiting:
                    state.out_list.append(
                        {
                            "text": "hello world tail",
                            "meta_info": {
                                "finish_reason": {"type": "stop"} if finished else None
                            },
                        }
                    )
                    state.finished = finished
                    if finished:
                        del tm.rid_to_state[obj.rid]
                    state.event.set()

            # The waiter binds ReqState before scheduler output can arrive.
            asyncio.get_running_loop().call_soon(respond)

        tm._dispatch_to_scheduler = Mock(side_effect=dispatch)
        return tm

    def _abort(self, tm, rid, status):
        reason = {"type": "abort", "message": "scheduler aborted"}
        if status is not None:
            reason["status_code"] = status
        tm._handle_abort_req(AbortReq(rid=rid, abort_all=False, finished_reason=reason))

    def _processor(self, tm, streaming):
        get_context().override(
            "asr_contract_test", enable_asr_decoder_streaming=streaming
        )
        policy = ResolvedEncoderWindowPolicy(
            config=EncoderWindowConfig(
                sample_rate=1,
                hop_length=2,
                n_fft=4,
                window_frames=8,
                window_samples=16,
                window_tokens=8,
                min_tail_samples=4,
                context_samples=4,
            ),
            policy=RealtimeEncoderWindowPolicy(
                min_audio_sec=0,
                max_audio_context_windows=2,
                supported_languages=("en",),
                decoder_prefix_max_tokens=3,
                decoder_prefix_holdback_units=0,
            ),
        )
        proc = RealtimeASRProcessor(
            tm, self.adapter, get_server_args(), encoder_window=policy
        )
        state = proc.create_state()
        state.decoder_suffix = DecoderSuffixState(
            emitted_text="old", pending="hello world tail"
        )
        state.audio.append_pcm(bytes(4))
        return proc, state

    async def _process(self, proc, state, *, is_last=False, callback=None):
        return await proc.process(
            state,
            is_last=is_last,
            language="en",
            sampling_params={},
            on_transcript_delta=callback or AsyncMock(),
        )

    async def _generate(self, tm, callback=None):
        return await generate_asr_transcript(
            tm, self.adapter, bytes(4), {}, "PROMPT:", on_update=callback
        )

    def _assert_closed(self, tm, *, aborted):
        self.assertFalse(tm.model_update_lock.reader_lock.held)
        aborts = [
            call.args[0]
            for call in tm._dispatch_to_scheduler.call_args_list
            if isinstance(call.args[0], AbortReq)
        ]
        self.assertEqual(len(aborts), int(aborted))

    def test_transient_abort_retains_audio_in_both_streaming_modes(self):
        async def run():
            for status in (500, 503):
                for streaming in (False, True):
                    with self.subTest(status=status, streaming=streaming):
                        tm = self._manager(abort_status=status)
                        proc, state = self._processor(tm, streaming)
                        self.assertEqual(await self._process(proc, state), "")
                        self.assertEqual(state.consecutive_window_failures, 1)
                        self.assertEqual(state.audio.last_attempted_offset_bytes, 4)
                        self.assertEqual(state.audio.last_processed_offset_bytes, 0)
                        self.assertEqual(bytes(state.audio.data), bytes(4))
                        self._assert_closed(tm, aborted=False)

        self._run(run())

    def test_bad_request_and_unclassified_aborts_are_not_retried(self):
        async def run():
            for status in (400, 429, None):
                for streaming in (False, True):
                    with self.subTest(status=status, streaming=streaming):
                        tm = self._manager(waiting=True)
                        proc, state = self._processor(tm, streaming)
                        task = asyncio.create_task(self._process(proc, state))
                        await tm.dispatched.wait()
                        rid = next(iter(tm.rid_to_state))
                        self._abort(tm, rid, status)
                        with self.assertRaises((ValueError, ASRBackendAborted)):
                            await task
                        self.assertEqual(state.consecutive_window_failures, 0)
                        self.assertEqual(state.audio.last_attempted_offset_bytes, 0)
                        self._assert_closed(tm, aborted=False)

        self._run(run())

    def test_tokenization_value_error_is_not_retried(self):
        async def run():
            for streaming in (False, True):
                with self.subTest(streaming=streaming):
                    tm = self._manager()
                    tm._tokenize_one_request = AsyncMock(
                        side_effect=ValueError("invalid input")
                    )
                    proc, state = self._processor(tm, streaming)
                    with self.assertRaisesRegex(ValueError, "invalid input"):
                        await self._process(proc, state)
                    self.assertEqual(state.consecutive_window_failures, 0)
                    self.assertFalse(tm.rid_to_state)
                    self._assert_closed(tm, aborted=False)

        self._run(run())

    def test_final_abort_and_abort_after_publication_fail_closed(self):
        async def run():
            for status in (500, 503):
                with self.subTest(status=status, phase="final"):
                    tm = self._manager(abort_status=status)
                    proc, state = self._processor(tm, False)
                    with self.assertRaises(ASRBackendAborted):
                        await self._process(proc, state, is_last=True)
                    self.assertEqual(state.consecutive_window_failures, 0)
                    self._assert_closed(tm, aborted=False)
                with self.subTest(status=status, phase="published"):
                    tm = self._manager()
                    proc, state = self._processor(tm, True)
                    published = []

                    async def publish(text):
                        published.append(text)
                        self._abort(tm, next(iter(tm.rid_to_state)), status)

                    with self.assertRaises(ASRBackendAborted):
                        await self._process(proc, state, callback=publish)
                    self.assertTrue(published)
                    self.assertEqual(state.consecutive_window_failures, 0)
                    self._assert_closed(tm, aborted=False)

        self._run(run())

    def test_normal_nonstream_return_releases_reader_before_return(self):
        async def run():
            tm = self._manager(finished=True)
            result = await self._generate(tm)
            self.assertEqual(result.text, "hello world tail")
            self.assertFalse(tm.rid_to_state)
            self._assert_closed(tm, aborted=False)

        self._run(run())

    def test_callback_failure_aborts_before_return_without_normalizing_it(self):
        async def run():
            for error in (RuntimeError("send failed"), HTTPException(503, "callback")):
                with self.subTest(error=type(error).__name__):
                    tm = self._manager()

                    async def callback(text):
                        raise error

                    with self.assertRaises(type(error)) as caught:
                        await self._generate(tm, callback)
                    self.assertIs(caught.exception, error)
                    self._assert_closed(tm, aborted=True)

        self._run(run())

    def test_cancellation_aborts_before_task_finishes(self):
        async def run():
            for phase in ("backend", "callback"):
                with self.subTest(phase=phase):
                    tm = self._manager(waiting=phase == "backend")
                    entered = asyncio.Event()

                    async def callback(text):
                        entered.set()
                        await asyncio.Event().wait()

                    task = asyncio.create_task(self._generate(tm, callback))
                    await (tm.dispatched if phase == "backend" else entered).wait()
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    self._assert_closed(tm, aborted=True)

        self._run(run())


if __name__ == "__main__":
    unittest.main()
