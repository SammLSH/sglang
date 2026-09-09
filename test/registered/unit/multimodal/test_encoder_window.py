"""Unit tests for sglang.srt.multimodal.encoder_window."""

import asyncio
import copy
import unittest
from types import SimpleNamespace

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that pull in sgl_kernel

import msgspec
import numpy as np
import torch
import uvloop
from transformers import WhisperFeatureExtractor

from sglang.srt.multimodal.encoder_window import (
    EncoderWindowCapability,
    EncoderWindowConfig,
    EncoderWindowMixin,
    EncoderWindowSpec,
    build_encoder_window_items,
    encoder_window_kwargs,
    resolve_encoder_window_config,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.executor import MultimodalProcessorExecutor
from sglang.srt.multimodal.processors.qwen3_asr import Qwen3ASRMultimodalProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

PLACEHOLDER = 99
PROMPT_IDS = [10, PLACEHOLDER, 11]


class _FakeExtractor:
    """Whisper-shaped extractor: one frame per hop, features carry the audio."""

    sampling_rate = 16
    hop_length = 2
    n_fft = 4
    dither = 0.0

    def __init__(self):
        self.calls = []

    def __call__(self, windows, **kwargs):
        self.calls.append(([len(window) for window in windows], kwargs))
        lengths = [len(window) // self.hop_length for window in windows]
        width = max(lengths)
        features = torch.zeros(len(windows), 2, width)
        masks = torch.zeros(len(windows), width, dtype=torch.long)
        for row, (window, length) in enumerate(zip(windows, lengths)):
            features[row, 0, :length] = torch.from_numpy(
                np.asarray(window[: length * self.hop_length : self.hop_length])
            )
            masks[row, :length] = 1
        return {"input_features": features, "attention_mask": masks}


class _FrameDroppingExtractor(_FakeExtractor):
    """Returns one frame fewer than the hop count predicts."""

    def __call__(self, windows, **kwargs):
        output = super().__call__(windows, **kwargs)
        return {
            "input_features": output["input_features"][:, :, :-1],
            "attention_mask": output["attention_mask"][:, :-1],
        }


class _FakeTokenizer:
    bos_token = None

    def __init__(self):
        self.calls = []

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        self.calls.append((text, add_special_tokens))
        return SimpleNamespace(input_ids=torch.tensor([PROMPT_IDS]))


class _FakeHFProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.feature_extractor = _FakeExtractor()
        self.length_calls = []

    def _get_feat_extract_output_lengths(self, lengths):
        self.length_calls.append(list(lengths))
        return torch.as_tensor(lengths)

    def clear_calls(self):
        self.tokenizer.calls.clear()
        self.feature_extractor.calls.clear()
        self.length_calls.clear()


class _OffsetExtractor(_FakeExtractor):
    """Makes an explicit processor override visible in the emitted features."""

    def __call__(self, windows, **kwargs):
        output = super().__call__(windows, **kwargs)
        output["input_features"] += 100
        return output


class _FakeBase:
    """Stands in for BaseMultimodalProcessor below the mixin."""

    def __init__(self):
        self.base_calls = []
        self._tokenizer_auto_adds_specials = False

    def process_and_combine_mm_data(
        self, base_output, mm_tokens, processor=None, **kwargs
    ):
        self.base_calls.append((base_output, mm_tokens, processor, kwargs))
        return "base"

    def _resolve_processor(self, processor=None):
        return self._processor, self._tokenizer

    def _finalize_mm_items(self, mm_items, *, images):
        return mm_items


class _WindowProcessor(EncoderWindowMixin, _FakeBase):
    def __init__(
        self,
        spec=EncoderWindowSpec(window_frames=8, alignment_frames=4),
        audio_config=None,
        output_lengths=None,
    ):
        super().__init__()
        self.extractor = _FakeExtractor()
        self._processor = SimpleNamespace(
            feature_extractor=self.extractor,
            _get_feat_extract_output_lengths=output_lengths
            or (lambda lengths: torch.as_tensor(lengths)),
        )
        self._tokenizer = _FakeTokenizer()
        self.audio_config = audio_config or {}
        self.spec = spec

    def encoder_window_spec(self):
        return self.spec


def _resolve(processor):
    return resolve_encoder_window_config(processor)


def _build(processor, sample_count, config=None):
    return build_encoder_window_items(
        processor,
        samples=np.arange(sample_count, dtype=np.float32),
        input_ids=torch.tensor(PROMPT_IDS),
        placeholder_token_id=PLACEHOLDER,
        config=config or _resolve(processor),
    )


# Qwen3-ASR's audio encoder output length (three stride-2 convolutions over
# 100-frame blocks); copied from sglang/srt/models/qwen3_omni_moe.py.
def _qwen_output_lengths(lengths):
    lengths = torch.as_tensor(lengths)
    remainder = lengths % 100
    reduced = (remainder - 1) // 2 + 1
    return ((reduced - 1) // 2 + 1 - 1) // 2 + 1 + lengths // 100 * 13


class TestResolveEncoderWindowConfig(CustomTestCase):
    def test_resolves_geometry_from_spec_and_extractor(self):
        config = _resolve(_WindowProcessor())

        self.assertEqual(
            config,
            EncoderWindowConfig(
                sample_rate=16,
                hop_length=2,
                n_fft=4,
                window_frames=8,
                window_samples=16,
                window_tokens=8,
                min_tail_samples=4,
                context_samples=2,
            ),
        )
        self.assertEqual(config.window_seconds, 1.0)

    def test_rejects_invalid_geometry_and_extractors(self):
        cases = {
            "not a multiple": _WindowProcessor(EncoderWindowSpec(8, 3)),
            "must be positive": _WindowProcessor(EncoderWindowSpec(0, 4)),
            "deterministic": _WindowProcessor(),
            "preprocessing values for padding": _WindowProcessor(
                audio_config={"padding": "max_length"}
            ),
            "unpadded frames": _WindowProcessor(),
        }
        cases["deterministic"].extractor.dither = 0.1
        cases[
            "unpadded frames"
        ]._processor.feature_extractor = _FrameDroppingExtractor()
        for message, processor in cases.items():
            with self.subTest(message):
                with self.assertRaisesRegex(ValueError, message):
                    _resolve(processor)


class TestBuildEncoderWindowItems(CustomTestCase):
    def test_layout_complete_windows_then_tail(self):
        processor = _WindowProcessor()

        items, input_ids = _build(processor, 36)

        self.assertEqual(
            [item.offsets for item in items], [[(1, 8)], [(9, 16)], [(17, 18)]]
        )
        self.assertEqual(input_ids.tolist(), [10] + [PLACEHOLDER] * 18 + [11])
        self.assertEqual([int(item.feature.shape[-1]) for item in items], [8, 8, 2])
        self.assertEqual(
            [
                int(item.model_specific_data["feature_attention_mask"].sum())
                for item in items
            ],
            [8, 8, 2],
        )
        self.assertTrue(all(item.pad_value is not None for item in items))
        # Each window is extracted on its own so batch composition never
        # changes a complete window's bytes.
        self.assertEqual(
            [lengths for lengths, _ in processor.extractor.calls[2:]], [[16], [18], [6]]
        )

        # An exact multiple of the window has no tail.
        exact_items, input_ids = _build(processor, 32)
        self.assertEqual([item.offsets for item in exact_items], [[(1, 8)], [(9, 16)]])
        self.assertEqual(int((input_ids == PLACEHOLDER).sum()), 16)

        # A tail shorter than n_fft merges into the preceding window, which is
        # then extracted from 17 samples; with a real STFT front end its frames
        # differ from the 16-sample window (see TestWhisperFrontend).
        items, _ = _build(processor, 33)
        self.assertEqual([item.offsets for item in items], [[(1, 8)], [(9, 16)]])
        self.assertEqual(items[0].hash, exact_items[0].hash)
        self.assertEqual(int(items[1].feature.shape[-1]), 8)
        self.assertEqual(
            [lengths for lengths, _ in processor.extractor.calls[-2:]], [[16], [19]]
        )

    def test_rejects_unwindowable_audio_and_prompts(self):
        processor = _WindowProcessor()
        config = _resolve(processor)

        with self.assertRaisesRegex(ValueError, "cannot be windowed"):
            _build(processor, 3, config)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            _build(processor, 0, config)
        with self.assertRaisesRegex(ValueError, "mono"):
            build_encoder_window_items(
                processor,
                samples=np.zeros((2, 16), dtype=np.float32),
                input_ids=torch.tensor(PROMPT_IDS),
                placeholder_token_id=PLACEHOLDER,
                config=config,
            )
        for prompt in ([10, 11], [PLACEHOLDER, PLACEHOLDER]):
            with self.assertRaisesRegex(ValueError, "exactly one unexpanded"):
                build_encoder_window_items(
                    processor,
                    samples=np.arange(16, dtype=np.float32),
                    input_ids=torch.tensor(prompt),
                    placeholder_token_id=PLACEHOLDER,
                    config=config,
                )
        # Geometry that no longer matches the extractor's output is rejected.
        drifted = msgspec.structs.replace(config, window_tokens=7)
        with self.assertRaisesRegex(ValueError, "geometry changed"):
            _build(processor, 32, drifted)

    def test_item_identity_is_position_independent_and_mask_aware(self):
        processor = _WindowProcessor()
        short_items, _ = _build(processor, 36)
        long_items, _ = _build(processor, 48)

        self.assertEqual(short_items[0].hash, long_items[0].hash)
        self.assertEqual(short_items[1].hash, long_items[1].hash)
        self.assertNotEqual(short_items[0].hash, short_items[1].hash)
        # The tail is content-addressed too, so it changes as audio arrives.
        tail_36 = short_items[2].hash
        tail_40 = _build(processor, 40)[0][2].hash
        self.assertNotEqual(tail_36, tail_40)

        # Zero-padded frames can be byte-identical for different valid lengths,
        # so the mask is part of the identity.
        feature = torch.zeros(1, 2, 8)
        full = processor.make_encoder_window_item(
            feature, torch.ones(1, 8, dtype=torch.long), [(1, 8)]
        )
        partial_mask = torch.ones(1, 8, dtype=torch.long)
        partial_mask[0, 6:] = 0
        partial = processor.make_encoder_window_item(feature, partial_mask, [(1, 6)])
        self.assertNotEqual(full.hash, partial.hash)
        self.assertNotEqual(full.pad_value, partial.pad_value)

    def test_leading_context_keeps_window_identity_across_requests(self):
        processor = _WindowProcessor()
        config = _resolve(processor)
        full_items, _ = _build(processor, 36)

        # A rolling request starts at window 1 and carries one hop of audio
        # before it purely as extraction context.
        samples = np.arange(36, dtype=np.float32)[16 - config.context_samples :]
        rolling_items, input_ids = build_encoder_window_items(
            processor,
            samples=samples,
            input_ids=torch.tensor(PROMPT_IDS),
            placeholder_token_id=PLACEHOLDER,
            config=config,
            leading_context_samples=config.context_samples,
        )

        self.assertEqual(
            [item.offsets for item in rolling_items], [[(1, 8)], [(9, 10)]]
        )
        self.assertEqual(input_ids.tolist(), [10] + [PLACEHOLDER] * 10 + [11])
        self.assertEqual(rolling_items[0].hash, full_items[1].hash)
        self.assertEqual(rolling_items[1].hash, full_items[2].hash)
        self.assertTrue(torch.equal(rolling_items[0].feature, full_items[1].feature))
        with self.assertRaisesRegex(ValueError, "leading context"):
            build_encoder_window_items(
                processor,
                samples=samples,
                input_ids=torch.tensor(PROMPT_IDS),
                placeholder_token_id=PLACEHOLDER,
                config=config,
                leading_context_samples=1,
            )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            build_encoder_window_items(
                processor,
                samples=samples[: config.context_samples],
                input_ids=torch.tensor(PROMPT_IDS),
                placeholder_token_id=PLACEHOLDER,
                config=config,
                leading_context_samples=config.context_samples,
            )


class TestEncoderWindowMixin(CustomTestCase):
    def test_mixin_dispatches_only_windowed_requests(self):
        processor = _WindowProcessor(audio_config={"do_normalize": False})
        base_output = BaseMultiModalProcessorOutput(
            input_text="audio", audios=[np.arange(36, dtype=np.float32)]
        )

        items, input_ids, ret = processor.process_and_combine_mm_data(
            base_output,
            MultimodalSpecialTokens(audio_token_id=PLACEHOLDER),
            mm_processor_kwargs=encoder_window_kwargs(),
        )

        self.assertEqual(ret, {})
        self.assertEqual(processor.base_calls, [])
        self.assertEqual(
            [item.offsets for item in items], [[(1, 8)], [(9, 16)], [(17, 18)]]
        )
        self.assertEqual(input_ids.tolist(), [10] + [PLACEHOLDER] * 18 + [11])
        self.assertEqual(processor._tokenizer.calls, [("audio", True)])
        _, kwargs = processor.extractor.calls[-1]
        self.assertEqual(
            kwargs,
            {
                "do_normalize": False,
                "sampling_rate": 16,
                "return_attention_mask": True,
                "return_tensors": "pt",
                "truncation": False,
                "padding": "longest",
            },
        )

        # Every other request falls through to the base processor unchanged.
        plain = BaseMultiModalProcessorOutput(input_text="audio")
        sentinel = object()
        self.assertEqual(processor.process_and_combine_mm_data(plain, "tokens"), "base")
        self.assertEqual(
            processor.process_and_combine_mm_data(
                plain, "tokens", processor=sentinel, mm_processor_kwargs=None
            ),
            "base",
        )
        self.assertEqual(
            processor.process_and_combine_mm_data(
                plain, "tokens", mm_processor_kwargs={"other": 1}
            ),
            "base",
        )
        self.assertEqual(
            processor.base_calls,
            [
                (plain, "tokens", None, {}),
                (plain, "tokens", sentinel, {}),
                (plain, "tokens", None, {}),
            ],
        )

    def test_windowed_path_rejects_unusable_inputs(self):
        processor = _WindowProcessor()
        kwargs = encoder_window_kwargs()
        tokens = MultimodalSpecialTokens(audio_token_id=PLACEHOLDER)
        audio = np.arange(36, dtype=np.float32)
        for name, base_output in {
            "two audios": BaseMultiModalProcessorOutput(
                input_text="a", audios=[audio, audio]
            ),
            "no audio": BaseMultiModalProcessorOutput(input_text="a"),
            "audio plus image": BaseMultiModalProcessorOutput(
                input_text="a", audios=[audio], images=[object()]
            ),
        }.items():
            with (
                self.subTest(name),
                self.assertRaisesRegex(ValueError, "exactly one audio"),
            ):
                processor.process_and_combine_mm_data(
                    base_output, tokens, mm_processor_kwargs=kwargs
                )
        with self.assertRaisesRegex(ValueError, "raw audio"):
            processor.process_and_combine_mm_data(
                BaseMultiModalProcessorOutput(
                    input_text="a", audios=[{"precomputed": True}]
                ),
                tokens,
                mm_processor_kwargs=kwargs,
            )
        with self.assertRaisesRegex(ValueError, "placeholder token id"):
            processor.process_and_combine_mm_data(
                BaseMultiModalProcessorOutput(input_text="a", audios=[audio]),
                MultimodalSpecialTokens(),
                mm_processor_kwargs=kwargs,
            )

    def test_request_cannot_override_server_geometry(self):
        processor = _WindowProcessor()
        base_output = BaseMultiModalProcessorOutput(
            input_text="audio", audios=[np.arange(36, dtype=np.float32)]
        )
        tokens = MultimodalSpecialTokens(audio_token_id=PLACEHOLDER)

        # Geometry fields in the request are ignored; the server's own
        # resolved geometry (16-sample windows) splits the audio.
        items, _, _ = processor.process_and_combine_mm_data(
            base_output,
            tokens,
            mm_processor_kwargs={
                "encoder_window": {
                    "window_samples": 1,
                    "hop_length": 1,
                    "leading_context_samples": 0,
                }
            },
        )
        self.assertEqual(
            [item.offsets for item in items], [[(1, 8)], [(9, 16)], [(17, 18)]]
        )
        with self.assertRaisesRegex(ValueError, "leading context"):
            processor.process_and_combine_mm_data(
                base_output,
                tokens,
                mm_processor_kwargs={"encoder_window": {"leading_context_samples": 1}},
            )
        with self.assertRaisesRegex(ValueError, "must be a mapping"):
            processor.process_and_combine_mm_data(
                base_output, tokens, mm_processor_kwargs={"encoder_window": True}
            )


class TestQwen3ASRDeclaration(CustomTestCase):
    def test_processor_declares_encoder_native_window(self):
        self.assertLess(
            Qwen3ASRMultimodalProcessor.__mro__.index(EncoderWindowMixin),
            Qwen3ASRMultimodalProcessor.__mro__.index(
                Qwen3ASRMultimodalProcessor.__mro__[-2]
            ),
        )
        processor = Qwen3ASRMultimodalProcessor.__new__(Qwen3ASRMultimodalProcessor)
        processor.hf_config = SimpleNamespace(
            thinker_config=SimpleNamespace(
                audio_config=SimpleNamespace(n_window_infer=800, n_window=50)
            )
        )

        self.assertEqual(
            processor.encoder_window_spec(),
            EncoderWindowSpec(window_frames=800, alignment_frames=100),
        )
        self.assertIsInstance(processor, EncoderWindowCapability)


class TestEncoderWindowWorkerProcessor(CustomTestCase):
    def setUp(self):
        # Use the real Qwen3-ASR mixin/base dispatch and processor resolution,
        # while keeping model loading and item transport outside this CPU test.
        self.owner = Qwen3ASRMultimodalProcessor.__new__(Qwen3ASRMultimodalProcessor)
        self.source = _FakeHFProcessor()
        self.owner._processor = self.source
        self.owner._tokenizer = self.source.tokenizer
        self.owner._tokenizer_auto_adds_specials = False
        self.owner.audio_config = {"do_normalize": False}
        self.owner.hf_config = SimpleNamespace(
            thinker_config=SimpleNamespace(
                audio_config=SimpleNamespace(n_window_infer=8, n_window=2)
            )
        )
        self.owner._finalize_mm_items = lambda items, *, images: items
        self.config = self.owner.encoder_window_config()
        self.base_output = BaseMultiModalProcessorOutput(
            input_text="audio", audios=[np.arange(36, dtype=np.float32)]
        )
        self.tokens = MultimodalSpecialTokens(audio_token_id=PLACEHOLDER)
        self.expected = self.owner.process_and_combine_mm_data(
            self.base_output,
            self.tokens,
            mm_processor_kwargs=encoder_window_kwargs(),
        )
        self.source.clear_calls()

    def assert_source_untouched(self):
        self.assertIs(self.owner._processor, self.source)
        self.assertIs(self.owner.encoder_window_config(), self.config)
        self.assertEqual(self.source.tokenizer.calls, [])
        self.assertEqual(self.source.feature_extractor.calls, [])
        self.assertEqual(self.source.length_calls, [])

    def test_explicit_override_supplies_extractor_and_output_lengths(self):
        worker = copy.deepcopy(self.source)
        worker.feature_extractor = _OffsetExtractor()

        items, input_ids, ret = self.owner.process_and_combine_mm_data(
            self.base_output,
            self.tokens,
            mm_processor_kwargs=encoder_window_kwargs(),
            processor=worker,
        )

        self.assert_source_untouched()
        self.assertEqual(worker.tokenizer.calls, [("audio", True)])
        self.assertEqual(len(worker.feature_extractor.calls), 3)
        self.assertEqual(worker.length_calls, [[8], [8], [2]])
        self.assertTrue(torch.equal(input_ids, self.expected[1]))
        self.assertEqual(ret, {})
        for item, expected in zip(items, self.expected[0]):
            self.assertTrue(torch.equal(item.feature, expected.feature + 100))
            self.assertTrue(
                torch.equal(
                    item.feature_attention_mask, expected.feature_attention_mask
                )
            )
            self.assertEqual(item.offsets, expected.offsets)
            self.assertNotEqual(item.hash, expected.hash)

    def test_real_executor_uses_and_reuses_its_worker_clone(self):
        self._assert_executor_uses_worker_clone(cold_config=False)

    def test_cold_config_probe_uses_the_worker_clone(self):
        self.owner._encoder_window_config = None
        self._assert_executor_uses_worker_clone(cold_config=True)

    def _assert_executor_uses_worker_clone(self, *, cold_config):
        executor = MultimodalProcessorExecutor(lambda: self.source, max_workers=1)
        self.owner.mm_processor_executor = executor
        worker_processors = []
        process = self.owner.process_and_combine_mm_data

        def record_worker(*args, processor=None, **kwargs):
            worker_processors.append(processor)
            return process(*args, processor=processor, **kwargs)

        self.owner.process_and_combine_mm_data = record_worker

        async def run_requests():
            return [
                await asyncio.wait_for(
                    self.owner.process_and_combine_mm_data_async(
                        self.base_output,
                        self.tokens,
                        mm_processor_kwargs=encoder_window_kwargs(),
                    ),
                    timeout=10,
                )
                for _ in range(2)
            ]

        # TokenizerManager installs uvloop in production. Keep this loop local
        # so the regression does not change the policy for unrelated tests.
        loop = uvloop.new_event_loop()
        try:
            results = loop.run_until_complete(run_requests())
        finally:
            executor.shutdown()
            loop.close()

        if cold_config:
            self.assertEqual(self.owner._encoder_window_config, self.config)
            self.config = self.owner._encoder_window_config
        self.assert_source_untouched()
        self.assertEqual(len(worker_processors), 2)
        worker = worker_processors[0]
        self.assertIsNot(worker, self.source)
        self.assertIsNot(worker.feature_extractor, self.source.feature_extractor)
        self.assertIs(worker, worker_processors[1])
        self.assertEqual(worker.tokenizer.calls, [("audio", True)] * 2)
        self.assertEqual(len(worker.feature_extractor.calls), 6 + 2 * int(cold_config))
        if cold_config:
            self.assertEqual(
                [lengths for lengths, _ in worker.feature_extractor.calls[:2]],
                [[16], [18]],
            )
        probe_lengths = [[8], [8], [8]] if cold_config else []
        self.assertEqual(worker.length_calls, probe_lengths + [[8], [8], [2]] * 2)
        for items, input_ids, ret in results:
            self.assertTrue(torch.equal(input_ids, self.expected[1]))
            self.assertEqual(ret, {})
            for item, expected in zip(items, self.expected[0]):
                self.assertTrue(torch.equal(item.feature, expected.feature))
                self.assertTrue(
                    torch.equal(
                        item.feature_attention_mask, expected.feature_attention_mask
                    )
                )
                self.assertEqual(item.offsets, expected.offsets)
                self.assertEqual(item.hash, expected.hash)


class TestWhisperFrontend(CustomTestCase):
    """The real Whisper feature extractor with Qwen3-ASR-0.6B geometry."""

    @classmethod
    def setUpClass(cls):
        cls.extractor = WhisperFeatureExtractor(
            feature_size=128,
            sampling_rate=16000,
            hop_length=160,
            n_fft=400,
            chunk_length=30,
        )
        cls.processor = _WindowProcessor(
            EncoderWindowSpec(window_frames=800, alignment_frames=100),
            output_lengths=_qwen_output_lengths,
        )
        cls.processor._processor.feature_extractor = cls.extractor
        cls.config = _resolve(cls.processor)

    def test_window_geometry_matches_the_encoder_output_length(self):
        self.assertEqual(self.config.window_samples, 128000)
        self.assertEqual(self.config.window_seconds, 8.0)
        self.assertEqual(self.config.window_tokens, int(_qwen_output_lengths([800])[0]))
        self.assertEqual(self.config.window_tokens, 104)
        self.assertEqual(self.config.min_tail_samples, 400)
        # Half an STFT frame (200 samples) rounded up to whole hops.
        self.assertEqual(self.config.context_samples, 320)

    def test_rejects_padding_that_changes_windows_with_leading_context(self):
        for multiple in (512, 16000):
            with self.subTest(pad_to_multiple_of=multiple):
                processor = _WindowProcessor(
                    self.processor.spec,
                    audio_config={"pad_to_multiple_of": multiple},
                    output_lengths=_qwen_output_lengths,
                )
                processor._processor.feature_extractor = self.extractor
                # These settings pass the standalone-window probe; only the
                # leading STFT context exposes their incompatible padding.
                standalone = processor.extract_encoder_window_features(
                    [np.zeros(self.config.window_samples, dtype=np.float32)]
                )
                self.assertEqual(standalone.features[0].shape[-1], 800)
                self.assertEqual(int(standalone.masks[0].sum()), 800)
                self.assertEqual(standalone.token_counts, [104])
                with self.assertRaisesRegex(
                    ValueError, "with leading context; expected 800 unpadded frames"
                ):
                    _resolve(processor)

    def test_accepts_padding_compatible_with_complete_window_geometry(self):
        processor = _WindowProcessor(
            self.processor.spec,
            audio_config={"pad_to_multiple_of": self.config.hop_length},
            output_lengths=_qwen_output_lengths,
        )
        processor._processor.feature_extractor = self.extractor
        config = _resolve(processor)
        self.assertEqual(config, self.config)
        samples = (
            np.random.default_rng(23)
            .normal(0, 0.1, size=2 * config.window_samples)
            .astype(np.float32)
        )
        kwargs = dict(
            samples=samples,
            input_ids=torch.tensor(PROMPT_IDS),
            placeholder_token_id=PLACEHOLDER,
            config=config,
        )
        items, input_ids = build_encoder_window_items(processor, **kwargs)
        expected, expected_ids = build_encoder_window_items(self.processor, **kwargs)
        self.assertTrue(torch.equal(input_ids, expected_ids))
        self.assertEqual(len(items), 2)
        for item, reference in zip(items, expected):
            self.assertEqual(item.feature.shape[-1], 800)
            self.assertEqual(int(item.feature_attention_mask.sum()), 800)
            self.assertTrue(torch.equal(item.feature, reference.feature))
            self.assertEqual(item.offsets, reference.offsets)
            self.assertEqual(item.hash, reference.hash)

    def test_worker_clone_preserves_features_and_hashes(self):
        audio = (
            np.random.default_rng(11)
            .normal(0, 0.1, size=2 * self.config.window_samples + 4000)
            .astype(np.float32)
        )
        kwargs = dict(
            samples=audio,
            input_ids=torch.tensor(PROMPT_IDS),
            placeholder_token_id=PLACEHOLDER,
            config=self.config,
        )
        expected, expected_ids = build_encoder_window_items(self.processor, **kwargs)
        worker = copy.deepcopy(self.processor._processor)
        items, input_ids = build_encoder_window_items(
            self.processor, processor=worker, **kwargs
        )

        self.assertIsNot(worker.feature_extractor, self.extractor)
        self.assertTrue(torch.equal(input_ids, expected_ids))
        self.assertEqual(len(items), 3)
        for item, reference in zip(items, expected):
            self.assertTrue(torch.equal(item.feature, reference.feature))
            self.assertTrue(
                torch.equal(
                    item.feature_attention_mask, reference.feature_attention_mask
                )
            )
            self.assertEqual(item.offsets, reference.offsets)
            self.assertEqual(item.hash, reference.hash)

    def test_complete_windows_are_bitwise_stable_across_requests(self):
        audio = (
            np.random.default_rng(7)
            .normal(0, 0.1, size=3 * self.config.window_samples + 200)
            .astype(np.float32)
        )

        def build(samples):
            return build_encoder_window_items(
                self.processor,
                samples=samples,
                input_ids=torch.tensor(PROMPT_IDS),
                placeholder_token_id=PLACEHOLDER,
                config=self.config,
            )[0]

        two_windows = build(audio[: 2 * self.config.window_samples + 4000])
        three_windows = build(audio)
        solo = self.extractor(
            [audio[: self.config.window_samples]],
            sampling_rate=16000,
            return_attention_mask=True,
            return_tensors="pt",
            truncation=False,
            padding="longest",
        )["input_features"]

        self.assertEqual(
            [item.offsets for item in three_windows[:2]],
            [[(1, 104)], [(105, 208)]],
        )
        for index in range(2):
            self.assertEqual(two_windows[index].hash, three_windows[index].hash)
            self.assertTrue(
                torch.equal(two_windows[index].feature, three_windows[index].feature)
            )
        self.assertTrue(torch.equal(three_windows[0].feature, solo))
        # A rolling request that starts at window 1 with leading context
        # reproduces window 1 bit for bit.
        rolling = build_encoder_window_items(
            self.processor,
            samples=audio[self.config.window_samples - self.config.context_samples :],
            input_ids=torch.tensor(PROMPT_IDS),
            placeholder_token_id=PLACEHOLDER,
            config=self.config,
            leading_context_samples=self.config.context_samples,
        )[0]
        self.assertEqual(rolling[0].hash, three_windows[1].hash)
        self.assertTrue(torch.equal(rolling[0].feature, three_windows[1].feature))
        # The 200-sample remainder is shorter than n_fft and merges backwards.
        self.assertEqual(len(three_windows), 3)
        self.assertEqual(three_windows[2].offsets, [(209, 313)])


if __name__ == "__main__":
    unittest.main()
