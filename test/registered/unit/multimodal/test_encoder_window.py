"""CPU contracts for audio windows, batching, cache identity, and worker isolation."""

# ruff: noqa: E402 -- CPU kernel stubs must precede runtime imports.

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that pull in sgl_kernel

import numpy as np
import torch
import uvloop
from transformers import WhisperFeatureExtractor

from sglang.srt.configs.qwen3_asr import Qwen3ASRProcessor
from sglang.srt.configs.qwen3_omni import Qwen3OmniMoeAudioEncoderConfig
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.qwen3_asr import Qwen3ASRForConditionalGeneration
from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeAudioEncoder
from sglang.srt.multimodal.encoder_window import (
    build_audio_window_items,
    build_audio_window_processor_kwargs,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
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


class _Tokenizer:
    bos_token = None

    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append(text)
        return SimpleNamespace(input_ids=torch.tensor([PROMPT_IDS]))


class _Extractor(WhisperFeatureExtractor):
    def __init__(self):
        super().__init__(
            feature_size=128, sampling_rate=16000, hop_length=160, n_fft=400
        )
        self.calls = []

    def __call__(self, windows, **kwargs):
        self.calls.append([len(window) for window in windows])
        return super().__call__(windows, **kwargs)


class _Processor(Qwen3ASRProcessor):
    """Use the real frontend and length mapping without loading a tokenizer."""

    def __init__(self):
        self.feature_extractor = _Extractor()
        self.tokenizer = _Tokenizer()
        self.length_calls = []

    def _get_feat_extract_output_lengths(self, lengths):
        self.length_calls.append(list(lengths))
        return super()._get_feat_extract_output_lengths(lengths)


def _owner():
    owner = Qwen3ASRMultimodalProcessor.__new__(Qwen3ASRMultimodalProcessor)
    owner._processor = _Processor()
    owner._tokenizer = owner._processor.tokenizer
    owner._tokenizer_auto_adds_specials = False
    owner.audio_config = {}
    owner.hf_config = SimpleNamespace(
        thinker_config=SimpleNamespace(
            audio_config=SimpleNamespace(n_window_infer=800, n_window=50)
        )
    )
    owner._finalize_mm_items = lambda items, *, images: items
    return owner


def _items(owner, samples, *, leading_context_samples=0):
    return build_audio_window_items(
        owner,
        samples=samples,
        input_ids=torch.tensor(PROMPT_IDS),
        placeholder_token_id=PLACEHOLDER,
        config=owner.audio_window_config(),
        leading_context_samples=leading_context_samples,
    )


class TestAudioWindows(CustomTestCase):
    def setUp(self):
        self.owner = _owner()
        self.config = self.owner.audio_window_config()
        self.audio = (
            np.random.default_rng(7)
            .normal(0, 0.1, 3 * self.config.window_samples + 4000)
            .astype(np.float32)
        )

    def test_config_checks_standalone_and_leading_context_padding(self):
        owner = _owner()
        owner.audio_config = {"pad_to_multiple_of": 512}
        # Standalone extraction passes; only the leading-context probe catches
        # padding that changes a rolling window's width.
        standalone = owner.prepare_audio_window(
            np.zeros(self.config.window_samples, dtype=np.float32)
        )
        self.assertEqual(standalone.item.feature.shape[-1], 800)
        with self.assertRaisesRegex(ValueError, "unpadded frames"):
            owner.audio_window_config()

    def test_shared_encoder_default_keeps_short_input_padding(self):
        config = Qwen3OmniMoeAudioEncoderConfig(
            num_mel_bins=8,
            encoder_layers=0,
            d_model=16,
            downsample_hidden_size=2,
            output_dim=16,
        )
        encoder = Qwen3OmniMoeAudioEncoder(config)
        widths = []
        handle = encoder.conv2d1.register_forward_pre_hook(
            lambda _module, args: widths.append(args[0].shape[-1])
        )
        try:
            with torch.no_grad():
                output = encoder(torch.zeros(8, 25), feature_lens=torch.tensor([25]))
        finally:
            handle.remove()
        self.assertEqual(widths, [25])
        self.assertEqual(output.last_hidden_state.shape, (4, 16))

    def test_windows_preserve_tails_offsets_and_cache_identity(self):
        window = self.config.window_samples
        items, ids = _items(self.owner, self.audio[: 2 * window + 4000])
        self.assertEqual(
            [item.offsets for item in items], [[(1, 104)], [(105, 208)], [(209, 212)]]
        )
        self.assertEqual(ids.tolist(), [10] + [PLACEHOLDER] * 212 + [11])
        self.assertEqual(
            [
                (item.feature.shape[-1], int(item.feature_attention_mask.sum()))
                for item in items
            ],
            [(800, 800), (800, 800), (25, 25)],
        )

        exact, _ = _items(self.owner, self.audio[: 2 * window])
        self.assertEqual(len(exact), 2)
        tiny_tail, _ = _items(self.owner, self.audio[: 2 * window + 200])
        self.assertEqual(len(tiny_tail), 2)
        self.assertNotEqual(tiny_tail[1].hash, exact[1].hash)
        self.assertGreater(tiny_tail[1].feature.shape[-1], 800)

        context = self.config.leading_context_samples
        self.assertEqual(context, 320)
        rolling, _ = _items(
            self.owner,
            self.audio[window - context : 2 * window + 4000],
            leading_context_samples=context,
        )
        for first, second in (
            (items[0], exact[0]),
            (items[1], exact[1]),
            (rolling[0], items[1]),
        ):
            self.assertEqual(first.hash, second.hash)
            self.assertTrue(torch.equal(first.feature, second.feature))
        self.assertNotEqual(rolling[0].offsets, items[1].offsets)

    def test_identity_includes_valid_frame_mask(self):
        feature = torch.zeros(1, 128, 2)
        outputs = [
            {"input_features": feature, "attention_mask": torch.tensor([mask])}
            for mask in ([1, 1], [1, 0])
        ]
        with patch.object(_Extractor, "__call__", side_effect=outputs):
            samples = np.zeros(320, dtype=np.float32)
            first = self.owner.prepare_audio_window(samples).item
            shorter = self.owner.prepare_audio_window(samples).item
        self.assertNotEqual(first.hash, shorter.hash)

    def test_opt_in_uses_server_config_and_a_cold_worker_clone(self):
        owner = _owner()
        source = owner._processor
        base = BaseMultiModalProcessorOutput(
            input_text="audio",
            audios=[self.audio[: self.config.window_samples + 4000]],
        )
        tokens = MultimodalSpecialTokens(audio_token_id=PLACEHOLDER)
        with patch.object(
            BaseMultimodalProcessor, "process_and_combine_mm_data", return_value="base"
        ) as fallback:
            self.assertEqual(owner.process_and_combine_mm_data(base, tokens), "base")
            fallback.assert_called_once()
        kwargs = build_audio_window_processor_kwargs()
        kwargs["encoder_window"]["window_samples"] = 1
        executor = MultimodalProcessorExecutor(lambda: source, max_workers=1)
        owner.mm_processor_executor = executor
        loop = uvloop.new_event_loop()
        try:
            items, ids, _ = loop.run_until_complete(
                asyncio.wait_for(
                    owner.process_and_combine_mm_data_async(
                        base, tokens, mm_processor_kwargs=kwargs
                    ),
                    timeout=10,
                )
            )
        finally:
            executor.shutdown()
            loop.close()
        self.assertEqual(len(items), 2)
        self.assertEqual(ids.tolist(), [10] + [PLACEHOLDER] * 108 + [11])
        self.assertEqual(
            (
                source.feature_extractor.calls,
                source.tokenizer.calls,
                source.length_calls,
            ),
            ([], [], []),
        )
        self.assertEqual(owner.audio_window_config(), self.config)


def _audio_item(frames, valid_frames=None):
    batch = len(valid_frames) if valid_frames is not None else 1
    feature = torch.arange(batch * 4 * frames, dtype=torch.float64).reshape(
        batch, 4, frames
    )
    data = {}
    if valid_frames is not None:
        data["feature_attention_mask"] = (
            torch.arange(frames)[None, :] < torch.tensor(valid_frames)[:, None]
        )
    return MultimodalDataItem(
        modality=Modality.AUDIO, feature=feature, model_specific_data=data
    )


class TestAudioBatching(CustomTestCase):
    def _assert_encoding(self, items, expected, expected_lengths):
        tower = Mock(dtype=torch.float32)
        tower.parameters.return_value = iter([torch.zeros(1)])
        Qwen3ASRForConditionalGeneration.get_audio_feature(
            SimpleNamespace(audio_tower=tower), items
        )
        features = tower.call_args.args[0]
        lengths = tower.call_args.kwargs["feature_lens"]
        self.assertEqual(lengths.tolist(), expected_lengths)
        self.assertEqual(lengths.dtype, torch.long)
        self.assertEqual(features.dtype, torch.float32)
        self.assertTrue(torch.equal(features, expected.float()))

    def test_mixed_masks_preserve_each_batch_row_in_item_order(self):
        items = [_audio_item(6), _audio_item(10, valid_frames=[2, 9])]
        expected = torch.cat(
            [
                items[0].feature[0],
                items[1].feature[0, :, :2],
                items[1].feature[1, :, :9],
            ],
            dim=1,
        )
        self._assert_encoding(items, expected, [6, 2, 9])

    def test_unmasked_variable_width_items_keep_original_lengths(self):
        items = [_audio_item(4), _audio_item(7)]
        expected = torch.cat([items[0].feature[0], items[1].feature[0]], dim=1)
        self._assert_encoding(items, expected, [4, 7])


if __name__ == "__main__":
    unittest.main()
