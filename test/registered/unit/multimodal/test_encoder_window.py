"""Audio window cache identity, prompt offsets, and encoder batching."""

# ruff: noqa: E402 -- CPU kernel stubs must precede runtime imports.

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that pull in sgl_kernel

import numpy as np
import torch
from transformers import WhisperFeatureExtractor

from sglang.srt.configs.qwen3_asr import Qwen3ASRProcessor
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.qwen3_asr import Qwen3ASRForConditionalGeneration
from sglang.srt.multimodal.encoder_window import (
    build_encoder_window_items,
    encoder_window_kwargs,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.qwen3_asr import Qwen3ASRMultimodalProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

PLACEHOLDER = 99
PROMPT_IDS = [10, PLACEHOLDER, 11]


def _owner():
    owner = Qwen3ASRMultimodalProcessor.__new__(Qwen3ASRMultimodalProcessor)
    # Keep the real feature extraction and token-length mapping.
    processor = Qwen3ASRProcessor.__new__(Qwen3ASRProcessor)
    processor.feature_extractor = WhisperFeatureExtractor(
        feature_size=128, sampling_rate=16000, hop_length=160, n_fft=400
    )
    processor.tokenizer = Mock(
        bos_token=None,
        return_value=SimpleNamespace(input_ids=torch.tensor([PROMPT_IDS])),
    )
    owner._processor = processor
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
    return build_encoder_window_items(
        owner,
        samples=samples,
        input_ids=torch.tensor(PROMPT_IDS),
        placeholder_token_id=PLACEHOLDER,
        config=owner.encoder_window_config(),
        leading_context_samples=leading_context_samples,
    )


class TestAudioWindows(CustomTestCase):
    def setUp(self):
        self.owner = _owner()
        self.config = self.owner.encoder_window_config()
        self.audio = (
            np.random.default_rng(7)
            .normal(0, 0.1, 2 * self.config.window_samples + 4000)
            .astype(np.float32)
        )

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
        with patch.object(WhisperFeatureExtractor, "__call__", side_effect=outputs):
            samples = np.zeros(320, dtype=np.float32)
            prepared = self.owner.extract_encoder_window_features([samples, samples])
            first, shorter = [
                self.owner.make_encoder_window_item(feature, mask, [(0, count - 1)])
                for feature, mask, count in zip(
                    prepared.features, prepared.masks, prepared.token_counts
                )
            ]
        self.assertNotEqual(first.hash, shorter.hash)

    def test_opt_in_uses_server_config(self):
        base = BaseMultiModalProcessorOutput(
            input_text="audio",
            audios=[self.audio[: self.config.window_samples + 4000]],
        )
        tokens = MultimodalSpecialTokens(audio_token_id=PLACEHOLDER)
        with patch.object(
            BaseMultimodalProcessor, "process_and_combine_mm_data", return_value="base"
        ) as fallback:
            self.assertEqual(
                self.owner.process_and_combine_mm_data(base, tokens), "base"
            )
            fallback.assert_called_once()

        kwargs = encoder_window_kwargs()
        kwargs["encoder_window"]["window_samples"] = 1
        items, ids, _ = self.owner.process_and_combine_mm_data(
            base, tokens, mm_processor_kwargs=kwargs
        )
        self.assertEqual(len(items), 2)
        self.assertEqual(ids.tolist(), [10] + [PLACEHOLDER] * 108 + [11])


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
    def test_variable_widths_preserve_each_batch_row_in_item_order(self):
        items = [_audio_item(6, valid_frames=[6]), _audio_item(10, valid_frames=[2, 9])]
        expected = torch.cat(
            [
                items[0].feature[0],
                items[1].feature[0, :, :2],
                items[1].feature[1, :, :9],
            ],
            dim=1,
        )

        tower = Mock(dtype=torch.float32)
        tower.parameters.return_value = iter([torch.zeros(1)])
        Qwen3ASRForConditionalGeneration.get_audio_feature(
            SimpleNamespace(audio_tower=tower), items
        )
        features = tower.call_args.args[0]
        lengths = tower.call_args.kwargs["feature_lens"]
        self.assertEqual(lengths.tolist(), [6, 2, 9])
        self.assertEqual(lengths.dtype, torch.long)
        self.assertEqual(features.dtype, torch.float32)
        self.assertTrue(torch.equal(features, expected.float()))


if __name__ == "__main__":
    unittest.main()
