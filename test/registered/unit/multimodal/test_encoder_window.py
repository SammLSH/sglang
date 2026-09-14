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
from sglang.srt.multimodal.encoder_window import encoder_window_kwargs
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
        return_value={"input_ids": torch.tensor([PROMPT_IDS])},
    )
    processor.tokenizer.convert_tokens_to_ids.return_value = PLACEHOLDER
    owner._processor = processor
    owner._tokenizer = owner._processor.tokenizer
    owner._tokenizer_auto_adds_specials = False
    owner.audio_config = {}
    owner.mm_feature_transport = "cpu"
    owner.ATTR_NAME_TO_MODALITY = {
        "input_features": Modality.AUDIO,
        "feature_attention_mask": Modality.AUDIO,
    }
    owner.FEATURE_NAMES = ["input_features"]
    owner.hf_config = SimpleNamespace(
        thinker_config=SimpleNamespace(
            audio_config=SimpleNamespace(n_window_infer=800, n_window=50)
        )
    )
    owner._finalize_mm_items = lambda items, *, images: items
    return owner


class TestAudioWindows(CustomTestCase):
    def setUp(self):
        self.owner = _owner()
        self.config = self.owner.encoder_window_config()
        self.audio = (
            np.random.default_rng(7)
            .normal(0, 0.1, 3 * self.config.window_samples + 4000)
            .astype(np.float32)
        )

    def test_windows_preserve_features_and_offsets(self):
        context = self.config.leading_context_samples
        base = BaseMultiModalProcessorOutput(
            input_text="audio",
            audios=[self.audio[self.config.window_samples - context :]],
        )
        items, ids, whole = self.owner.process_and_combine_mm_data(
            base,
            MultimodalSpecialTokens(audio_token_id=PLACEHOLDER),
            mm_processor_kwargs=encoder_window_kwargs(leading_context_samples=context),
        )
        self.assertEqual([item.feature.shape[-1] for item in items], [800, 825])
        self.assertEqual([item.offsets for item in items], [[(1, 104)], [(105, 212)]])
        self.assertEqual(ids.tolist(), [10] + [PLACEHOLDER] * 212 + [11])
        # Keep whole-request feature values; exclude extraction context and padding.
        leading_frames = context // self.owner._processor.feature_extractor.hop_length
        valid_frames = int(whole["feature_attention_mask"].sum())
        self.assertTrue(
            torch.equal(
                torch.cat([item.feature for item in items], dim=-1),
                whole["input_features"][..., leading_frames:valid_frames],
            )
        )

    def test_identity_includes_valid_frame_mask(self):
        feature = torch.zeros(1, 128, 2)
        source = MultimodalDataItem(modality=Modality.AUDIO, feature=feature)
        first, shorter = [
            self.owner.make_encoder_window_item(
                source, feature=feature, mask=torch.tensor([mask]), offsets=[(0, 0)]
            )
            for mask in ([1, 1], [1, 0])
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
        with patch.object(
            self.owner, "process_mm_data", wraps=self.owner.process_mm_data
        ) as process:
            items, ids, _ = self.owner.process_and_combine_mm_data(
                base, tokens, mm_processor_kwargs=kwargs
            )
            process.assert_called_once()
        self.assertEqual(len(items), 1)
        self.assertEqual(ids.tolist(), [10] + [PLACEHOLDER] * 108 + [11])
        self.owner.audio_config = {"truncation": True}
        with self.assertRaisesRegex(ValueError, "truncation=False"):
            self.owner.encoder_window_config()


class TestAudioBatching(CustomTestCase):
    def test_variable_widths_preserve_each_batch_row_in_item_order(self):
        items = []
        for frames, valid_frames in ((6, [6]), (10, [2, 9])):
            feature = torch.arange(len(valid_frames) * 4 * frames).reshape(
                -1, 4, frames
            )
            mask = torch.arange(frames)[None, :] < torch.tensor(valid_frames)[:, None]
            items.append(
                MultimodalDataItem(
                    modality=Modality.AUDIO,
                    feature=feature.float(),
                    model_specific_data={"feature_attention_mask": mask},
                )
            )
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
        self.assertTrue(torch.equal(features, expected.float()))


if __name__ == "__main__":
    unittest.main()
