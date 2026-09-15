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
from sglang.srt.configs.qwen3_omni import Qwen3OmniMoeAudioEncoderConfig
from sglang.srt.managers.mm_utils import concat_padded_audio_features
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.qwen3_asr import Qwen3ASRForConditionalGeneration
from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeAudioEncoder
from sglang.srt.multimodal.encoder_window import build_encoder_window_items
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


def make_processor() -> Qwen3ASRMultimodalProcessor:
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
    def setUp(self) -> None:
        self.owner = make_processor()
        self.config = self.owner.encoder_window_config()
        self.audio = (
            np.random.default_rng(7)
            .normal(0, 0.1, 3 * self.config.window_samples + 4000)
            .astype(np.float32)
        )

    def test_windows_preserve_features_and_offsets(self) -> None:
        context = self.config.leading_context_samples
        base = BaseMultiModalProcessorOutput(
            input_text="audio",
            audios=[self.audio[self.config.window_samples - context :]],
        )
        items, ids, whole = self.owner.process_and_combine_mm_data(
            base,
            MultimodalSpecialTokens(audio_token_id=PLACEHOLDER),
            mm_processor_kwargs={
                "encoder_window": {"leading_context_samples": context}
            },
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

    def test_cache_miss_subsets_preserve_encoder_convolution_outputs(self) -> None:
        # Run real convolution, padding, positions and projections on CPU.
        # Attention layers and checkpoint weights require the GPU smoke test.
        config = Qwen3OmniMoeAudioEncoderConfig(
            num_mel_bins=128,
            encoder_layers=0,
            d_model=16,
            downsample_hidden_size=4,
            output_dim=16,
            n_window=50,
            n_window_infer=800,
        )
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(7)
            encoder = Qwen3OmniMoeAudioEncoder(config).eval()
            for parameter in encoder.parameters():
                parameter.uniform_(-0.2, 0.2)

            def _encode(items: list[MultimodalDataItem]) -> torch.Tensor:
                return Qwen3ASRForConditionalGeneration.get_audio_feature(
                    SimpleNamespace(audio_tower=encoder), items
                )

            for tail_frames in (1, 25, 99, 100, 101, 799):
                frames = 1600 + tail_frames
                token_count = self.owner.encoder_window_output_lengths([frames])[0]
                source = MultimodalDataItem(
                    modality=Modality.AUDIO,
                    feature=torch.randn(1, 128, frames),
                    model_specific_data={
                        "feature_attention_mask": torch.ones(
                            1, frames, dtype=torch.long
                        )
                    },
                )
                items, _ = build_encoder_window_items(
                    self.owner,
                    item=source,
                    input_ids=torch.tensor([10] + [PLACEHOLDER] * token_count + [11]),
                    placeholder_token_id=PLACEHOLDER,
                    config=self.config,
                )
                whole = _encode([source])
                expected = [
                    whole[start - 1 : end]
                    for item in items
                    for start, end in item.offsets
                ]
                # Last-only, interleaved and mixed-order misses must retain semantics.
                for misses in (
                    (0,),
                    (-1,),
                    tuple(range(len(items))),
                    tuple(range(0, len(items), 2)),
                    (-1, 0),
                ):
                    with self.subTest(tail_frames=tail_frames, misses=misses):
                        torch.testing.assert_close(
                            _encode([items[index] for index in misses]),
                            torch.cat([expected[index] for index in misses]),
                            rtol=1e-5,
                            atol=1e-7,
                        )
                if tail_frames == 25:
                    unmerged = self.owner.make_encoder_window_item(
                        source,
                        feature=source.feature[..., -tail_frames:],
                        mask=torch.ones(1, tail_frames, dtype=torch.long),
                        offsets=[],
                    )
                    tail = _encode([unmerged])
                    self.assertGreater(
                        (whole[-len(tail) :] - tail).abs().max().item(), 1e-5
                    )
                else:
                    continue

    def test_identity_includes_valid_frame_mask(self) -> None:
        feature = torch.zeros(1, 128, 2)
        source = MultimodalDataItem(modality=Modality.AUDIO, feature=feature)
        first, shorter = [
            self.owner.make_encoder_window_item(
                source, feature=feature, mask=torch.tensor([mask]), offsets=[(0, 0)]
            )
            for mask in ([1, 1], [1, 0])
        ]
        self.assertNotEqual(first.hash, shorter.hash)

    def test_opt_in_uses_server_config(self) -> None:
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

        kwargs = {"encoder_window": {"window_samples": 1}}
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
    def test_variable_widths_preserve_each_batch_row_in_item_order(self) -> None:
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

        # Single and equal-width batches use the original tensors or direct cat.
        feature, mask = concat_padded_audio_features([items[0]])
        self.assertIs(feature, items[0].feature)
        self.assertIs(mask, items[0].model_specific_data["feature_attention_mask"])
        feature, mask = concat_padded_audio_features([items[0], items[0]])
        self.assertTrue(torch.equal(feature, items[0].feature.repeat(2, 1, 1)))
        self.assertEqual(mask.sum(dim=1).tolist(), [6, 6])


if __name__ == "__main__":
    unittest.main()
