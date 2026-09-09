"""Unit tests for concat_padded_audio_features and its Qwen3-ASR consumer."""

import unittest

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that pull in sgl_kernel

import torch
from torch import nn

from sglang.srt.managers.mm_utils import concat_padded_audio_features
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.qwen3_asr import Qwen3ASRForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_N_MELS = 4


def _audio_item(frames: int, *, valid_frames=None, with_mask=True, seed=0):
    feature = torch.randn(
        1, _N_MELS, frames, generator=torch.Generator().manual_seed(seed)
    )
    model_specific_data = {}
    if with_mask:
        mask = torch.zeros(1, frames, dtype=torch.int32)
        mask[:, : (frames if valid_frames is None else valid_frames)] = 1
        model_specific_data["feature_attention_mask"] = mask
    return MultimodalDataItem(
        modality=Modality.AUDIO,
        feature=feature,
        model_specific_data=model_specific_data,
    )


class _RecordingAudioTower(nn.Module):
    """Stands in for the audio encoder and records what it was called with."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.dtype = torch.float32
        self.calls = []

    def forward(self, input_features, feature_lens=None):
        self.calls.append((input_features, feature_lens))
        hidden = torch.zeros(int(feature_lens.sum()), 2)
        return type("Output", (), {"last_hidden_state": hidden})()


class TestConcatPaddedAudioFeatures(CustomTestCase):
    def test_equal_width_matches_torch_cat_bitwise(self):
        items = [_audio_item(10, seed=1), _audio_item(10, seed=2)]

        features, mask = concat_padded_audio_features(items)

        self.assertTrue(torch.equal(features, torch.cat([i.feature for i in items])))
        self.assertTrue(
            torch.equal(
                mask,
                torch.cat(
                    [i.model_specific_data["feature_attention_mask"] for i in items]
                ),
            )
        )

    def test_variable_width_pads_features_and_masks_with_zeros(self):
        items = [
            _audio_item(6, seed=1),
            _audio_item(10, seed=2),
            _audio_item(3, seed=3),
        ]

        features, mask = concat_padded_audio_features(items)

        self.assertEqual(tuple(features.shape), (3, _N_MELS, 10))
        self.assertEqual(tuple(mask.shape), (3, 10))
        self.assertEqual(mask.sum(dim=-1).tolist(), [6, 10, 3])
        for row, item in enumerate(items):
            width = item.feature.shape[-1]
            self.assertTrue(torch.equal(features[row, :, :width], item.feature[0]))
            self.assertFalse(features[row, :, width:].any())
            self.assertFalse(mask[row, width:].any())

        # A mask shorter than its feature width is carried through unchanged.
        _, mask = concat_padded_audio_features(
            [_audio_item(8, valid_frames=5, seed=1), _audio_item(12, seed=2)]
        )
        self.assertEqual(mask.sum(dim=-1).tolist(), [5, 12])

    def test_missing_masks_are_none_or_synthesized(self):
        items = [_audio_item(4, with_mask=False), _audio_item(7, with_mask=False)]
        features, mask = concat_padded_audio_features(items)
        self.assertIsNone(mask)
        self.assertEqual(tuple(features.shape), (2, _N_MELS, 7))

        # An unmasked item next to a masked one gets an all-ones mask.
        items = [_audio_item(4, with_mask=False), _audio_item(7, valid_frames=6)]
        _, mask = concat_padded_audio_features(items)
        self.assertEqual(mask[0].tolist(), [1, 1, 1, 1, 0, 0, 0])
        self.assertEqual(mask[1].tolist(), [1, 1, 1, 1, 1, 1, 0])

    def test_rejects_empty_and_non_3d_features(self):
        with self.assertRaises(ValueError):
            concat_padded_audio_features([])
        bad = MultimodalDataItem(
            modality=Modality.AUDIO, feature=torch.zeros(_N_MELS, 5)
        )
        with self.assertRaisesRegex(ValueError, "batch, n_mels, frames"):
            concat_padded_audio_features([bad])


class TestQwen3ASRVariableWidthBatch(CustomTestCase):
    def _model(self):
        model = Qwen3ASRForConditionalGeneration.__new__(
            Qwen3ASRForConditionalGeneration
        )
        nn.Module.__init__(model)
        model.audio_tower = _RecordingAudioTower()
        return model

    def test_get_audio_feature_batches_different_durations(self):
        model = self._model()
        items = [_audio_item(6, seed=1), _audio_item(10, valid_frames=9, seed=2)]

        hidden = model.get_audio_feature(items)

        ((input_features, feature_lens),) = model.audio_tower.calls
        # The encoder receives only valid frames, laid out as (n_mels, frames).
        self.assertEqual(feature_lens.tolist(), [6, 9])
        self.assertEqual(tuple(input_features.shape), (_N_MELS, 15))
        self.assertTrue(torch.equal(input_features[:, :6], items[0].feature[0]))
        self.assertTrue(torch.equal(input_features[:, 6:], items[1].feature[0, :, :9]))
        self.assertEqual(tuple(hidden.shape), (15, 2))


if __name__ == "__main__":
    unittest.main()
