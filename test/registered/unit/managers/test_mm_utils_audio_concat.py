"""Audio batching preserves valid frame order and every item's original length."""

# ruff: noqa: E402 -- CPU kernel stubs must precede runtime imports.

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that pull in sgl_kernel

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.qwen3_asr import Qwen3ASRForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _item(frames, valid_frames=None):
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
        tower.assert_called_once()
        features = tower.call_args.args[0]
        lengths = tower.call_args.kwargs["feature_lens"]
        self.assertEqual(lengths.tolist(), expected_lengths)
        self.assertEqual(lengths.dtype, torch.long)
        self.assertEqual(features.dtype, torch.float32)
        self.assertTrue(torch.equal(features, expected.float()))

    def test_mixed_masks_preserve_each_batch_row_in_item_order(self):
        items = [_item(6), _item(10, valid_frames=[2, 9])]
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
        items = [_item(4), _item(7)]
        expected = torch.cat([items[0].feature[0], items[1].feature[0]], dim=1)
        self._assert_encoding(items, expected, [4, 7])


if __name__ == "__main__":
    unittest.main()
