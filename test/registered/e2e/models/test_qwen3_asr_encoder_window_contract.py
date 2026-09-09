"""GPU contract behind realtime encoder windowing for Qwen3-ASR.

Encoder windowing reuses a complete window's encoder output across requests,
which is only sound if that output does not depend on the audio around the
window. This test builds the sglang model in-process and compares
``get_audio_feature`` on a whole clip against the concatenation of the
per-window items the encoder_window builder produces for the same clip.

Reuse the template for another model by pointing ``MODEL``, ``PROCESSOR``
and the capability at it; the thresholds describe what Qwen3-ASR-0.6B
measures with leading-context extraction (see the builder), where the
residual differences come from the extractor's per-input log-mel floor.
"""

import os
import socket
import unittest

import numpy as np
import requests
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig

import sglang.srt.configs  # noqa: F401  registers the Qwen3-ASR config classes
from sglang.srt.configs.qwen3_asr import Qwen3ASRProcessor
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.multimodal.encoder_window import (
    EncoderWindowMixin,
    EncoderWindowSpec,
    build_encoder_window_items,
    resolve_encoder_window_config,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.utils import load_audio
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")

MODEL = "Qwen/Qwen3-ASR-0.6B"
SAMPLE_RATE = 16000
# Two public clips, ~25 s together: three complete 8 s windows plus a tail.
AUDIO_URLS = [
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
    "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/audios/Trump_WEF_2018_10s.mp3",
]
PLACEHOLDER = 99


class _Capability(EncoderWindowMixin):
    """The processor-side capability, without a full multimodal processor."""

    def __init__(self, processor, audio_config):
        self._processor = processor
        self.audio_config = {}
        self._audio_config = audio_config

    def encoder_window_spec(self):
        return EncoderWindowSpec(
            window_frames=int(self._audio_config.n_window_infer),
            alignment_frames=2 * int(self._audio_config.n_window),
        )


def _download_samples(url):
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return load_audio(response.content, sr=SAMPLE_RATE, mono=True).astype(np.float32)


class TestQwen3ASREncoderWindowContract(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
        cls.snapshot = snapshot_download(
            MODEL, allow_patterns=["*.json", "*.safetensors", "*.txt"]
        )
        cls.parallel_override = get_parallel().override(
            tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0
        )
        cls.parallel_override.__enter__()
        cls.context_override = get_context().override_server_args(
            model_path=cls.snapshot
        )
        cls.context_override.__enter__()

        from sglang.srt.models.qwen3_asr import Qwen3ASRForConditionalGeneration

        config = AutoConfig.from_pretrained(cls.snapshot)
        default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("cuda"):
                cls.model = Qwen3ASRForConditionalGeneration(config)
        finally:
            torch.set_default_dtype(default_dtype)
        state = load_file(os.path.join(cls.snapshot, "model.safetensors"))
        cls.model.load_weights(iter(state.items()))
        cls.model.eval()

        cls.processor = Qwen3ASRProcessor.from_pretrained(cls.snapshot)
        cls.capability = _Capability(cls.processor, config.thinker_config.audio_config)
        cls.window = resolve_encoder_window_config(cls.capability)
        cls.audio = np.concatenate([_download_samples(url) for url in AUDIO_URLS])[
            : 3 * cls.window.window_samples + SAMPLE_RATE
        ]

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "model"):
            del cls.model
        for override in ("context_override", "parallel_override"):
            if hasattr(cls, override):
                getattr(cls, override).__exit__(None, None, None)
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()

    def _embed(self, items):
        with torch.no_grad():
            out = self.model.get_audio_feature(items).float()
        return out.reshape(-1, out.shape[-1])

    def _whole_clip_item(self, samples):
        features = self.capability.extract_encoder_window_features([samples])
        return MultimodalDataItem(
            modality=Modality.AUDIO,
            feature=features.features[0],
            model_specific_data={"feature_attention_mask": features.masks[0]},
        )

    def _window_items(self, samples, leading_context_samples=0):
        items, _ = build_encoder_window_items(
            self.capability,
            samples=samples,
            input_ids=torch.tensor([10, PLACEHOLDER, 11]),
            placeholder_token_id=PLACEHOLDER,
            config=self.window,
            leading_context_samples=leading_context_samples,
        )
        return items

    def test_geometry(self):
        self.assertEqual(self.window.window_seconds, 8.0)
        self.assertEqual(self.window.window_tokens, 104)
        self.assertEqual(self.window.context_samples, 320)

    def test_windows_reproduce_the_whole_clip_encoding(self):
        items = self._window_items(self.audio)
        self.assertEqual(len(items), 4)
        self.assertEqual(
            [item.offsets[0][1] - item.offsets[0][0] + 1 for item in items][:3],
            [104] * 3,
        )

        whole = self._embed([self._whole_clip_item(self.audio)])
        per_window = torch.cat([self._embed([item]) for item in items])
        self.assertEqual(tuple(whole.shape), tuple(per_window.shape))
        cosine = torch.nn.functional.cosine_similarity(whole, per_window, dim=-1).cpu()

        tokens = self.window.window_tokens
        for index in range(len(items)):
            segment = cosine[index * tokens : (index + 1) * tokens]
            with self.subTest(window=index):
                self.assertGreaterEqual(float(segment.mean()), 0.98)
                # The first token of a window is the one the extractor's
                # boundary can still touch; leading context keeps it close.
                self.assertGreaterEqual(float(segment[0]), 0.95)
        self.assertGreaterEqual(float(cosine.median()), 0.999)
        self.assertGreaterEqual(float((cosine >= 0.99).float().mean()), 0.95)
        # The extractor normalizes log-mel features by each input's own peak,
        # so a handful of near-silent tokens can land far from the whole-clip
        # encoding; they must stay rare.
        self.assertLessEqual(float((cosine < 0.9).float().mean()), 0.02)

        # Variable-width items share one encoder call through the padded
        # concat and match the per-item results up to bf16 noise.
        batched = self._embed(items)
        self.assertLess(float((batched - per_window).abs().max()), 5e-2)

    def test_rolling_request_reuses_window_identity(self):
        items = self._window_items(self.audio)
        start = self.window.window_samples - self.window.context_samples
        rolling = self._window_items(
            self.audio[start:], leading_context_samples=self.window.context_samples
        )
        self.assertEqual(rolling[0].hash, items[1].hash)
        self.assertEqual(rolling[1].hash, items[2].hash)
        self.assertTrue(torch.equal(rolling[0].feature, items[1].feature))


if __name__ == "__main__":
    unittest.main()
