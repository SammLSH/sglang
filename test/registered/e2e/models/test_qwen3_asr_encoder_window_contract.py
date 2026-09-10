"""Check Qwen3-ASR encoder equivalence required for window reuse.

Compare whole-clip, per-window and batched encoding on the same real audio.
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
    build_audio_window_items,
    resolve_audio_window_geometry,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.utils import load_audio
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")

MODEL = "Qwen/Qwen3-ASR-0.6B"
SAMPLE_RATE = 16000
# One original ~15 s recording: a complete 8 s window plus a tail.
AUDIO_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
PLACEHOLDER = 99


class TestQwen3ASREncoderWindowContract(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
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
        from sglang.srt.multimodal.processors.qwen3_asr import (
            Qwen3ASRMultimodalProcessor,
        )

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

        cls.capability = Qwen3ASRMultimodalProcessor.__new__(
            Qwen3ASRMultimodalProcessor
        )
        cls.capability._processor = Qwen3ASRProcessor.from_pretrained(cls.snapshot)
        cls.capability.audio_config = {}
        cls.capability.hf_config = config
        cls.window = resolve_audio_window_geometry(cls.capability)
        response = requests.get(AUDIO_URL, timeout=120)
        response.raise_for_status()
        cls.audio = load_audio(response.content, sr=SAMPLE_RATE, mono=True).astype(
            np.float32
        )

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

    def _items(self, samples):
        items, _ = build_audio_window_items(
            self.capability,
            samples=samples,
            input_ids=torch.tensor([10, PLACEHOLDER, 11]),
            placeholder_token_id=PLACEHOLDER,
            geometry=self.window,
        )
        return items

    def test_windows_reproduce_the_whole_clip_encoding(self):
        items = self._items(self.audio)
        self.assertEqual(len(items), 2)
        features = self.capability.extract_audio_window_features([self.audio])
        whole_item = MultimodalDataItem(
            modality=Modality.AUDIO,
            feature=features.features[0],
            model_specific_data={"feature_attention_mask": features.masks[0]},
        )
        whole = self._embed([whole_item])
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
        self.assertGreaterEqual(float((cosine >= 0.99).float().mean()), 0.95)

        # Variable-width items share one encoder call and match the per-item
        # results up to bf16 noise.
        batched = self._embed(items)
        self.assertLess(float((batched - per_window).abs().max()), 5e-2)

    def test_short_tail_encoding_does_not_depend_on_other_cache_misses(self):
        # A non-block-aligned tail must encode identically whether the preceding
        # complete window is also a cache miss or is already cached.
        samples = self.audio[: self.window.window_samples + 37 * self.window.hop_length]
        items = self._items(samples)
        cold = self._embed(items)[self.window.window_tokens :]
        hot = self._embed([items[-1]])
        self.assertEqual(cold.shape, hot.shape)
        # Check both scale and direction, allowing bf16 batching differences.
        relative_error = (cold - hot).norm() / cold.norm()
        cosine = torch.nn.functional.cosine_similarity(cold, hot, dim=-1)
        self.assertLess(float(relative_error), 0.05)
        self.assertGreater(float(cosine.min()), 0.999)


if __name__ == "__main__":
    unittest.main()
