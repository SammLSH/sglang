"""Qwen3-ASR model compatible with HuggingFace weights"""

import logging
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.configs.qwen3_asr import Qwen3ASRConfig
from sglang.srt.configs.qwen3_omni import Qwen3OmniMoeAudioEncoderConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeAudioEncoder
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class Qwen3ASRForConditionalGeneration(nn.Module):
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3ASRConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        thinker_config = config.thinker_config

        if getattr(thinker_config, "audio_config", None) is None:
            thinker_config.audio_config = Qwen3OmniMoeAudioEncoderConfig()

        self.audio_tower = Qwen3OmniMoeAudioEncoder(
            thinker_config.audio_config,
            pad_to_full_conv_block=2 * thinker_config.audio_config.n_window == 100,
        )
        self.language_model = Qwen3ForCausalLM(
            thinker_config.text_config,
            quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        if not items:
            raise ValueError("audio encoding requires at least one item")
        device = next(self.audio_tower.parameters()).device
        masks = [getattr(item, "feature_attention_mask", None) for item in items]
        mask_device = next((mask.device for mask in masks if mask is not None), None)
        features, lengths, valid_frames = [], [], []
        for item, mask in zip(items, masks):
            feature = item.feature
            if not isinstance(feature, torch.Tensor) or feature.ndim != 3:
                raise ValueError(
                    "audio features must be (batch, n_mels, frames) tensors"
                )
            batch, _, frames = feature.shape
            # Each row is a view: concatenate different durations directly
            # without padding or copying to flatten a multi-row item.
            features.extend(feature.unbind(0))
            if mask_device is None:
                lengths.extend([frames] * batch)
            else:
                if mask is None:
                    mask = torch.ones(
                        (batch, frames), dtype=torch.bool, device=mask_device
                    )
                lengths.append(mask.sum(dim=1, dtype=torch.long))
                valid_frames.append(mask.flatten())

        input_features = torch.cat(features, dim=1).to(
            device=device, dtype=self.audio_tower.dtype
        )
        if mask_device is None:
            audio_feature_lengths = torch.tensor(
                lengths, dtype=torch.long, device=device
            )
        else:
            audio_feature_lengths = torch.cat(lengths).to(device)
            # Select once after concatenation, avoiding per-item GPU boolean
            # indexing and keeping (n_mels, total_valid_frames) for the encoder.
            valid = torch.cat(valid_frames).to(device=device, dtype=torch.bool)
            input_features = input_features[:, valid]
        return self.audio_tower(
            input_features, feature_lens=audio_feature_lengths
        ).last_hidden_state

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.AUDIO: self.get_audio_feature,
            },
            positions=positions,
        )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        llm_stacked_params = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        # Audio tower has separate q/k/v in checkpoint → stack into qkv_proj
        audio_stacked_params = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            if (
                getattr(
                    self.config.thinker_config.text_config, "tie_word_embeddings", False
                )
                and "lm_head.weight" in name
            ):
                continue

            if "talker" in name or "code2wav" in name:
                continue

            if name.startswith("thinker.audio_tower."):
                name = name.replace("thinker.audio_tower.", "audio_tower.", 1)
            elif name.startswith("thinker.lm_head."):
                name = name.replace("thinker.lm_head.", "language_model.lm_head.", 1)
            elif name.startswith("thinker.model."):
                name = name.replace("thinker.model.", "language_model.model.", 1)

            is_audio = "audio_tower" in name

            # Audio tower: remap out_proj → proj for VisionAttention
            if is_audio and "out_proj" in name:
                name = name.replace("out_proj", "proj")

            stacked_params = audio_stacked_params if is_audio else llm_stacked_params

            for param_name, weight_name, shard_id in stacked_params:
                if weight_name not in name:
                    continue
                name_tmp = name.replace(weight_name, param_name)
                if name_tmp.endswith(".bias") and name_tmp not in params_dict:
                    continue
                if name_tmp not in params_dict:
                    continue
                param = params_dict[name_tmp]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = Qwen3ASRForConditionalGeneration
