from __future__ import annotations

from typing import Optional, Sequence

import msgspec
import numpy as np
import torch
from pydantic import JsonValue
from transformers import ProcessorMixin

from sglang.srt.managers.mm_utils import hash_feature
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalDataValue,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)


class EncoderWindowSpec(msgspec.Struct, frozen=True):
    """Describe feature sizes the model can encode and cache as separate windows."""

    window_frames: int
    alignment_frames: int
    feature_time_dim: int = -1
    mask_key: str = "feature_attention_mask"
    # Keep short tails with a complete window when encoding alone changes padding.
    merge_tail_below_frames: int = 0

    def __post_init__(self) -> None:
        if self.window_frames <= 0 or self.alignment_frames <= 0:
            raise ValueError("encoder window frame counts must be positive")
        if self.window_frames % self.alignment_frames:
            raise ValueError(
                f"encoder window of {self.window_frames} frames is not a multiple of "
                f"the {self.alignment_frames}-frame encoder block"
            )
        if not 0 <= self.merge_tail_below_frames <= self.window_frames:
            raise ValueError("encoder tail merge threshold must be within one window")


class EncoderWindowConfig(msgspec.Struct, frozen=True):
    """Use the model's sample and token counts when splitting audio into windows."""

    sample_rate: int
    window_samples: int
    window_tokens: int
    # Preceding samples let feature extraction compute the first frame correctly.
    leading_context_samples: int

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.window_tokens <= 0:
            raise ValueError(
                "audio window sample rate and token count must be positive"
            )
        else:
            return

    @property
    def window_seconds(self) -> float:
        return self.window_samples / self.sample_rate


def build_encoder_window_items(
    window_processor: EncoderWindowMixin,
    *,
    item: MultimodalDataItem,
    input_ids: torch.Tensor,
    placeholder_token_id: int,
    config: EncoderWindowConfig,
    leading_context_samples: int = 0,
    processor: Optional[ProcessorMixin] = None,
) -> tuple[list[MultimodalDataItem], torch.Tensor]:
    """Split audio features and placeholder tokens so complete windows can be cached."""
    spec = window_processor.encoder_window_spec()
    feature = torch.as_tensor(item.feature)
    mask = torch.as_tensor(item.model_specific_data[spec.mask_key])
    if feature.ndim != 3 or mask.ndim != 2:
        raise ValueError(
            "encoder windowing requires batched audio features and a frame mask"
        )
    feature_time_dim = spec.feature_time_dim % feature.ndim
    if (
        feature.shape[0] != 1
        or feature_time_dim == 0
        or mask.shape != (1, feature.shape[feature_time_dim])
    ):
        raise ValueError(
            "encoder windowing requires one audio feature batch and matching frame mask"
        )
    valid_frames = int(mask.sum())
    expected_mask = (
        torch.arange(mask.shape[-1], device=mask.device)[None, :] < valid_frames
    )
    if not torch.equal(mask.bool(), expected_mask):
        raise ValueError("encoder windowing requires a contiguous valid-frame prefix")
    samples_per_frame = config.window_samples // spec.window_frames
    if leading_context_samples not in (0, config.leading_context_samples):
        raise ValueError(
            "encoder window leading context must match the server frame configuration"
        )
    else:
        leading_frames = leading_context_samples // samples_per_frame
    encoder_frame_count = valid_frames - leading_frames
    if encoder_frame_count <= 0:
        raise ValueError(
            "encoder windowing requires non-empty audio after leading context"
        )

    input_ids = torch.as_tensor(input_ids).flatten()
    audio_token_positions = (input_ids == placeholder_token_id).nonzero(as_tuple=True)[
        0
    ]
    original_audio_token_count = window_processor.encoder_window_output_lengths(
        [valid_frames], processor=processor
    )[0]
    if (
        audio_token_positions.numel() != original_audio_token_count
        or original_audio_token_count <= 0
        or int(audio_token_positions[-1] - audio_token_positions[0]) + 1
        != original_audio_token_count
    ):
        raise ValueError(
            "encoder windowing requires one expanded audio-token span matching the features"
        )
    audio_token_start = int(audio_token_positions[0])
    audio_token_end = audio_token_start + original_audio_token_count

    # Keep short tails with a complete window so cache hits on other windows
    # cannot change how the encoder pads the tail.
    window_end_frames = list(
        range(spec.window_frames, encoder_frame_count, spec.window_frames)
    )
    tail_frames = encoder_frame_count % spec.window_frames
    if window_end_frames and 0 < tail_frames < spec.merge_tail_below_frames:
        window_end_frames[-1] = encoder_frame_count
    else:
        window_end_frames.append(encoder_frame_count)
    window_frame_counts = [
        end - start for start, end in zip([0] + window_end_frames, window_end_frames)
    ]
    window_token_counts = window_processor.encoder_window_output_lengths(
        window_frame_counts, processor=processor
    )
    encoder_token_count = window_processor.encoder_window_output_lengths(
        [encoder_frame_count], processor=processor
    )[0]
    if sum(window_token_counts) != encoder_token_count:
        raise ValueError(
            "encoder window token counts must preserve the whole-audio length"
        )
    else:
        window_items = []
    frame_start, token_start = leading_frames, audio_token_start
    for frame_count, token_count in zip(window_frame_counts, window_token_counts):
        window_item = window_processor.make_encoder_window_item(
            item,
            feature=feature.narrow(
                feature_time_dim, frame_start, frame_count
            ).contiguous(),
            mask=mask[..., frame_start : frame_start + frame_count].contiguous(),
            offsets=[(token_start, token_start + token_count - 1)],
        )
        window_items.append(window_item)
        frame_start += frame_count
        token_start += token_count

    if encoder_token_count != original_audio_token_count:
        # Feature extraction includes preceding audio; remove its placeholder
        # tokens because those frames are excluded from the encoder input.
        input_ids = torch.cat(
            [
                input_ids[: audio_token_start + encoder_token_count],
                input_ids[audio_token_end:],
            ]
        )
    return window_items, input_ids


# Derive window sizes from the model so request options cannot change alignment.
ENCODER_WINDOW_KWARG = "encoder_window"
LEADING_CONTEXT_KEY = "leading_context_samples"


class EncoderWindowMixin(BaseMultimodalProcessor):
    """Reuse feature extraction and split its output into cacheable audio windows."""

    cached_window_config: Optional[EncoderWindowConfig] = None

    def encoder_window_spec(self) -> EncoderWindowSpec:
        raise NotImplementedError

    def encoder_window_config(
        self, *, processor: Optional[ProcessorMixin] = None
    ) -> EncoderWindowConfig:
        if self.audio_config.get("truncation", False):
            raise ValueError("encoder windowing requires audio truncation=False")
        if self.cached_window_config is not None:
            return self.cached_window_config
        processor = self._processor if processor is None else processor
        spec = self.encoder_window_spec()
        feature_extractor = processor.feature_extractor
        hop_length, n_fft = (
            int(feature_extractor.hop_length),
            int(feature_extractor.n_fft),
        )
        if hop_length <= 0 or n_fft <= 0:
            raise ValueError("feature extractor hop_length and n_fft must be positive")
        self.cached_window_config = EncoderWindowConfig(
            sample_rate=int(feature_extractor.sampling_rate),
            window_samples=spec.window_frames * hop_length,
            window_tokens=self.encoder_window_output_lengths(
                [spec.window_frames], processor=processor
            )[0],
            # The first STFT frame needs half an FFT window of preceding samples.
            leading_context_samples=-(-(n_fft // 2) // hop_length) * hop_length,
        )
        return self.cached_window_config

    def encoder_window_output_lengths(
        self,
        frame_lengths: Sequence[int],
        *,
        processor: Optional[ProcessorMixin] = None,
    ) -> list[int]:
        processor = self._processor if processor is None else processor
        return [
            int(token_count)
            for token_count in processor._get_feat_extract_output_lengths(frame_lengths)
        ]

    def make_encoder_window_item(
        self,
        source: MultimodalDataItem,
        *,
        feature: torch.Tensor,
        mask: torch.Tensor,
        offsets: Sequence[tuple[int, int]],
    ) -> MultimodalDataItem:
        item = msgspec.structs.replace(
            source,
            feature=feature,
            offsets=list(offsets),
            hash=None,
            pad_value=None,
            model_specific_data={
                **source.model_specific_data,
                self.encoder_window_spec().mask_key: mask,
            },
        )
        item.set_hash(hash_feature([feature, mask]))
        return item

    def process_and_combine_mm_data(
        self,
        base_output: BaseMultiModalProcessorOutput,
        mm_tokens: MultimodalSpecialTokens,
        *,
        processor: Optional[ProcessorMixin] = None,
        mm_processor_kwargs: Optional[dict[str, JsonValue]] = None,
        **kwargs: object,
    ) -> tuple[list[MultimodalDataItem], torch.Tensor, dict[str, MultimodalDataValue]]:
        encoder_window_options = (
            mm_processor_kwargs.get(ENCODER_WINDOW_KWARG)
            if mm_processor_kwargs
            else None
        )
        if encoder_window_options is None:
            return super().process_and_combine_mm_data(
                base_output, mm_tokens, processor=processor, **kwargs
            )
        if not isinstance(encoder_window_options, dict):
            raise ValueError(
                f"mm_processor_kwargs[{ENCODER_WINDOW_KWARG!r}] must be a mapping"
            )
        audios = base_output.audios or []
        if len(audios) != 1 or len(base_output.organize_results()) != 1:
            raise ValueError(
                "encoder windowing requires exactly one audio input and no other media"
            )
        if not isinstance(audios[0], np.ndarray) or audios[0].ndim != 1:
            raise ValueError("encoder windowing requires mono raw audio samples")
        processor, _ = self._resolve_processor(processor)
        config = self.encoder_window_config(processor=processor)
        # Extract features once so splitting does not change preprocessing.
        # Split before hashing so each window can reuse cached encoder output.
        audio_items, input_ids, processor_output = self._process_and_collect_mm_items(
            input_text=base_output.input_text,
            audios=audios,
            processor=processor,
            **kwargs,
        )
        if len(audio_items) != 1 or audio_items[0].modality != Modality.AUDIO:
            raise ValueError("encoder windowing requires one processed audio item")
        audio_token_id = mm_tokens.get_token_id_by_modality(Modality.AUDIO)
        if audio_token_id is None:
            raise ValueError("encoder windowing requires an audio placeholder token id")
        window_items, input_ids = build_encoder_window_items(
            self,
            item=audio_items[0],
            input_ids=input_ids,
            placeholder_token_id=audio_token_id,
            config=config,
            leading_context_samples=int(
                encoder_window_options.get(LEADING_CONTEXT_KEY, 0)
            ),
            processor=processor,
        )
        return (
            self._finalize_mm_items(window_items, images=None),
            input_ids,
            processor_output,
        )
