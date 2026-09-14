"""Split whole-audio processor features for existing multimodal caches.

Models declare independently encodable frame boundaries and output lengths.
The normal processor extracts the whole request once; this module partitions
its features and audio-token span before hashing and transport. Encoder work
and cache storage stay in the existing multimodal scheduler.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import msgspec
import numpy as np
import torch

from sglang.srt.managers.mm_utils import hash_feature
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem


class EncoderWindowSpec(msgspec.Struct, frozen=True):
    """Independent encoder windows and the processor's feature/mask layout."""

    window_frames: int
    alignment_frames: int
    feature_time_dim: int = -1
    mask_key: str = "feature_attention_mask"

    def __post_init__(self):
        if self.window_frames <= 0 or self.alignment_frames <= 0:
            raise ValueError("encoder window frame counts must be positive")
        if self.window_frames % self.alignment_frames:
            raise ValueError(
                f"encoder window of {self.window_frames} frames is not a multiple of "
                f"the {self.alignment_frames}-frame encoder block"
            )


class EncoderWindowConfig(msgspec.Struct, frozen=True):
    """Server-resolved window sizes in samples and encoder output tokens."""

    sample_rate: int
    window_samples: int
    window_tokens: int
    # Merge a shorter tail into the preceding window's cache item.
    min_tail_samples: int
    # Extraction context before the request body, excluded from encoder input.
    leading_context_samples: int

    def __post_init__(self):
        if (
            min(
                self.sample_rate,
                self.window_samples,
                self.window_tokens,
                self.min_tail_samples,
            )
            <= 0
        ):
            raise ValueError(
                "audio window sample rate, sizes, and token count must be positive"
            )
        if self.leading_context_samples < 0:
            raise ValueError("audio window leading context must be non-negative")

    @property
    def window_seconds(self) -> float:
        return self.window_samples / self.sample_rate


@runtime_checkable
class EncoderWindowCapability(Protocol):
    """Window metadata and item construction, independent of feature extraction.

    The mixin supplies defaults for frame-based audio processors. A model with a
    different sample/frame mapping or per-window metadata can override those parts
    while keeping its normal whole-audio processor.
    """

    def encoder_window_spec(self) -> EncoderWindowSpec: ...

    def encoder_window_config(
        self, *, processor: Any = None
    ) -> EncoderWindowConfig: ...

    def encoder_window_output_lengths(
        self, frame_lens: Sequence[int], *, processor: Any = None
    ) -> list[int]: ...

    def make_encoder_window_item(
        self,
        source: MultimodalDataItem,
        *,
        feature: torch.Tensor,
        mask: torch.Tensor,
        offsets: Sequence[tuple[int, int]],
    ) -> MultimodalDataItem: ...


def build_encoder_window_items(
    cap: EncoderWindowCapability,
    *,
    item: MultimodalDataItem,
    input_ids: torch.Tensor,
    placeholder_token_id: int,
    config: EncoderWindowConfig,
    leading_context_samples: int = 0,
    processor: Any = None,
) -> tuple[list[MultimodalDataItem], torch.Tensor]:
    """Partition already-extracted features and an expanded audio-token span.

    Feature values retain the whole request's preprocessing semantics. Growing
    or rolling audio can change old features; their hashes must then miss.
    Leading context is removed after extraction and excluded from token counts.
    """
    spec = cap.encoder_window_spec()
    feature = torch.as_tensor(item.feature)
    mask = torch.as_tensor(getattr(item, spec.mask_key))
    if feature.ndim != 3 or mask.ndim != 2:
        raise ValueError(
            "encoder windowing requires batched audio features and a frame mask"
        )
    time_dim = spec.feature_time_dim % feature.ndim
    if (
        feature.shape[0] != 1
        or time_dim == 0
        or mask.shape != (1, feature.shape[time_dim])
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

    samples_per_frame, remainder = divmod(config.window_samples, spec.window_frames)
    if remainder or samples_per_frame <= 0:
        raise ValueError("encoder window samples must align to feature frames")
    if (
        leading_context_samples not in (0, config.leading_context_samples)
        or leading_context_samples % samples_per_frame
    ):
        raise ValueError(
            "encoder window leading context must match the server frame configuration"
        )
    leading_frames = leading_context_samples // samples_per_frame
    body_frames = valid_frames - leading_frames
    if body_frames <= 0:
        raise ValueError(
            "encoder windowing requires non-empty audio after leading context"
        )

    input_ids = torch.as_tensor(input_ids).flatten()
    positions = (input_ids == placeholder_token_id).nonzero(as_tuple=True)[0]
    original_tokens = cap.encoder_window_output_lengths(
        [valid_frames], processor=processor
    )[0]
    if (
        positions.numel() != original_tokens
        or original_tokens <= 0
        or int(positions[-1] - positions[0]) + 1 != original_tokens
    ):
        raise ValueError(
            "encoder windowing requires one expanded audio-token span matching the features"
        )
    audio_start = int(positions[0])
    audio_end = audio_start + original_tokens

    # Keep the existing tiny-tail grouping while cutting features, not PCM.
    window_ends = list(range(spec.window_frames, body_frames + 1, spec.window_frames))
    tail_frames = body_frames % spec.window_frames
    if window_ends and 0 < tail_frames * samples_per_frame < config.min_tail_samples:
        window_ends.pop()
    if not window_ends or window_ends[-1] < body_frames:
        window_ends.append(body_frames)
    frame_lengths = [end - start for start, end in zip([0] + window_ends, window_ends)]
    token_counts = cap.encoder_window_output_lengths(frame_lengths, processor=processor)
    body_tokens = cap.encoder_window_output_lengths([body_frames], processor=processor)[
        0
    ]
    if (
        len(token_counts) != len(frame_lengths)
        or any(count <= 0 for count in token_counts)
        or sum(token_counts) != body_tokens
        or body_tokens > original_tokens
        or any(
            frames == spec.window_frames and count != config.window_tokens
            for frames, count in zip(frame_lengths, token_counts)
        )
    ):
        raise ValueError(
            "encoder window token counts must preserve the whole-audio length"
        )

    items = []
    frame_start, token_start = leading_frames, audio_start
    for frames, count in zip(frame_lengths, token_counts):
        window_item = cap.make_encoder_window_item(
            item,
            feature=feature.narrow(time_dim, frame_start, frames).contiguous(),
            mask=mask[..., frame_start : frame_start + frames].contiguous(),
            offsets=[(token_start, token_start + count - 1)],
        )
        items.append(window_item)
        frame_start += frames
        token_start += count

    if body_tokens != original_tokens:
        # The processor expanded the context frames too. Keep only body tokens
        # while preserving the original text tokens on both sides of the audio.
        input_ids = torch.cat(
            [input_ids[: audio_start + body_tokens], input_ids[audio_end:]]
        )
    return items, input_ids


# Requests select windowing and carry leading context; sizes stay server-owned.
ENCODER_WINDOW_KWARG = "encoder_window"
LEADING_CONTEXT_KEY = "leading_context_samples"


def encoder_window_kwargs(*, leading_context_samples: int = 0) -> dict:
    """Build the mm_processor_kwargs payload for one windowed request."""
    return {ENCODER_WINDOW_KWARG: {LEADING_CONTEXT_KEY: int(leading_context_samples)}}


class EncoderWindowMixin:
    """Partition normal processor output using model-declared encoder windows."""

    _cached_window_config: Optional[EncoderWindowConfig] = None

    def encoder_window_spec(self) -> EncoderWindowSpec:
        raise NotImplementedError

    def encoder_window_config(self, *, processor: Any = None) -> EncoderWindowConfig:
        if self._cached_window_config is None:
            processor = self._processor if processor is None else processor
            spec = self.encoder_window_spec()
            extractor = processor.feature_extractor
            hop_length, n_fft = int(extractor.hop_length), int(extractor.n_fft)
            if hop_length <= 0 or n_fft <= 0:
                raise ValueError(
                    "feature extractor hop_length and n_fft must be positive"
                )
            self._cached_window_config = EncoderWindowConfig(
                sample_rate=int(extractor.sampling_rate),
                window_samples=spec.window_frames * hop_length,
                window_tokens=self.encoder_window_output_lengths(
                    [spec.window_frames], processor=processor
                )[0],
                min_tail_samples=n_fft,
                # Preserve preceding half-frame samples, rounded to whole hops.
                leading_context_samples=-(-(n_fft // 2) // hop_length) * hop_length,
            )
        return self._cached_window_config

    def encoder_window_output_lengths(
        self, frame_lens: Sequence[int], *, processor: Any = None
    ) -> list[int]:
        processor = self._processor if processor is None else processor
        return [
            int(count)
            for count in processor._get_feat_extract_output_lengths(frame_lens)
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
        base_output,
        mm_tokens,
        *,
        mm_processor_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        window_request = (
            mm_processor_kwargs.get(ENCODER_WINDOW_KWARG)
            if mm_processor_kwargs
            else None
        )
        if window_request is None:
            return super().process_and_combine_mm_data(base_output, mm_tokens, **kwargs)
        if not isinstance(window_request, dict):
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

        processor, _ = self._resolve_processor(kwargs.pop("processor", None))
        config = self.encoder_window_config(processor=processor)
        # Reuse the normal whole-audio frontend, including preprocessing options
        # and tokenization. Partition before hashes and transport are finalized.
        collected, input_ids, ret = self._process_and_collect_mm_items(
            input_text=base_output.input_text,
            audios=audios,
            processor=processor,
            **kwargs,
        )
        if len(collected) != 1 or collected[0].modality != Modality.AUDIO:
            raise ValueError("encoder windowing requires one processed audio item")
        token_id = mm_tokens.get_token_id_by_modality(Modality.AUDIO)
        if token_id is None:
            raise ValueError("encoder windowing requires an audio placeholder token id")
        items, input_ids = build_encoder_window_items(
            self,
            item=collected[0],
            input_ids=input_ids,
            placeholder_token_id=token_id,
            config=config,
            leading_context_samples=int(window_request.get(LEADING_CONTEXT_KEY, 0)),
            processor=processor,
        )
        return self._finalize_mm_items(items, images=None), input_ids, ret
