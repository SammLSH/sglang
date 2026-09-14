"""Opt-in encoder windows for existing multimodal caches.

Models declare window geometry. The capability separates feature preparation
from the shared window builder; the mixin supplies default frontend methods and
integrates the builder with BaseMultimodalProcessor. Encoder execution and cache
storage remain in the existing multimodal scheduler.
"""

from __future__ import annotations

from typing import (
    Any,
    Iterator,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

import msgspec
import numpy as np
import torch

from sglang.srt.managers.mm_utils import hash_feature
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem


class EncoderWindowSpec(msgspec.Struct, frozen=True):
    """Independently encodable feature frames and their encoder block alignment."""

    window_frames: int
    alignment_frames: int


class EncoderWindowConfig(msgspec.Struct, frozen=True):
    """Server-resolved window sizes in samples and encoder output tokens."""

    sample_rate: int
    window_samples: int
    window_tokens: int
    # Shorter tail bodies are merged into the preceding complete window.
    min_tail_samples: int
    # Extraction context before a window, excluded from its encoder input.
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


class WindowFeatures(NamedTuple):
    """One feature tensor, valid-frame mask, and token count per window."""

    features: list[torch.Tensor]
    masks: list[torch.Tensor]
    token_counts: list[int]


@runtime_checkable
class EncoderWindowCapability(Protocol):
    """What the window builder needs from a processor.

    EncoderWindowMixin supplies defaults except for encoder_window_spec. The
    defaults use a frontend with centered frames, sampling_rate, hop_length,
    and n_fft. Models override hooks when their frontend or item layout differs.
    """

    def encoder_window_spec(self) -> EncoderWindowSpec: ...

    def encoder_window_config(
        self, *, processor: Any = None
    ) -> EncoderWindowConfig: ...

    def encoder_window_feature_extractor(self, *, processor: Any = None) -> Any: ...

    def extract_encoder_window_features(
        self,
        windows: Sequence[np.ndarray],
        *,
        leading_frames: Optional[Sequence[int]] = None,
        processor: Any = None,
    ) -> WindowFeatures:
        """Extract each window independently, removing its leading context frames."""
        ...

    def encoder_window_output_lengths(
        self, frame_lens: Sequence[int], *, processor: Any = None
    ) -> list[int]: ...

    def make_encoder_window_item(
        self,
        feature: torch.Tensor,
        mask: torch.Tensor,
        offsets: Sequence[tuple[int, int]],
    ) -> MultimodalDataItem: ...

    def encoder_window_placeholder_token_id(self, mm_tokens: Any) -> int: ...


def resolve_encoder_window_config(
    cap: EncoderWindowCapability, *, processor: Any = None
) -> EncoderWindowConfig:
    """Resolve geometry and probe standalone and leading-context feature shapes."""
    spec = cap.encoder_window_spec()
    feature_extractor = cap.encoder_window_feature_extractor(processor=processor)
    if spec.window_frames <= 0 or spec.alignment_frames <= 0:
        raise ValueError("encoder window frame counts must be positive")
    if spec.window_frames % spec.alignment_frames:
        raise ValueError(
            f"encoder window of {spec.window_frames} frames is not a multiple of "
            f"the {spec.alignment_frames}-frame encoder block"
        )
    hop_length, n_fft = int(feature_extractor.hop_length), int(feature_extractor.n_fft)
    if hop_length <= 0 or n_fft <= 0:
        raise ValueError("feature extractor hop_length and n_fft must be positive")
    if float(getattr(feature_extractor, "dither", 0.0)) != 0.0:
        raise ValueError(
            "encoder windowing requires deterministic feature extraction (dither must be 0)"
        )

    # Centered frames need preceding half-frame samples, rounded up to whole hops.
    leading_context_samples = -(-(n_fft // 2) // hop_length) * hop_length
    config = EncoderWindowConfig(
        sample_rate=int(feature_extractor.sampling_rate),
        window_samples=spec.window_frames * hop_length,
        window_tokens=cap.encoder_window_output_lengths(
            [spec.window_frames], processor=processor
        )[0],
        min_tail_samples=n_fft,
        leading_context_samples=leading_context_samples,
    )
    for leading in (0, leading_context_samples):
        probe = cap.extract_encoder_window_features(
            [np.zeros(config.window_samples + leading, dtype=np.float32)],
            leading_frames=[leading // hop_length],
            processor=processor,
        )
        _validate_audio_window_features(
            probe.features[0], probe.masks[0], spec.window_frames
        )
        if probe.token_counts[0] != config.window_tokens:
            raise ValueError(
                f"feature extractor produced {probe.token_counts[0]} encoder tokens for one window; "
                f"the output-length function predicts {config.window_tokens}"
            )
    return config


def build_encoder_window_items(
    cap: EncoderWindowCapability,
    *,
    samples: np.ndarray,
    input_ids: torch.Tensor,
    placeholder_token_id: int,
    config: EncoderWindowConfig,
    leading_context_samples: int = 0,
    processor: Any = None,
) -> tuple[list[MultimodalDataItem], torch.Tensor]:
    """Split audio, prepare independent windows, and expand one audio placeholder.

    Complete windows precede the tail. A sub-minimum remainder is merged into
    the last window and changes its identity. Leading samples provide extraction
    context and are excluded from encoder positions and prompt token counts.
    """
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError("encoder windowing requires mono audio samples")
    if leading_context_samples not in (0, config.leading_context_samples):
        raise ValueError(
            f"leading context must be 0 or {config.leading_context_samples} samples, "
            f"got {leading_context_samples}"
        )
    if samples.size <= leading_context_samples:
        raise ValueError("encoder windowing requires non-empty audio")

    input_ids = torch.as_tensor(input_ids).flatten()
    positions = (input_ids == placeholder_token_id).nonzero(as_tuple=True)[0]
    if positions.numel() != 1:
        raise ValueError(
            "encoder windowing expects exactly one unexpanded audio placeholder, "
            f"found {int(positions.numel())}"
        )
    position = int(positions[0])
    windows = list(_iter_audio_windows(samples, config, leading_context_samples))
    hop_length = int(
        cap.encoder_window_feature_extractor(processor=processor).hop_length
    )
    prepared = cap.extract_encoder_window_features(
        [window for window, _, _ in windows],
        leading_frames=[context // hop_length for _, context, _ in windows],
        processor=processor,
    )
    if (
        not len(windows)
        == len(prepared.features)
        == len(prepared.masks)
        == len(prepared.token_counts)
    ):
        raise ValueError(
            "feature extraction must return one feature, mask, and token count per window"
        )

    items: list[MultimodalDataItem] = []
    start = position
    for (_, _, complete), feature, mask, count in zip(
        windows, prepared.features, prepared.masks, prepared.token_counts
    ):
        count = int(count)
        if count <= 0:
            raise ValueError("audio window produced no encoder tokens")
        if complete:
            _validate_audio_window_features(
                feature, mask, cap.encoder_window_spec().window_frames
            )
            if count != config.window_tokens:
                raise ValueError(
                    f"complete audio window produced {count} encoder tokens; "
                    f"expected {config.window_tokens}"
                )
        item = cap.make_encoder_window_item(feature, mask, [(start, start + count - 1)])
        if (
            item.modality != Modality.AUDIO
            or item.hash is None
            or item.pad_value is None
        ):
            raise ValueError("audio window processor must return a hashed audio item")
        items.append(item)
        start += count

    expanded_input_ids = torch.cat(
        [
            input_ids[:position],
            torch.full(
                (start - position,),
                placeholder_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            ),
            input_ids[position + 1 :],
        ]
    )
    return items, expanded_input_ids


def _iter_audio_windows(
    samples: np.ndarray, config: EncoderWindowConfig, leading_context_samples: int
) -> Iterator[tuple[np.ndarray, int, bool]]:
    """Yield each audio view, its leading sample count, and whether it is complete."""
    body_samples = samples.size - leading_context_samples
    complete_count, tail_samples = divmod(body_samples, config.window_samples)
    if 0 < tail_samples < config.min_tail_samples:
        if complete_count == 0:
            raise ValueError(
                f"audio shorter than {config.min_tail_samples} samples cannot be windowed"
            )
        complete_count -= 1

    window_ends = [
        leading_context_samples + (index + 1) * config.window_samples
        for index in range(complete_count)
    ]
    if not window_ends or window_ends[-1] < samples.size:
        window_ends.append(samples.size)

    start = leading_context_samples
    for index, end in enumerate(window_ends):
        context_samples = min(config.leading_context_samples, start)
        yield (
            samples[start - context_samples : end],
            context_samples,
            index < complete_count,
        )
        start = end


# Requests select windowing and carry leading context; sizes stay server-owned.
ENCODER_WINDOW_KWARG = "encoder_window"
LEADING_CONTEXT_KEY = "leading_context_samples"
FEATURE_ATTENTION_MASK_KEY = "feature_attention_mask"


def encoder_window_kwargs(*, leading_context_samples: int = 0) -> dict:
    """Build the mm_processor_kwargs payload for one windowed request."""
    return {ENCODER_WINDOW_KWARG: {LEADING_CONTEXT_KEY: int(leading_context_samples)}}


class EncoderWindowMixin:
    """Opt-in windowing for BaseMultimodalProcessor subclasses.

    Declare encoder_window_spec to use the default feature preparation. Override
    capability methods only where the frontend or multimodal item format differs.
    The request's processor argument selects the executor-local frontend.
    """

    _cached_window_config: Optional[EncoderWindowConfig] = None

    def encoder_window_spec(self) -> EncoderWindowSpec:
        raise NotImplementedError

    def encoder_window_config(self, *, processor: Any = None) -> EncoderWindowConfig:
        if self._cached_window_config is None:
            self._cached_window_config = resolve_encoder_window_config(
                self, processor=processor
            )
        return self._cached_window_config

    def encoder_window_feature_extractor(self, *, processor: Any = None) -> Any:
        processor = self._processor if processor is None else processor
        return processor.feature_extractor

    def encoder_window_extract_kwargs(self, *, processor: Any = None) -> dict:
        extractor = self.encoder_window_feature_extractor(processor=processor)
        required = {
            "return_attention_mask": True,
            "return_tensors": "pt",
            "truncation": False,
            "padding": "longest",
            "sampling_rate": int(extractor.sampling_rate),
        }
        kwargs = dict(self.audio_config or {})
        conflicts = sorted(
            key
            for key, value in required.items()
            if key in kwargs and kwargs[key] != value
        )
        if conflicts:
            raise ValueError(
                "encoder windowing fixes the audio preprocessing values for "
                + ", ".join(conflicts)
                + "; remove them from --mm-process-config"
            )
        kwargs.update(required)
        return kwargs

    def extract_encoder_window_features(
        self,
        windows: Sequence[np.ndarray],
        *,
        leading_frames: Optional[Sequence[int]] = None,
        processor: Any = None,
    ) -> WindowFeatures:
        extractor = self.encoder_window_feature_extractor(processor=processor)
        kwargs = self.encoder_window_extract_kwargs(processor=processor)
        if leading_frames is None:
            leading_frames = [0] * len(windows)
        features, masks, token_counts = [], [], []
        # Independent calls keep cache identity independent of request batch size.
        for window, leading in zip(windows, leading_frames):
            output = extractor([np.asarray(window, dtype=np.float32)], **kwargs)
            feature = torch.as_tensor(output["input_features"])
            mask = torch.as_tensor(output["attention_mask"])
            if leading:
                feature = feature[..., leading:].contiguous()
                mask = mask[..., leading:].contiguous()
            features.append(feature)
            masks.append(mask)
            token_counts.append(
                self.encoder_window_output_lengths(
                    [int(mask.sum())], processor=processor
                )[0]
            )
        return WindowFeatures(features, masks, token_counts)

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
        feature: torch.Tensor,
        mask: torch.Tensor,
        offsets: Sequence[tuple[int, int]],
    ) -> MultimodalDataItem:
        item = MultimodalDataItem(
            modality=Modality.AUDIO,
            offsets=list(offsets),
            feature=feature,
            model_specific_data={FEATURE_ATTENTION_MASK_KEY: mask},
        )
        # Padded features with different valid lengths must not share cached outputs.
        item.set_hash(hash_feature([feature, mask]))
        return item

    def encoder_window_placeholder_token_id(self, mm_tokens: Any) -> int:
        token_id = mm_tokens.get_token_id_by_modality(Modality.AUDIO)
        if token_id is None:
            raise ValueError("encoder windowing requires an audio placeholder token id")
        return token_id

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
        if not isinstance(audios[0], np.ndarray):
            raise ValueError("encoder windowing requires raw audio samples")

        processor, tokenizer = self._resolve_processor(kwargs.get("processor"))
        config = self.encoder_window_config(processor=processor)
        input_ids = self._tokenize_for_encoder_windows(
            tokenizer, base_output.input_text
        )
        items, input_ids = build_encoder_window_items(
            self,
            samples=audios[0],
            input_ids=input_ids,
            placeholder_token_id=self.encoder_window_placeholder_token_id(mm_tokens),
            config=config,
            leading_context_samples=int(window_request.get(LEADING_CONTEXT_KEY, 0)),
            processor=processor,
        )
        return self._finalize_mm_items(items, images=None), input_ids, {}

    def _tokenize_for_encoder_windows(self, tokenizer, input_text: str) -> torch.Tensor:
        # Preserve the base special-token policy, avoiding a duplicate template BOS.
        add_special_tokens = True
        if self._tokenizer_auto_adds_specials:
            bos = getattr(tokenizer, "bos_token", None)
            if bos and input_text.startswith(bos):
                add_special_tokens = False
        return tokenizer(
            input_text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids.flatten()


def _validate_audio_window_features(
    feature: torch.Tensor, mask: torch.Tensor, window_frames: int
) -> None:
    frames, width = int(mask.sum()), int(feature.shape[-1])
    if frames != window_frames or width != window_frames:
        raise ValueError(
            f"feature extractor produced {frames} valid frames in a {width}-frame tensor "
            f"for one window; expected {window_frames} unpadded frames"
        )
