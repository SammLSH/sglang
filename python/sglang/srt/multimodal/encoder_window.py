"""Prepare independently encodable audio windows for existing multimodal caches.

The shared pipeline plans sample ranges and assigns prompt token offsets. An
``AudioWindowProcessor`` supplies the server's window config and prepares each
window's features, model-specific fields, and cache identity. Custom processor
paths can call ``build_audio_window_items`` directly; the mixin integrates it
with ``BaseMultimodalProcessor`` and provides default window feature preparation.

Aligned complete windows with the same preceding samples retain their identity.
Feature extraction still runs on every request. Windowed features need not equal
whole-clip features; encoder independence and transcription quality require
model-specific validation. Encoder execution and caching remain in the scheduler.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
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


class AudioEncoderWindowConfig(msgspec.Struct, frozen=True):
    """Server-resolved window sizes in samples and encoder output tokens.

    Callers keep starts window-aligned and retain the same leading samples
    when rolling. Feature frame sizes and frontend padding are model-owned.
    """

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


class PreparedAudioWindow(NamedTuple):
    """One fresh, hashed audio item and its predicted encoder output length.

    The item includes every encoder input in its cache identity, including any
    valid-length masks. The shared pipeline fills its prompt offsets later.
    """

    item: MultimodalDataItem
    token_count: int


@runtime_checkable
class AudioWindowProcessor(Protocol):
    """Window capability independent of processor inheritance and frontend type.

    Implementations opt in only when complete windows are independently
    encodable. ``processor`` supplies an executor-local frontend when applicable.
    """

    def audio_window_config(
        self, *, processor: Any = None
    ) -> AudioEncoderWindowConfig: ...

    def prepare_audio_window(
        self,
        samples: np.ndarray,
        *,
        leading_context_samples: int = 0,
        processor: Any = None,
    ) -> PreparedAudioWindow:
        """Prepare audio including its leading extraction context.

        Exclude that context from encoder positions. Return a fresh item whose
        encoder output is independent of prompt offsets and other cache misses.
        """
        ...


def build_audio_window_items(
    window_processor: AudioWindowProcessor,
    *,
    samples: np.ndarray,
    input_ids: torch.Tensor,
    placeholder_token_id: int,
    config: AudioEncoderWindowConfig,
    leading_context_samples: int = 0,
    processor: Any = None,
) -> tuple[list[MultimodalDataItem], torch.Tensor]:
    """Prepare windows and expand one audio placeholder to their token spans.

    ``samples`` is mono audio at the configured rate, optionally prefixed with
    the first window's extraction context. Complete windows precede the tail;
    a sub-minimum remainder is merged into the last window and changes its
    identity. Each item receives one contiguous, inclusive prompt offset.

    This function accepts any ``AudioWindowProcessor``. It neither tokenizes
    text nor assumes a feature shape, mask name, or frontend implementation.
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

    items: list[MultimodalDataItem] = []
    start = position
    for window, context_samples, complete in _iter_audio_windows(
        samples, config, leading_context_samples
    ):
        prepared = window_processor.prepare_audio_window(
            window, leading_context_samples=context_samples, processor=processor
        )
        item, count = prepared.item, int(prepared.token_count)
        if count <= 0:
            raise ValueError("audio window produced no encoder tokens")
        if complete and count != config.window_tokens:
            raise ValueError(
                f"complete audio window produced {count} encoder tokens; "
                f"expected {config.window_tokens}"
            )
        if (
            item.modality != Modality.AUDIO
            or item.hash is None
            or item.pad_value is None
        ):
            raise ValueError("audio window processor must return a hashed audio item")
        item.offsets = [(start, start + count - 1)]
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
    samples: np.ndarray,
    config: AudioEncoderWindowConfig,
    leading_context_samples: int,
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


# Requests select windowing and carry leading context; window sizes stay server-owned.
ENCODER_WINDOW_KWARG = "encoder_window"
LEADING_CONTEXT_KEY = "leading_context_samples"


def build_audio_window_processor_kwargs(*, leading_context_samples: int = 0) -> dict:
    """Build the mm_processor_kwargs payload for one windowed request."""
    return {ENCODER_WINDOW_KWARG: {LEADING_CONTEXT_KEY: int(leading_context_samples)}}


class WindowedAudioProcessorMixin:
    """Default window integration for ``BaseMultimodalProcessor`` subclasses.

    List the mixin before the base class and declare ``audio_encoder_window_spec``
    to use the default feature preparation. It uses the resolved processor's
    ``feature_extractor`` and ``_get_feat_extract_output_lengths``, and stores the
    mask as ``feature_attention_mask``.

    These defaults expect centered feature frames with a fixed hop size and
    extractor attributes ``sampling_rate``, ``hop_length``, and ``n_fft``.

    Other frontends can override ``resolve_audio_window_config`` and
    ``prepare_audio_window`` without declaring a frame-based spec. Custom
    processing paths can implement ``AudioWindowProcessor`` and call the shared
    builder directly.
    """

    _cached_window_config: Optional[AudioEncoderWindowConfig] = None

    def audio_encoder_window_spec(self) -> AudioEncoderWindowSpec:
        """Declare independently encodable feature frames for the default frontend."""
        raise NotImplementedError

    def resolve_audio_window_config(
        self, *, processor: Any = None
    ) -> AudioEncoderWindowConfig:
        """Resolve the default window sizes and validate feature shapes."""
        processor = self._processor if processor is None else processor
        return resolve_default_audio_window_config(
            self.audio_encoder_window_spec(),
            feature_extractor=processor.feature_extractor,
            output_lengths=processor._get_feat_extract_output_lengths,
            audio_config=self.audio_config,
        )

    def prepare_audio_window(
        self,
        samples: np.ndarray,
        *,
        leading_context_samples: int = 0,
        processor: Any = None,
    ) -> PreparedAudioWindow:
        """Prepare one window using the request's resolved feature extractor."""
        processor = self._processor if processor is None else processor
        return prepare_default_audio_window(
            samples,
            feature_extractor=processor.feature_extractor,
            output_lengths=processor._get_feat_extract_output_lengths,
            window_frames=self.audio_encoder_window_spec().window_frames,
            mask_key="feature_attention_mask",
            audio_config=self.audio_config,
            leading_context_samples=leading_context_samples,
        )

    def audio_window_config(self, *, processor: Any = None) -> AudioEncoderWindowConfig:
        """Resolve once at realtime startup, or lazily on the executor's frontend."""
        if self._cached_window_config is None:
            self._cached_window_config = self.resolve_audio_window_config(
                processor=processor
            )
        return self._cached_window_config

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
        config = self.audio_window_config(processor=processor)
        placeholder_token_id = mm_tokens.get_token_id_by_modality(Modality.AUDIO)
        if placeholder_token_id is None:
            raise ValueError("encoder windowing requires an audio placeholder token id")
        input_ids = self._tokenize_audio_window_prompt(
            tokenizer, base_output.input_text
        )
        items, input_ids = build_audio_window_items(
            self,
            samples=audios[0],
            input_ids=input_ids,
            placeholder_token_id=placeholder_token_id,
            config=config,
            leading_context_samples=int(window_request.get(LEADING_CONTEXT_KEY, 0)),
            processor=processor,
        )
        return self._finalize_mm_items(items, images=None), input_ids, {}

    def _tokenize_audio_window_prompt(self, tokenizer, input_text: str) -> torch.Tensor:
        # Preserve the base special-token policy, avoiding a duplicate template BOS.
        add_special_tokens = True
        if self._tokenizer_auto_adds_specials:
            bos = getattr(tokenizer, "bos_token", None)
            if bos and input_text.startswith(bos):
                add_special_tokens = False
        return tokenizer(
            input_text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids.flatten()


# Default preparation for extractors using centered frames and a fixed hop size.


class AudioEncoderWindowSpec(msgspec.Struct, frozen=True):
    """An independently encodable feature span and its encoder block alignment.

    Used by the default config resolver; other frontends can resolve a sample-based
    ``AudioEncoderWindowConfig`` directly. Declaring frames alone does not prove
    encoder independence.
    """

    window_frames: int
    alignment_frames: int


def resolve_default_audio_window_config(
    spec: AudioEncoderWindowSpec,
    *,
    feature_extractor: Any,
    output_lengths: Callable[[list[int]], Sequence[int] | torch.Tensor],
    audio_config: Optional[dict] = None,
) -> AudioEncoderWindowConfig:
    """Resolve sample ranges and probe standalone and leading-context feature shapes.

    Silent probes validate feature widths and predicted output lengths without
    running the encoder. Models provide their own frame-to-token length mapping.
    """
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

    # Centered feature frames need preceding half-frame samples, rounded up to
    # whole hops.
    leading_context_samples = -(-(n_fft // 2) // hop_length) * hop_length
    config = AudioEncoderWindowConfig(
        sample_rate=int(feature_extractor.sampling_rate),
        window_samples=spec.window_frames * hop_length,
        window_tokens=int(output_lengths([spec.window_frames])[0]),
        min_tail_samples=n_fft,
        leading_context_samples=leading_context_samples,
    )
    for leading in (0, leading_context_samples):
        feature, mask = _extract_audio_window_features(
            np.zeros(config.window_samples + leading, dtype=np.float32),
            feature_extractor=feature_extractor,
            audio_config=audio_config,
            leading_context_samples=leading,
        )
        _validate_audio_window_features(feature, mask, spec.window_frames)
        tokens = int(output_lengths([int(mask.sum())])[0])
        if tokens != config.window_tokens:
            raise ValueError(
                f"feature extractor produced {tokens} encoder tokens for one window; "
                f"the output-length function predicts {config.window_tokens}"
            )
    return config


def prepare_default_audio_window(
    samples: np.ndarray,
    *,
    feature_extractor: Any,
    output_lengths: Callable[[list[int]], Sequence[int] | torch.Tensor],
    window_frames: int,
    mask_key: str,
    audio_config: Optional[dict] = None,
    leading_context_samples: int = 0,
) -> PreparedAudioWindow:
    """Prepare one window; the caller selects the mask field and length mapping."""
    feature, mask = _extract_audio_window_features(
        samples,
        feature_extractor=feature_extractor,
        audio_config=audio_config,
        leading_context_samples=leading_context_samples,
    )
    if len(samples) - leading_context_samples == window_frames * int(
        feature_extractor.hop_length
    ):
        _validate_audio_window_features(feature, mask, window_frames)
    item = MultimodalDataItem(
        modality=Modality.AUDIO,
        feature=feature,
        model_specific_data={mask_key: mask},
    )
    # Identical padded features with different valid lengths must not share KV or embeddings.
    item.set_hash(hash_feature([feature, mask]))
    return PreparedAudioWindow(item, int(output_lengths([int(mask.sum())])[0]))


def _extract_audio_window_features(
    samples: np.ndarray,
    *,
    feature_extractor: Any,
    audio_config: Optional[dict],
    leading_context_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    required = {
        "return_attention_mask": True,
        "return_tensors": "pt",
        "truncation": False,
        "padding": "longest",
        "sampling_rate": int(feature_extractor.sampling_rate),
    }
    kwargs = dict(audio_config or {})
    conflicts = sorted(
        key for key, value in required.items() if key in kwargs and kwargs[key] != value
    )
    if conflicts:
        raise ValueError(
            "encoder windowing fixes the audio preprocessing values for "
            + ", ".join(conflicts)
            + "; remove them from --mm-process-config"
        )
    kwargs.update(required)
    # Per-window calls keep cache identity independent of the request's batch size.
    output = feature_extractor([np.asarray(samples, dtype=np.float32)], **kwargs)
    feature = torch.as_tensor(output["input_features"])
    mask = torch.as_tensor(output["attention_mask"])
    if leading_context_samples:
        context_frames = leading_context_samples // int(feature_extractor.hop_length)
        feature = feature[..., context_frames:].contiguous()
        mask = mask[..., context_frames:].contiguous()
    return feature, mask


def _validate_audio_window_features(
    feature: torch.Tensor, mask: torch.Tensor, window_frames: int
) -> None:
    frames, width = int(mask.sum()), int(feature.shape[-1])
    if frames != window_frames or width != window_frames:
        raise ValueError(
            f"feature extractor produced {frames} valid frames in a {width}-frame tensor "
            f"for one window; expected {window_frames} unpadded frames"
        )
