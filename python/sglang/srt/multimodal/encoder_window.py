"""Encoder windowing: split one audio input into independently encodable
windows so complete windows can be reused through the multimodal embedding
cache across rolling requests.

A model opts in by mixing ``EncoderWindowMixin`` into its multimodal
processor and declaring an ``EncoderWindowSpec``; entrypoints discover it
with ``isinstance(processor, EncoderWindowCapability)``. Requests activate
it by passing ``mm_processor_kwargs=encoder_window_kwargs(...)``; the window
geometry is always the server's own.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional, Protocol, Sequence, runtime_checkable

import msgspec
import numpy as np
import torch

from sglang.srt.managers.mm_utils import hash_feature
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem


class EncoderWindowSpec(msgspec.Struct, frozen=True):
    """Window geometry a model declares, in feature frames.

    Contract: for any span of ``window_frames`` frames that starts on a
    window boundary, the encoder output depends only on that span's audio.
    ``alignment_frames`` is the block the encoder front end works in
    (convolution chunking, positional-embedding reset), so a window must be a
    whole number of blocks. A model whose encoder attends across the whole
    clip must not declare a spec.
    """

    window_frames: int
    alignment_frames: int


class EncoderWindowConfig(msgspec.Struct, frozen=True):
    """Geometry resolved once from a spec and a feature extractor.

    The server resolves it for its own processor; requests cannot supply or
    override it, so every request splits audio identically and complete
    windows keep a stable cache identity.
    """

    sample_rate: int
    hop_length: int
    n_fft: int
    window_frames: int
    window_samples: int
    window_tokens: int
    # A trailing partial window shorter than this cannot be extracted on its
    # own; it is merged into the preceding complete window instead.
    min_tail_samples: int
    # Preceding audio (whole hops covering half an STFT frame) extracted with
    # each window and cropped from its features, so a window's leading frames
    # see real samples instead of the extractor's reflect padding.
    context_samples: int

    @property
    def window_seconds(self) -> float:
        return self.window_samples / self.sample_rate


class WindowFeatures(NamedTuple):
    """Per-window extractor output. Windows may have different frame counts."""

    # Each (1, n_mels, frames_i).
    features: list[torch.Tensor]
    # Each (1, frames_i); 1 marks a valid frame.
    masks: list[torch.Tensor]
    # Encoder tokens produced by each window.
    token_counts: list[int]


@runtime_checkable
class EncoderWindowCapability(Protocol):
    """What the window builder needs from a processor.

    ``EncoderWindowMixin`` provides defaults for everything except
    ``encoder_window_spec``. A model overrides a method only when its feature
    extractor or item layout differs from the Whisper-style default.
    """

    def encoder_window_spec(self) -> EncoderWindowSpec:
        """Declare the independently encodable window geometry."""

    def encoder_window_config(self, *, processor: Any = None) -> EncoderWindowConfig:
        """Return the server's resolved geometry, cached after the first call."""

    def encoder_window_feature_extractor(self, *, processor: Any = None) -> Any:
        """Return the extractor exposing sampling_rate, hop_length and n_fft."""

    def extract_encoder_window_features(
        self,
        windows: Sequence[np.ndarray],
        *,
        leading_frames: Optional[Sequence[int]] = None,
        processor: Any = None,
    ) -> WindowFeatures:
        """Extract each window's features independently of the other windows.

        ``leading_frames[i]`` frames at the start of window ``i`` are context
        (preceding audio) and must be cropped from the returned features.
        ``processor`` is the executor's worker-local HF processor, when supplied.
        """

    def encoder_window_output_lengths(
        self, frame_lens: Sequence[int], *, processor: Any = None
    ) -> list[int]:
        """Map feature frame counts to encoder token counts."""

    def make_encoder_window_item(
        self,
        feature: torch.Tensor,
        mask: torch.Tensor,
        offsets: Sequence[tuple[int, int]],
    ) -> MultimodalDataItem:
        """Build the item for one window, including its hash and pad value."""

    def encoder_window_placeholder_token_id(self, mm_tokens: Any) -> int:
        """Return the placeholder token id the audio run expands."""


def resolve_encoder_window_config(
    cap: EncoderWindowCapability, *, processor: Any = None
) -> EncoderWindowConfig:
    """Validate a processor's window declaration against its feature extractor.

    Runs once at startup so a misconfigured server fails before its first
    windowed request. The self-check extracts silent windows with and without
    leading STFT context and requires the predicted frame and token counts.
    Lazy resolution inside an executor uses its worker-local ``processor``.
    """
    spec = cap.encoder_window_spec()
    if spec.window_frames <= 0 or spec.alignment_frames <= 0:
        raise ValueError("encoder window geometry must be positive")
    if spec.window_frames % spec.alignment_frames:
        raise ValueError(
            f"encoder window of {spec.window_frames} frames is not a multiple of "
            f"the {spec.alignment_frames}-frame encoder block"
        )

    extractor = cap.encoder_window_feature_extractor(processor=processor)
    sample_rate = int(extractor.sampling_rate)
    hop_length = int(extractor.hop_length)
    n_fft = int(extractor.n_fft)
    if hop_length <= 0 or n_fft <= 0:
        raise ValueError("feature extractor hop_length and n_fft must be positive")
    # Dither makes the same audio hash differently each time and defeats window
    # reuse silently; `dither` is probed because only some HF extractors have it.
    if float(getattr(extractor, "dither", 0.0)) != 0.0:
        raise ValueError(
            "encoder windowing requires deterministic feature extraction "
            "(dither must be 0)"
        )

    window_samples = spec.window_frames * hop_length
    # Whisper-style extractors center frame k on sample k * hop and reflect-pad
    # beyond the array, so a window's first frames differ from the same frames
    # computed inside the full signal. Prepending the preceding half frame,
    # rounded up to whole hops, restores the real samples there.
    context_samples = -(-(n_fft // 2) // hop_length) * hop_length
    window_tokens = int(
        cap.encoder_window_output_lengths([spec.window_frames], processor=processor)[0]
    )
    if window_tokens <= 0:
        raise ValueError("an encoder window produces no encoder tokens")

    # Padding may preserve a standalone window's width yet add frames when
    # preceding STFT context changes the input length. Probe both runtime forms.
    for leading_samples in (0, context_samples):
        probe = cap.extract_encoder_window_features(
            [np.zeros(window_samples + leading_samples, dtype=np.float32)],
            leading_frames=[leading_samples // hop_length],
            processor=processor,
        )
        frames = int(probe.masks[0].sum())
        width = int(probe.features[0].shape[-1])
        context = " with leading context" if leading_samples else ""
        if frames != spec.window_frames or width != spec.window_frames:
            raise ValueError(
                f"feature extractor produced {frames} valid frames in a {width}-frame "
                f"tensor for one window{context}; expected "
                f"{spec.window_frames} unpadded frames"
            )
        if probe.token_counts[0] != window_tokens:
            raise ValueError(
                f"feature extractor produced {probe.token_counts[0]} encoder tokens "
                f"for one window{context}; the output-length function predicts "
                f"{window_tokens}"
            )

    return EncoderWindowConfig(
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_fft=n_fft,
        window_frames=spec.window_frames,
        window_samples=window_samples,
        window_tokens=window_tokens,
        min_tail_samples=n_fft,
        context_samples=context_samples,
    )


def _check_complete_windows(
    features: WindowFeatures, config: EncoderWindowConfig
) -> None:
    widths = [int(feature.shape[-1]) for feature in features.features]
    if widths != [config.window_frames] * len(widths) or list(
        features.token_counts
    ) != [config.window_tokens] * len(widths):
        raise ValueError(
            "encoder window geometry changed: complete windows produced "
            f"{widths} frames and {list(features.token_counts)} tokens; expected "
            f"{config.window_frames} frames and {config.window_tokens} tokens each"
        )


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
    """Turn one audio input into complete-window items plus a mutable tail.

    ``samples`` is mono audio at ``config.sample_rate``; its first
    ``leading_context_samples`` (0 or ``config.context_samples``) are the
    audio that preceded the request's first window and are only used as
    extraction context. ``input_ids`` must contain exactly one unexpanded
    placeholder token; it is expanded to one token per encoder output.
    Complete windows come first, in time order, and the tail (if any) is
    last, with contiguous offsets over the placeholder run. Every window is
    extracted from its own audio plus the preceding context, never from what
    follows it, so a complete window's features and hash do not depend on
    the audio that arrived after it.
    ``processor`` selects the worker-local frontend without changing the
    capability's cached geometry or its shared processor.
    """
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError("encoder windowing requires mono audio samples")
    if leading_context_samples not in (0, config.context_samples):
        raise ValueError(
            f"leading context must be 0 or {config.context_samples} samples, "
            f"got {leading_context_samples}"
        )
    body_samples = samples.size - leading_context_samples
    if body_samples <= 0:
        raise ValueError("encoder windowing requires non-empty audio")

    window_samples = config.window_samples
    complete_count, tail_samples = divmod(body_samples, window_samples)
    if 0 < tail_samples < config.min_tail_samples:
        if complete_count == 0:
            raise ValueError(
                f"audio shorter than {config.min_tail_samples} samples cannot be "
                "windowed"
            )
        # The extractor needs at least n_fft samples. Hand the short remainder
        # to the preceding window; that window is then no longer a stable
        # complete window, so it is hashed and re-encoded like any tail.
        complete_count -= 1

    features: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    token_counts: list[int] = []

    def extraction_input(start: int, end: int) -> tuple[np.ndarray, int]:
        context = min(config.context_samples, start)
        return samples[start - context : end], context // config.hop_length

    first_window_start = leading_context_samples
    window_inputs, window_leads = [], []
    for index in range(complete_count):
        start = first_window_start + index * window_samples
        audio, lead = extraction_input(start, start + window_samples)
        window_inputs.append(audio)
        window_leads.append(lead)
    if window_inputs:
        complete = cap.extract_encoder_window_features(
            window_inputs, leading_frames=window_leads, processor=processor
        )
        _check_complete_windows(complete, config)
        features.extend(complete.features)
        masks.extend(complete.masks)
        token_counts.extend(int(count) for count in complete.token_counts)

    tail_start = first_window_start + complete_count * window_samples
    tail, tail_lead = extraction_input(tail_start, samples.size)
    if tail_start < samples.size:
        tail_features = cap.extract_encoder_window_features(
            [tail], leading_frames=[tail_lead], processor=processor
        )
        if int(tail_features.token_counts[0]) <= 0:
            raise ValueError("audio tail produced no encoder tokens")
        features.extend(tail_features.features)
        masks.extend(tail_features.masks)
        token_counts.append(int(tail_features.token_counts[0]))

    input_ids = torch.as_tensor(input_ids).flatten()
    positions = (input_ids == placeholder_token_id).nonzero(as_tuple=True)[0]
    if positions.numel() != 1:
        raise ValueError(
            "encoder windowing expects exactly one unexpanded audio placeholder, "
            f"found {int(positions.numel())}"
        )
    position = int(positions[0])
    expanded_input_ids = torch.cat(
        [
            input_ids[:position],
            torch.full(
                (sum(token_counts),),
                placeholder_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            ),
            input_ids[position + 1 :],
        ]
    )

    items: list[MultimodalDataItem] = []
    start = position
    for feature, mask, count in zip(features, masks, token_counts):
        end = start + count - 1
        items.append(cap.make_encoder_window_item(feature, mask, [(start, end)]))
        start = end + 1
    return items, expanded_input_ids


# Key inside GenerateReqInput.mm_processor_kwargs that opts a request into
# windowing. Its value carries only the request's leading context; the
# geometry is always the server's.
ENCODER_WINDOW_KWARG = "encoder_window"
LEADING_CONTEXT_KEY = "leading_context_samples"
FEATURE_ATTENTION_MASK_KEY = "feature_attention_mask"


def encoder_window_kwargs(*, leading_context_samples: int = 0) -> dict:
    """Build the mm_processor_kwargs payload for one windowed request."""
    return {ENCODER_WINDOW_KWARG: {LEADING_CONTEXT_KEY: int(leading_context_samples)}}


# Extraction settings windowing depends on: no truncation, no padding beyond
# the window itself, and a mask that makes each item's frame count explicit.
_REQUIRED_EXTRACT_KWARGS = {
    "return_attention_mask": True,
    "return_tensors": "pt",
    "truncation": False,
    "padding": "longest",
}


class EncoderWindowMixin:
    """Add encoder windowing to a multimodal processor.

    List it before ``BaseMultimodalProcessor`` in the class bases. The mixin
    caches only its resolved geometry: it intercepts
    ``process_and_combine_mm_data`` only when the request carries
    ``mm_processor_kwargs["encoder_window"]``, and every other request falls
    through to the base implementation unchanged.
    """

    _encoder_window_config: Optional[EncoderWindowConfig] = None

    # ---- capability defaults (see EncoderWindowCapability) ----------------

    def encoder_window_feature_extractor(self, *, processor: Any = None) -> Any:
        processor = self._processor if processor is None else processor
        return processor.feature_extractor

    def encoder_window_extract_kwargs(self, *, processor: Any = None) -> dict:
        """Merge the server's audio preprocessing config with the fixed values.

        A conflicting ``--mm-process-config`` fails here, which startup policy
        resolution calls, rather than on the first windowed request.
        """
        extractor = self.encoder_window_feature_extractor(processor=processor)
        required = dict(
            _REQUIRED_EXTRACT_KWARGS, sampling_rate=int(extractor.sampling_rate)
        )
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
        features: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        token_counts: list[int] = []
        # One extractor call per window. Batched STFT/mel kernels are not
        # bitwise reproducible across batch sizes, and a complete window's
        # cache identity must not depend on how many windows share a request.
        for window, lead in zip(windows, leading_frames):
            output = extractor([np.asarray(window, dtype=np.float32)], **kwargs)
            feature = torch.as_tensor(output["input_features"])
            mask = torch.as_tensor(output["attention_mask"])
            if lead:
                # The context frames only served the STFT; drop them so the
                # item covers exactly its own window.
                feature = feature[..., lead:].contiguous()
                mask = mask[..., lead:].contiguous()
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
        # Probes the HF processor's private helper; a model whose processor
        # lacks it overrides this method instead.
        processor = self._processor if processor is None else processor
        output_lengths = getattr(processor, "_get_feat_extract_output_lengths", None)
        if output_lengths is None:
            raise NotImplementedError(
                f"{type(self).__name__} must implement encoder_window_output_lengths"
            )
        lengths = torch.as_tensor(output_lengths(list(frame_lens))).flatten()
        return [int(length) for length in lengths.tolist()]

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
        # The mask is part of the identity: zero-padded mel frames can be
        # byte-identical for inputs whose valid frame counts differ.
        item.set_hash(hash_feature([feature, mask]))
        return item

    def encoder_window_placeholder_token_id(self, mm_tokens: Any) -> int:
        token_id = mm_tokens.get_token_id_by_modality(Modality.AUDIO)
        if token_id is None:
            raise ValueError("encoder windowing requires an audio placeholder token id")
        return token_id

    # ---- wiring -------------------------------------------------------------

    def encoder_window_config(self, *, processor: Any = None) -> EncoderWindowConfig:
        """The server's window geometry, resolved once per processor.

        Requests opt into windowing but never choose the geometry.
        """
        if self._encoder_window_config is None:
            self._encoder_window_config = resolve_encoder_window_config(
                self, processor=processor
            )
        return self._encoder_window_config

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
        return self._process_encoder_windows(
            base_output,
            mm_tokens,
            self.encoder_window_config(processor=kwargs.get("processor")),
            leading_context_samples=int(window_request.get(LEADING_CONTEXT_KEY, 0)),
            processor=kwargs.get("processor"),
        )

    def _process_encoder_windows(
        self,
        base_output,
        mm_tokens,
        config: EncoderWindowConfig,
        *,
        leading_context_samples: int = 0,
        processor=None,
    ):
        audios = base_output.audios or []
        if len(audios) != 1 or len(base_output.organize_results()) != 1:
            raise ValueError(
                "encoder windowing requires exactly one audio input and no other media"
            )
        if not isinstance(audios[0], np.ndarray):
            raise ValueError("encoder windowing requires raw audio samples")

        processor, tokenizer = self._resolve_processor(processor)
        input_ids = self._tokenize_for_encoder_windows(
            tokenizer, base_output.input_text
        )
        items, input_ids = build_encoder_window_items(
            self,
            samples=audios[0],
            input_ids=input_ids,
            placeholder_token_id=self.encoder_window_placeholder_token_id(mm_tokens),
            config=config,
            leading_context_samples=leading_context_samples,
            processor=processor,
        )
        return self._finalize_mm_items(items, images=None), input_ids, {}

    def _tokenize_for_encoder_windows(self, tokenizer, input_text: str) -> torch.Tensor:
        # Same special-token policy as the base text path: keep the tokenizer's
        # defaults unless they would duplicate a BOS the template already wrote.
        add_special_tokens = True
        if self._tokenizer_auto_adds_specials:
            bos = getattr(tokenizer, "bos_token", None)
            if bos and input_text.startswith(bos):
                add_special_tokens = False
        return tokenizer(
            input_text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids.flatten()
