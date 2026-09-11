"""Prepare audio-window features and items for encoder-output reuse.

Each windowed request extracts features and hashes them. With aligned starts
and the same leading context, unchanged complete windows retain their hashes
so the existing multimodal cache can reuse encoder outputs on a hit. This
module prepares features and token offsets; it does not run the encoder or
manage the embedding cache.

A model opts in by mixing ``WindowedAudioProcessorMixin`` into its multimodal
processor and declaring an ``AudioEncoderWindowSpec``; entrypoints discover it
with ``isinstance(processor, WindowedAudioProcessorMixin)``. Requests activate
windowing through the payload built by ``build_audio_window_processor_kwargs``.
The window config is always the server's own.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional, Sequence

import msgspec
import numpy as np
import torch

from sglang.srt.managers.mm_utils import hash_feature
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem


class AudioEncoderWindowSpec(msgspec.Struct, frozen=True):
    """Window config a model declares, in feature frames.

    An aligned span of ``window_frames`` feature frames must be encodable
    independently of other windows. Feature extraction may use fixed
    preceding audio context; the independence contract applies to the encoder
    inputs, not to the raw audio alone.

    ``alignment_frames`` is the encoder's convolution/position block size,
    so a window must contain a whole number of blocks. A model whose encoder
    attends across the whole clip must not declare a spec.
    """

    window_frames: int
    alignment_frames: int


class AudioEncoderWindowConfig(msgspec.Struct, frozen=True):
    """Config resolved once from a spec and a feature extractor.

    Requests use the server's processor config. Rolling callers must keep
    window starts aligned and supply the same preceding audio context for
    unchanged complete windows to retain their hashes.
    """

    sample_rate: int
    hop_length: int
    window_frames: int
    window_samples: int
    window_tokens: int
    # Minimum tail body length extracted separately. Shorter remainders are
    # merged into the preceding complete window.
    min_tail_samples: int
    # Preceding audio (whole hops covering half an STFT frame) extracted with
    # each window and cropped from its features, so a window's leading frames
    # see real samples instead of the extractor's reflect padding.
    context_samples: int

    @property
    def window_seconds(self) -> float:
        return self.window_samples / self.sample_rate


class AudioWindowFeatures(NamedTuple):
    """Per-window encoder inputs and predicted output lengths.

    Leading context frames have been removed. Windows may have different
    frame counts; these tensors are features, not cached encoder outputs.
    """

    # Each (1, n_mels, frames_i).
    features: list[torch.Tensor]
    # Each (1, frames_i); 1 marks a valid frame.
    masks: list[torch.Tensor]
    # Predicted encoder output token count for each window.
    token_counts: list[int]


def resolve_audio_window_config(
    window_processor: WindowedAudioProcessorMixin, *, processor: Any = None
) -> AudioEncoderWindowConfig:
    """Validate a processor's window declaration against its feature extractor.

    Realtime startup resolves this eagerly when windowing is enabled; other
    callers may resolve it lazily. The mixin caches the result. Silent probes
    with and without leading STFT context check feature shapes and predicted
    output lengths without running the encoder. An executor supplies its
    worker-local ``processor`` for lazy resolution.
    """
    spec = window_processor.audio_encoder_window_spec()
    if spec.window_frames <= 0 or spec.alignment_frames <= 0:
        raise ValueError("encoder window frame counts must be positive")
    if spec.window_frames % spec.alignment_frames:
        raise ValueError(
            f"encoder window of {spec.window_frames} frames is not a multiple of "
            f"the {spec.alignment_frames}-frame encoder block"
        )

    extractor = window_processor.audio_window_feature_extractor(processor=processor)
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
        window_processor.encoder_output_token_counts(
            [spec.window_frames], processor=processor
        )[0]
    )
    if window_tokens <= 0:
        raise ValueError("an encoder window produces no encoder tokens")

    # Padding may preserve a standalone window's width yet add frames when
    # preceding STFT context changes the input length. Probe both runtime forms.
    for leading_samples in (0, context_samples):
        probe = window_processor.extract_audio_window_features(
            [np.zeros(window_samples + leading_samples, dtype=np.float32)],
            leading_context_frames=[leading_samples // hop_length],
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

    return AudioEncoderWindowConfig(
        sample_rate=sample_rate,
        hop_length=hop_length,
        window_frames=spec.window_frames,
        window_samples=window_samples,
        window_tokens=window_tokens,
        min_tail_samples=n_fft,
        context_samples=context_samples,
    )


def _validate_complete_window_config(
    features: AudioWindowFeatures, config: AudioEncoderWindowConfig
) -> None:
    widths = [int(feature.shape[-1]) for feature in features.features]
    if widths != [config.window_frames] * len(widths) or list(
        features.token_counts
    ) != [config.window_tokens] * len(widths):
        raise ValueError(
            "encoder window config mismatch: complete windows produced "
            f"{widths} frames and {list(features.token_counts)} tokens; expected "
            f"{config.window_frames} frames and {config.window_tokens} tokens each"
        )


def build_audio_window_items(
    window_processor: WindowedAudioProcessorMixin,
    *,
    samples: np.ndarray,
    input_ids: torch.Tensor,
    placeholder_token_id: int,
    config: AudioEncoderWindowConfig,
    leading_context_samples: int = 0,
    processor: Any = None,
) -> tuple[list[MultimodalDataItem], torch.Tensor]:
    """Build complete-window items followed by an optional trailing item.

    ``samples`` is mono audio at ``config.sample_rate``; its first
    ``leading_context_samples`` (0 or ``config.context_samples``) are the
    audio that preceded the request's first window and are only used as
    extraction context. ``input_ids`` must contain exactly one unexpanded
    placeholder token; it is expanded to one token per encoder output position.
    Complete windows come first, in time order, and the tail (if any) is
    last, with contiguous, inclusive token offsets over the placeholder run.
    A sub-minimum tail is merged into the preceding window, so the trailing
    item can be longer than ``config.window_samples``. Every window is
    extracted from its own audio plus the preceding context, never from what
    follows it, so a complete window's features and hash do not depend on
    the audio that arrived after it.
    ``processor`` selects the worker-local frontend without changing the
    window processor's cached config or its shared processor.
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
        # Merge a short remainder into the preceding window. The combined
        # trailing item gets its own feature/mask hash and follows normal
        # cache lookup; it cannot reuse the preceding window as a separate item.
        complete_count -= 1

    first_window_start = leading_context_samples
    window_ends = [
        first_window_start + (index + 1) * window_samples
        for index in range(complete_count)
    ]
    has_tail = not window_ends or window_ends[-1] < samples.size
    if has_tail:
        window_ends.append(samples.size)

    window_inputs, window_context_frames = [], []
    start = first_window_start
    for end in window_ends:
        context_samples = min(config.context_samples, start)
        window_inputs.append(samples[start - context_samples : end])
        window_context_frames.append(context_samples // config.hop_length)
        start = end
    extracted = window_processor.extract_audio_window_features(
        window_inputs,
        leading_context_frames=window_context_frames,
        processor=processor,
    )
    _validate_complete_window_config(
        AudioWindowFeatures(
            extracted.features[:complete_count],
            extracted.masks[:complete_count],
            extracted.token_counts[:complete_count],
        ),
        config,
    )
    if has_tail and int(extracted.token_counts[-1]) <= 0:
        raise ValueError("audio tail produced no encoder tokens")
    features, masks, counts = extracted
    token_counts = [int(count) for count in counts]

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
        items.append(
            window_processor.make_audio_window_item(feature, mask, [(start, end)])
        )
        start = end + 1
    return items, expanded_input_ids


# Key inside GenerateReqInput.mm_processor_kwargs that opts a request into
# windowing. Its value carries only the request's leading context; the
# config is always the server's.
ENCODER_WINDOW_KWARG = "encoder_window"
LEADING_CONTEXT_KEY = "leading_context_samples"
FEATURE_ATTENTION_MASK_KEY = "feature_attention_mask"


def build_audio_window_processor_kwargs(*, leading_context_samples: int = 0) -> dict:
    """Build the mm_processor_kwargs payload for one windowed request."""
    return {ENCODER_WINDOW_KWARG: {LEADING_CONTEXT_KEY: int(leading_context_samples)}}


# Extract windows without truncation and with per-window padding and a valid-
# frame mask. Validation probes reject padding that changes complete-window sizes.
_REQUIRED_EXTRACT_KWARGS = {
    "return_attention_mask": True,
    "return_tensors": "pt",
    "truncation": False,
    "padding": "longest",
}


class WindowedAudioProcessorMixin:
    """Add encoder-aligned audio preprocessing to a multimodal processor.

    Split audio into windows, extract their features, and build multimodal
    items whose hashes support encoder-output reuse.

    List it before ``BaseMultimodalProcessor`` in the class bases. The mixin
    caches only its resolved config: it intercepts
    ``process_and_combine_mm_data`` only when the request carries
    ``mm_processor_kwargs["encoder_window"]``, and every other request falls
    through to the base implementation unchanged.
    """

    _cached_window_config: Optional[AudioEncoderWindowConfig] = None

    def audio_encoder_window_spec(self) -> AudioEncoderWindowSpec:
        """Declare the model's independently encodable window size and alignment."""
        raise NotImplementedError

    def audio_window_feature_extractor(self, *, processor: Any = None) -> Any:
        processor = self._processor if processor is None else processor
        return processor.feature_extractor

    def audio_window_extraction_kwargs(self, *, processor: Any = None) -> dict:
        """Merge the server's audio preprocessing config with the fixed values.

        A conflicting ``--mm-process-config`` fails here. Realtime startup
        calls this during policy resolution when windowing is enabled.
        """
        extractor = self.audio_window_feature_extractor(processor=processor)
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

    def extract_audio_window_features(
        self,
        windows: Sequence[np.ndarray],
        *,
        leading_context_frames: Optional[Sequence[int]] = None,
        processor: Any = None,
    ) -> AudioWindowFeatures:
        """Extract each window and crop its leading context frames.

        ``processor`` selects the executor's worker-local HF processor.
        """
        extractor = self.audio_window_feature_extractor(processor=processor)
        kwargs = self.audio_window_extraction_kwargs(processor=processor)
        if leading_context_frames is None:
            leading_context_frames = [0] * len(windows)
        features: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        token_counts: list[int] = []
        # One extractor call per window. Batched STFT/mel kernels are not
        # bitwise reproducible across batch sizes, and a complete window's
        # cache identity must not depend on how many windows share a request.
        for window, context_frames in zip(windows, leading_context_frames):
            output = extractor([np.asarray(window, dtype=np.float32)], **kwargs)
            feature = torch.as_tensor(output["input_features"])
            mask = torch.as_tensor(output["attention_mask"])
            if context_frames:
                # The context frames only served the STFT; drop them so the
                # item covers exactly its own window.
                feature = feature[..., context_frames:].contiguous()
                mask = mask[..., context_frames:].contiguous()
            features.append(feature)
            masks.append(mask)
            token_counts.append(
                self.encoder_output_token_counts(
                    [int(mask.sum())], processor=processor
                )[0]
            )
        return AudioWindowFeatures(features, masks, token_counts)

    def encoder_output_token_counts(
        self, frame_counts: Sequence[int], *, processor: Any = None
    ) -> list[int]:
        # Probes the HF processor's private helper; a model whose processor
        # lacks it overrides this method instead.
        processor = self._processor if processor is None else processor
        output_lengths = getattr(processor, "_get_feat_extract_output_lengths", None)
        if output_lengths is None:
            raise NotImplementedError(
                f"{type(self).__name__} must implement encoder_output_token_counts"
            )
        lengths = torch.as_tensor(output_lengths(list(frame_counts))).flatten()
        return [int(length) for length in lengths.tolist()]

    def make_audio_window_item(
        self,
        feature: torch.Tensor,
        mask: torch.Tensor,
        offsets: Sequence[tuple[int, int]],
    ) -> MultimodalDataItem:
        """Build a hashed item with inclusive token offsets in the expanded prompt."""
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

    def audio_placeholder_token_id(self, mm_tokens: Any) -> int:
        token_id = mm_tokens.get_token_id_by_modality(Modality.AUDIO)
        if token_id is None:
            raise ValueError("encoder windowing requires an audio placeholder token id")
        return token_id

    # Integration with BaseMultimodalProcessor.

    def audio_window_config(self, *, processor: Any = None) -> AudioEncoderWindowConfig:
        """The server's window config, resolved once per processor.

        Requests opt into windowing but never choose the config.
        """
        if self._cached_window_config is None:
            self._cached_window_config = resolve_audio_window_config(
                self, processor=processor
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
        processor = kwargs.get("processor")
        config = self.audio_window_config(processor=processor)
        leading_context_samples = int(window_request.get(LEADING_CONTEXT_KEY, 0))
        audios = base_output.audios or []
        if len(audios) != 1 or len(base_output.organize_results()) != 1:
            raise ValueError(
                "encoder windowing requires exactly one audio input and no other media"
            )
        if not isinstance(audios[0], np.ndarray):
            raise ValueError("encoder windowing requires raw audio samples")

        processor, tokenizer = self._resolve_processor(processor)
        input_ids = self._tokenize_audio_window_prompt(
            tokenizer, base_output.input_text
        )
        items, input_ids = build_audio_window_items(
            self,
            samples=audios[0],
            input_ids=input_ids,
            placeholder_token_id=self.audio_placeholder_token_id(mm_tokens),
            config=config,
            leading_context_samples=leading_context_samples,
            processor=processor,
        )
        return self._finalize_mm_items(items, images=None), input_ids, {}

    def _tokenize_audio_window_prompt(self, tokenizer, input_text: str) -> torch.Tensor:
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
