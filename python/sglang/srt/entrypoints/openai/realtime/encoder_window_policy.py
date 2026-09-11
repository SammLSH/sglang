"""Startup resolution of the realtime encoder-window policy.

Resolved once per server from the adapter's declaration and the multimodal
processor's window capability, then shared by every realtime connection.
"""

from __future__ import annotations

import logging
import math
import zlib
from typing import Any, Optional

import msgspec

from sglang.srt.entrypoints.openai.realtime.audio_buffer import PCM_SAMPLE_WIDTH_BYTES
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    RealtimeEncoderWindowPolicy,
    TranscriptionAdapter,
)
from sglang.srt.multimodal.encoder_window import (
    AudioEncoderWindowConfig,
    WindowedAudioProcessorMixin,
)

logger = logging.getLogger(__name__)


class ResolvedEncoderWindowPolicy(msgspec.Struct, frozen=True):
    """Adapter policy paired with the model's resolved window config."""

    config: AudioEncoderWindowConfig
    policy: RealtimeEncoderWindowPolicy
    # Data-parallel worker count the sessions are pinned across; 1 disables
    # pinning. Pinning keeps a session's window embeddings on one rank so the
    # embedding cache and the radix prefix cache can hit.
    dp_size: int = 1

    @property
    def window_bytes(self) -> int:
        return self.config.window_samples * PCM_SAMPLE_WIDTH_BYTES

    @property
    def context_bytes(self) -> int:
        return self.config.context_samples * PCM_SAMPLE_WIDTH_BYTES

    def supports_language(self, language: Optional[str]) -> bool:
        return self.policy.supports_language(language)

    def activation_threshold_bytes(
        self, chunk_size_bytes: int, chunk_size_sec: float
    ) -> int:
        """Return the encoder window activation threshold in audio bytes.

        Round up to a whole number of inference chunks. Activation requires
        a non-final decode ending strictly after this threshold.
        """
        return math.ceil(self.policy.min_audio_sec / chunk_size_sec) * chunk_size_bytes

    def pinned_dp_rank(self, session_id: str) -> Optional[int]:
        if self.dp_size <= 1:
            return None
        return zlib.crc32(session_id.encode("utf-8")) % self.dp_size


def resolve_realtime_encoder_window_policy(
    *,
    adapter: TranscriptionAdapter,
    mm_processor: Any,
    tokenizer: Any,
    max_buffer_seconds: float,
    dp_size: int,
    min_audio_sec: Optional[float] = None,
    max_audio_context_windows: Optional[int] = None,
    decoder_prefix_max_tokens: Optional[int] = None,
    decoder_prefix_holdback_units: Optional[int] = None,
) -> Optional[ResolvedEncoderWindowPolicy]:
    """Build an encoder window policy from adapter defaults and server overrides.

    If the adapter or processor lacks encoder window support, warn and return
    None to keep realtime ASR in cumulative mode. A missing tokenizer or invalid
    encoder window config raises an error during startup.
    """
    policy = adapter.realtime_encoder_window_policy
    if policy is None or not isinstance(mm_processor, WindowedAudioProcessorMixin):
        logger.warning(
            "[realtime] --enable-asr-encoder-window is set but the model or its "
            "transcription adapter does not declare encoder windowing; realtime "
            "transcription stays cumulative"
        )
        return None
    if not isinstance(policy, RealtimeEncoderWindowPolicy):
        raise TypeError(
            "realtime_encoder_window_policy must return RealtimeEncoderWindowPolicy"
        )
    overrides = {}
    if min_audio_sec is not None:
        overrides["min_audio_sec"] = min_audio_sec
    if max_audio_context_windows is not None:
        overrides["max_audio_context_windows"] = max_audio_context_windows
    if decoder_prefix_max_tokens is not None:
        overrides["decoder_prefix_max_tokens"] = decoder_prefix_max_tokens
    if decoder_prefix_holdback_units is not None:
        overrides["decoder_prefix_holdback_units"] = decoder_prefix_holdback_units
    if overrides:
        policy = msgspec.structs.replace(policy, **overrides)
    if tokenizer is None:
        raise RuntimeError(
            "encoder-window ASR requires a tokenizer for the decoder prefix"
        )

    config = mm_processor.audio_window_config()
    if config.sample_rate != adapter.model_sample_rate:
        raise ValueError(
            f"feature extractor sample rate {config.sample_rate} differs from the "
            f"model sample rate {adapter.model_sample_rate}"
        )
    if max_buffer_seconds <= policy.min_audio_sec:
        logger.warning(
            "[realtime] encoder windowing activates after %.0f s of audio but "
            "--asr-max-buffer-seconds is %s; raise it above the threshold to "
            "process longer items",
            policy.min_audio_sec,
            max_buffer_seconds,
        )
    logger.info(
        "[realtime] encoder windowing enabled: %.1f s windows of %d tokens, "
        "activation after %g s, max_context_windows %d, languages %s, dp_size %d, "
        "prefix_tokens %d, holdback_units %d",
        config.window_seconds,
        config.window_tokens,
        policy.min_audio_sec,
        policy.max_audio_context_windows,
        ",".join(policy.supported_languages),
        dp_size,
        policy.decoder_prefix_max_tokens,
        policy.decoder_prefix_holdback_units,
    )
    return ResolvedEncoderWindowPolicy(
        config=config, policy=policy, dp_size=max(1, int(dp_size))
    )
