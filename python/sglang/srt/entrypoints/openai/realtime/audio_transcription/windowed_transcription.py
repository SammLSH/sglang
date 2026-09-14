"""Encoder-window input, continuation decoding, and startup configuration.

Resolve model defaults and server overrides once at startup, then share the
immutable result across realtime connections.
"""

from __future__ import annotations

import logging
import math
import zlib
from copy import copy
from typing import Any, Dict, Optional

import msgspec
from msgspec.structs import replace

from sglang.srt.entrypoints.openai.realtime.audio_transcription.audio_buffer import (
    PCM_SAMPLE_WIDTH_BYTES,
    snapshot_samples,
)
from sglang.srt.entrypoints.openai.realtime.audio_transcription.transcription_mode import (
    TranscriptCandidateCallback,
    TranscriptionMode,
    TranscriptionOutcome,
    TranscriptionStep,
)
from sglang.srt.entrypoints.openai.realtime.audio_transcription.transcription_state import (
    CumulativeState,
    RealtimeTranscriptionState,
    WindowedState,
)
from sglang.srt.entrypoints.openai.realtime.audio_transcription.transcription_suffix import (
    TranscriptionSuffixState,
)
from sglang.srt.entrypoints.openai.streaming_asr import (
    TranscriptionBackendAborted,
    generate_transcript,
    split_units,
)
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    RealtimeEncoderWindowPolicy,
    TranscriptionAdapter,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.multimodal.encoder_window import (
    EncoderWindowCapability,
    EncoderWindowConfig,
    encoder_window_kwargs,
)

logger = logging.getLogger(__name__)

# Failed window attempts allowed before fallback or termination.
_MAX_CONSECUTIVE_WINDOW_FAILURES = 2


class ResolvedEncoderWindowPolicy(msgspec.Struct, frozen=True):
    """Adapter policy paired with the model's resolved window config."""

    config: EncoderWindowConfig
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
        return self.config.leading_context_samples * PCM_SAMPLE_WIDTH_BYTES

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


class EncoderWindowMode(TranscriptionMode):
    """Prepare window handoffs and rolling continuations as tentative outcomes."""

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        adapter: TranscriptionAdapter,
        *,
        encoder_window: ResolvedEncoderWindowPolicy,
        chunk_size_bytes: int,
        activation_threshold_bytes: int,
        routed_dp_rank: Optional[int] = None,
    ) -> None:
        super().__init__(tokenizer_manager, adapter, routed_dp_rank=routed_dp_rank)
        self.encoder_window = encoder_window
        self.activation_threshold_bytes = activation_threshold_bytes
        window_bytes = encoder_window.window_bytes
        # Allow one extra window to resolve a stall, including a first
        # handoff whose activation threshold exceeds the rolling context.
        self._max_retained_bytes = (
            max(
                (encoder_window.policy.max_audio_context_windows + 1) * window_bytes,
                activation_threshold_bytes + chunk_size_bytes,
            )
            + window_bytes
        )

    def can_activate(
        self,
        state: RealtimeTranscriptionState,
        *,
        end_offset_bytes: int,
        is_last: bool,
    ) -> bool:
        current = state.mode_state
        return (
            isinstance(current, CumulativeState)
            and not current.window_disabled
            and not is_last
            and end_offset_bytes > self.activation_threshold_bytes
        )

    def build_step(
        self,
        state: RealtimeTranscriptionState,
        *,
        end_offset_bytes: int,
        is_last: bool,
    ) -> TranscriptionStep:
        """Snapshot a first handoff or the next rolling continuation request."""
        current = state.mode_state
        if isinstance(current, CumulativeState):
            snapshot = replace(current, transcript=copy(current.transcript))
            suffix_state = TranscriptionSuffixState()
            handoff_pending = snapshot.transcript.unpublished_text(
                emitted_text=state.emitted_text, split_cjk=True
            )
            # Keep cumulative audio until a window candidate is accepted.
            start_offset_bytes = 0
        else:
            snapshot = replace(current, suffix=copy(current.suffix))
            suffix_state = snapshot.suffix
            handoff_pending = ""
            start_offset_bytes = self._encoder_window_start_offset(
                state, end_offset_bytes
            )
        decoder_prefix = suffix_state.bounded_prefix(
            self.tokenizer_manager.tokenizer,
            self.encoder_window.policy.decoder_prefix_max_tokens,
            emitted_text=state.emitted_text,
        )
        if (
            state.emitted_text or suffix_state.confirmed_pending_chars
        ) and not decoder_prefix:
            raise RuntimeError(
                "realtime ASR decoder prefix budget cannot retain a complete text unit"
            )
        return TranscriptionStep(
            is_last=is_last,
            start_offset_bytes=start_offset_bytes,
            end_offset_bytes=end_offset_bytes,
            last_attempted_offset_bytes=state.audio.last_attempted_offset_bytes,
            last_processed_offset_bytes=state.audio.last_processed_offset_bytes,
            emitted_text=state.emitted_text,
            mode_state=snapshot,
            leading_context_bytes=min(
                self.encoder_window.context_bytes,
                start_offset_bytes - state.audio.base_offset_bytes,
            ),
            decoder_prefix=decoder_prefix,
            handoff_pending=handoff_pending,
        )

    def _encoder_window_start_offset(
        self, state: RealtimeTranscriptionState, end_offset_bytes: int
    ) -> int:
        """Keep starts window-aligned without advancing past unconfirmed audio."""
        current = state.mode_state
        assert isinstance(current, WindowedState)
        audio = state.audio
        window_bytes = self.encoder_window.window_bytes
        complete_end = end_offset_bytes // window_bytes * window_bytes
        context_windows = self.encoder_window.policy.max_audio_context_windows
        start = min(
            complete_end - context_windows * window_bytes,
            audio.last_processed_offset_bytes // window_bytes * window_bytes,
        )
        # Compaction keeps the previous start's extraction context resident.
        if audio.base_offset_bytes == 0:
            floor = 0
        else:
            resident = audio.base_offset_bytes + self.encoder_window.context_bytes
            floor = -(-resident // window_bytes) * window_bytes
        # A small amount of text after a stall cannot release the retained
        # interval at once. Advance by at most one native window.
        start = max(floor, min(start, floor + window_bytes))
        if current.unconfirmed_start is not None:
            start = min(start, current.unconfirmed_start)
        if end_offset_bytes - start > self._max_retained_bytes:
            raise RuntimeError(
                "realtime ASR transcript did not advance within the retained audio limit"
            )
        return start

    async def execute_step(
        self,
        state: RealtimeTranscriptionState,
        step: TranscriptionStep,
        *,
        sampling_params: Dict[str, Any],
        on_candidate: Optional[TranscriptCandidateCallback],
    ) -> TranscriptionOutcome:
        """Decode and reconcile against the request's private candidate snapshot."""
        before = step.mode_state
        suffix_before = (
            before.suffix
            if isinstance(before, WindowedState)
            else TranscriptionSuffixState(pending=step.handoff_pending)
        )
        if step.start_offset_bytes % self.encoder_window.window_bytes:
            raise RuntimeError("encoder-window request is not window aligned")
        samples = await snapshot_samples(
            state.audio,
            step.start_offset_bytes - step.leading_context_bytes,
            step.end_offset_bytes,
        )

        async def publish_snapshot(text: str) -> None:
            assert on_candidate is not None
            units = split_units(text)
            if len(units) < 2:
                return
            # Preserve a leading continuation space after confirmed holdback.
            snapshot = text[: text.rfind(units[-1])].rstrip()
            candidate = suffix_before.reconcile(
                snapshot,
                is_last=step.is_last,
                holdback_units=self.encoder_window.policy.decoder_prefix_holdback_units,
            ).delta
            await on_candidate(candidate)

        try:
            generation = await generate_transcript(
                tokenizer_manager=self.tokenizer_manager,
                adapter=self.adapter,
                audio_data=samples,
                sampling_params=sampling_params,
                decoder_prefix=step.decoder_prefix,
                mm_processor_kwargs=encoder_window_kwargs(
                    leading_context_samples=step.leading_context_bytes
                    // PCM_SAMPLE_WIDTH_BYTES,
                ),
                routed_dp_rank=self.routed_dp_rank,
                on_update=publish_snapshot if on_candidate is not None else None,
            )
        except TranscriptionBackendAborted as error:
            # A published continuation cannot be retracted for a retry.
            if (
                not error.retryable
                or step.is_last
                or state.emitted_text != step.emitted_text
            ):
                raise
            logger.warning(
                "[realtime] encoder-window ASR step aborted (%s); retaining audio",
                error,
            )
            return self._failed_outcome(step)
        if generation is None:
            if step.is_last:
                raise RuntimeError("final realtime ASR request returned no response")
            logger.warning("[realtime] encoder-window ASR step returned no response")
            return self._failed_outcome(step)
        if generation.finish_reason == "length":
            if step.is_last or state.emitted_text != step.emitted_text:
                raise RuntimeError("realtime ASR decode reached max_new_tokens")
            logger.warning(
                "[realtime] encoder-window ASR step reached max_new_tokens; "
                "retaining audio for retry"
            )
            return self._failed_outcome(step)
        return self._reconcile_encoder_window_text(step, suffix_before, generation.text)

    def _failed_outcome(self, step: TranscriptionStep) -> TranscriptionOutcome:
        """Retain the active mode and audio, or reject an unrecoverable stall."""
        before = step.mode_state
        if isinstance(before, CumulativeState):
            failures = before.handoff_failures + 1
            disabled = failures >= _MAX_CONSECUTIVE_WINDOW_FAILURES
            if disabled:
                logger.warning(
                    "[realtime] encoder-window handoff failed %d times; this item "
                    "falls back to cumulative transcription",
                    failures,
                )
            next_state = replace(
                before, handoff_failures=failures, window_disabled=disabled
            )
        else:
            failures = before.consecutive_failures + 1
            if failures >= _MAX_CONSECUTIVE_WINDOW_FAILURES:
                # Old PCM may already be gone after activation. Leave the
                # accepted state intact and fail instead of switching back.
                raise RuntimeError(
                    "encoder-window ASR failed repeatedly after activation"
                )
            next_state = replace(before, consecutive_failures=failures)
        return TranscriptionOutcome(next_mode_state=next_state, audio_covered=False)

    def _reconcile_encoder_window_text(
        self,
        step: TranscriptionStep,
        suffix_state: TranscriptionSuffixState,
        text: str,
    ) -> TranscriptionOutcome:
        """Prepare candidate acceptance and the corresponding PCM release bound."""
        update = suffix_state.reconcile(
            text,
            is_last=step.is_last,
            holdback_units=self.encoder_window.policy.decoder_prefix_holdback_units,
        )
        if (
            step.is_last
            and step.end_offset_bytes > step.last_processed_offset_bytes
            and len(suffix_state.pending) > suffix_state.confirmed_pending_chars
            and not text
        ):
            raise RuntimeError("final realtime ASR recovery returned empty text")
        before = step.mode_state
        if (
            update.empty_continuation
            and not step.is_last
            and step.end_offset_bytes > step.last_processed_offset_bytes
            and not before.deferred_empty_continuation
        ):
            # An empty first continuation remains a cumulative handoff attempt.
            next_state = replace(before, deferred_empty_continuation=True)
            if isinstance(next_state, WindowedState):
                next_state = replace(
                    next_state, unconfirmed_start=step.start_offset_bytes
                )
            return TranscriptionOutcome(next_mode_state=next_state, audio_covered=False)
        audio_covered = (
            step.is_last
            or bool(update.delta)
            # Confirmed holdback is retained in decoder context. Unconfirmed
            # candidates still require their audio to remain available.
            or len(update.pending) == update.confirmed_pending_chars
        )
        return TranscriptionOutcome(
            next_mode_state=WindowedState(
                suffix=TranscriptionSuffixState(
                    pending=update.pending,
                    confirmed_pending_chars=update.confirmed_pending_chars,
                ),
                unconfirmed_start=None if audio_covered else step.start_offset_bytes,
            ),
            audio_covered=audio_covered,
            delta=update.delta,
            discard_before_bytes=(
                max(0, step.start_offset_bytes - self.encoder_window.context_bytes)
                if audio_covered and not step.is_last
                else None
            ),
        )

    def flush_pending(self, state: RealtimeTranscriptionState) -> TranscriptionOutcome:
        current = state.mode_state
        assert isinstance(current, WindowedState)
        suffix = copy(current.suffix)
        delta = suffix.flush()
        return TranscriptionOutcome(
            next_mode_state=replace(current, suffix=suffix),
            audio_covered=False,
            delta=delta,
        )


def resolve_realtime_encoder_window_policy(
    *,
    adapter: TranscriptionAdapter,
    tokenizer_manager: TokenizerManager,
    serving_config: Any,
) -> Optional[ResolvedEncoderWindowPolicy]:
    """Build an encoder window policy from adapter defaults and server overrides.

    If the adapter or processor lacks encoder window support, warn and return
    None to keep realtime ASR in cumulative mode. A missing tokenizer or invalid
    encoder window config raises an error during startup.
    """
    mm_processor = tokenizer_manager.mm_processor
    tokenizer = tokenizer_manager.tokenizer
    max_buffer_seconds = serving_config.asr_max_buffer_seconds
    dp_size = tokenizer_manager.elastic_worker_count

    policy = adapter.realtime_encoder_window_policy
    if policy is None or not isinstance(mm_processor, EncoderWindowCapability):
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
    overrides = {
        "min_audio_sec": serving_config.asr_encoder_window_min_audio_seconds,
        "max_audio_context_windows": serving_config.asr_encoder_window_max_context_windows,
        "decoder_prefix_max_tokens": serving_config.asr_decoder_prefix_max_tokens,
        "decoder_prefix_holdback_units": serving_config.asr_decoder_prefix_holdback_units,
    }
    overrides = {name: value for name, value in overrides.items() if value is not None}
    if overrides:
        policy = replace(policy, **overrides)
    if tokenizer is None:
        raise RuntimeError(
            "encoder-window ASR requires a tokenizer for the decoder prefix"
        )

    config = mm_processor.encoder_window_config()
    if config.sample_rate != adapter.model_sample_rate:
        raise ValueError(
            f"feature extractor sample rate {config.sample_rate} differs from the "
            f"model sample rate {adapter.model_sample_rate}"
        )
    resolved = ResolvedEncoderWindowPolicy(
        config=config, policy=policy, dp_size=max(1, int(dp_size))
    )
    chunk_seconds = adapter.chunked_streaming_config["chunk_size_sec"]
    bytes_per_second = adapter.model_sample_rate * PCM_SAMPLE_WIDTH_BYTES
    chunk_bytes = int(chunk_seconds * bytes_per_second)
    # Non-final requests end on chunk boundaries and must strictly exceed
    # the aligned threshold. A cap equal to the first eligible end is enough.
    first_window_end = (
        resolved.activation_threshold_bytes(chunk_bytes, chunk_seconds) + chunk_bytes
    )
    if max_buffer_seconds * bytes_per_second < first_window_end:
        logger.warning(
            "[realtime] encoder windowing needs at least %g s of audio but "
            "--asr-max-buffer-seconds is %s; raise it to allow windowing",
            first_window_end / bytes_per_second,
            max_buffer_seconds,
        )
    logger.info(
        "[realtime] encoder windowing enabled: %.1f s windows of %d tokens, "
        "activation after %g s, max_context_windows %d, dp_size %d, "
        "prefix_tokens %d, holdback_units %d",
        config.window_seconds,
        config.window_tokens,
        policy.min_audio_sec,
        policy.max_audio_context_windows,
        dp_size,
        policy.decoder_prefix_max_tokens,
        policy.decoder_prefix_holdback_units,
    )
    return resolved
