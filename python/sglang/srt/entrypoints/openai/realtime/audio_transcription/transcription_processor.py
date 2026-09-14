"""Coordinate realtime transcription modes, text publication, and audio progress.

Each call fixes a mode and request, runs that mode, publishes the remaining
text, then commits its outcome. Per-item progress stays in
RealtimeTranscriptionState.
EncoderWindowMode is selected for eligible requests; accepting its first
continuation candidate installs WindowedState independently of audio coverage.
Configuration is resolved once at server startup.
"""

from __future__ import annotations

import math
from typing import Any, Awaitable, Callable, Dict, Optional

from sglang.srt.entrypoints.openai.realtime.audio_transcription.audio_buffer import (
    PCM_SAMPLE_WIDTH_BYTES,
    AudioBuffer,
)
from sglang.srt.entrypoints.openai.realtime.audio_transcription.cumulative_transcription import (
    CumulativeMode,
)
from sglang.srt.entrypoints.openai.realtime.audio_transcription.transcription_mode import (
    TranscriptionMode,
    TranscriptionOutcome,
    TranscriptionStep,
)
from sglang.srt.entrypoints.openai.realtime.audio_transcription.transcription_state import (
    CumulativeState,
    RealtimeTranscriptionState,
)
from sglang.srt.entrypoints.openai.realtime.audio_transcription.windowed_transcription import (
    EncoderWindowMode,
    ResolvedEncoderWindowPolicy,
)
from sglang.srt.entrypoints.openai.streaming_asr import (
    StreamingASRState,
)
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_serving

TranscriptDeltaCallback = Callable[[str], Awaitable[None]]


class RealtimeTranscriptionProcessor:
    """Run a fixed transcription flow with cumulative or encoder-window logic."""

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        adapter: TranscriptionAdapter,
        *,
        encoder_window: Optional[ResolvedEncoderWindowPolicy] = None,
        session_id: str = "",
    ) -> None:
        # Snapshot defaults so every audio item starts with the same parameters.
        self._chunked_streaming_config = dict(adapter.chunked_streaming_config)
        pcm_bytes_per_second = adapter.model_sample_rate * PCM_SAMPLE_WIDTH_BYTES

        chunk_size_sec = self._chunked_streaming_config["chunk_size_sec"]
        if not math.isfinite(chunk_size_sec) or chunk_size_sec <= 0:
            raise ValueError("realtime ASR chunk_size_sec must be finite and positive")
        self.chunk_size_bytes = int(chunk_size_sec * pcm_bytes_per_second)
        if self.chunk_size_bytes <= 0:
            raise ValueError(
                "realtime ASR chunk_size_sec is shorter than one PCM sample"
            )
        if self.chunk_size_bytes % PCM_SAMPLE_WIDTH_BYTES:
            raise ValueError(
                "realtime ASR chunk_size_sec must resolve to whole PCM samples"
            )
        self.max_buffer_bytes = (
            get_serving().asr_max_buffer_seconds * pcm_bytes_per_second
        )
        self.decoder_streaming = bool(get_serving().enable_asr_decoder_streaming)

        activation_threshold_bytes: Optional[int] = None
        routed_dp_rank: Optional[int] = None
        if encoder_window is not None:
            if encoder_window.window_bytes <= 0:
                raise ValueError("invalid audio encoder-window config")
            activation_threshold_bytes = encoder_window.activation_threshold_bytes(
                self.chunk_size_bytes, chunk_size_sec
            )
            if session_id:
                routed_dp_rank = encoder_window.pinned_dp_rank(session_id)

        self._cumulative_mode = CumulativeMode(
            tokenizer_manager, adapter, routed_dp_rank=routed_dp_rank
        )
        self._window_mode: Optional[EncoderWindowMode] = None
        if encoder_window is not None:
            self._window_mode = EncoderWindowMode(
                tokenizer_manager,
                adapter,
                encoder_window=encoder_window,
                chunk_size_bytes=self.chunk_size_bytes,
                activation_threshold_bytes=activation_threshold_bytes,
                routed_dp_rank=routed_dp_rank,
            )

    def create_state(self) -> RealtimeTranscriptionState:
        return RealtimeTranscriptionState(
            audio=AudioBuffer(),
            mode_state=CumulativeState(
                transcript=StreamingASRState(**self._chunked_streaming_config)
            ),
        )

    def is_chunk_ready(self, state: RealtimeTranscriptionState) -> bool:
        """True once a full chunk of audio has arrived past the last attempt."""
        return (
            state.audio.received_bytes - state.audio.last_attempted_offset_bytes
            >= self.chunk_size_bytes
        )

    async def process(
        self,
        state: RealtimeTranscriptionState,
        *,
        is_last: bool,
        sampling_params: Dict[str, Any],
        on_transcript_delta: TranscriptDeltaCallback,
    ) -> None:
        """Run one mode, publish its text, then apply the accepted outcome."""
        audio = state.audio
        # Window handoff needs the inference chunk cadence. Ordinary cumulative
        # requests and final requests cover all received audio, as on main.
        end_offset_bytes = (
            audio.received_bytes
            if is_last or self._window_mode is None
            else min(
                audio.received_bytes,
                audio.last_attempted_offset_bytes + self.chunk_size_bytes,
            )
        )
        mode = self._select_mode(
            state, end_offset_bytes=end_offset_bytes, is_last=is_last
        )
        step = mode.build_step(
            state, end_offset_bytes=end_offset_bytes, is_last=is_last
        )

        async def publish_candidate(candidate: str) -> None:
            await self._publish_candidate(
                state,
                candidate,
                emitted_text_before=step.emitted_text,
                on_transcript_delta=on_transcript_delta,
            )

        outcome = await mode.execute_step(
            state,
            step,
            sampling_params=sampling_params,
            on_candidate=(
                publish_candidate if self.decoder_streaming and not is_last else None
            ),
        )
        await self._publish_candidate(
            state,
            outcome.delta,
            emitted_text_before=step.emitted_text,
            on_transcript_delta=on_transcript_delta,
            conflict_error="completed ASR decode revised already streamed text",
        )
        self._commit_outcome(state, step, outcome)

    async def flush_pending_transcript(
        self,
        state: RealtimeTranscriptionState,
        on_transcript_delta: TranscriptDeltaCallback,
    ) -> None:
        """Publish held text and accept its state without advancing audio."""
        emitted_text_before = state.emitted_text
        mode = self._active_mode(state)
        outcome = mode.flush_pending(state)
        await self._publish_candidate(
            state,
            outcome.delta,
            emitted_text_before=emitted_text_before,
            on_transcript_delta=on_transcript_delta,
            conflict_error="ASR flush revised already published text",
        )
        self._commit_outcome(state, None, outcome)

    async def _publish_candidate(
        self,
        state: RealtimeTranscriptionState,
        candidate: str,
        *,
        emitted_text_before: str,
        on_transcript_delta: TranscriptDeltaCallback,
        conflict_error: Optional[str] = None,
    ) -> None:
        """Send the candidate's unsent suffix and record successful publication.

        Intermediate revisions are skipped. Final candidates supply
        conflict_error so revisions fail before state or audio is committed.
        """
        full = emitted_text_before + candidate
        if not full.startswith(state.emitted_text):
            if conflict_error is not None:
                raise RuntimeError(conflict_error)
            return
        delta = full[len(state.emitted_text) :]
        if delta:
            await on_transcript_delta(delta)
            state.emitted_text += delta

    def _active_mode(self, state: RealtimeTranscriptionState) -> TranscriptionMode:
        if state.encoder_window_active:
            assert self._window_mode is not None
            return self._window_mode
        return self._cumulative_mode

    def _select_mode(
        self,
        state: RealtimeTranscriptionState,
        *,
        end_offset_bytes: int,
        is_last: bool,
    ) -> TranscriptionMode:
        """Select a tentative window handoff or execute the accepted mode."""
        if self._window_mode is not None and self._window_mode.can_activate(
            state, end_offset_bytes=end_offset_bytes, is_last=is_last
        ):
            return self._window_mode
        return self._active_mode(state)

    def _commit_outcome(
        self,
        state: RealtimeTranscriptionState,
        step: Optional[TranscriptionStep],
        outcome: TranscriptionOutcome,
    ) -> None:
        """Apply mode and audio progress after publication succeeds.

        A flush has no step and only replaces candidate state. Modes compute
        their next state and safe PCM release bound; publication stays intact.
        """
        audio = state.audio
        if step is not None:
            audio.last_attempted_offset_bytes = step.end_offset_bytes
        state.mode_state = outcome.next_mode_state
        if step is not None and outcome.audio_covered:
            audio.last_processed_offset_bytes = step.end_offset_bytes
        if outcome.discard_before_bytes is not None:
            audio.discard_before(outcome.discard_before_bytes)
