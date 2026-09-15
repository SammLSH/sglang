from __future__ import annotations

import math
from typing import Awaitable, Callable, Dict, Optional

from pydantic import JsonValue

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
        self.chunked_streaming_config = dict(adapter.chunked_streaming_config)
        pcm_bytes_per_second = adapter.model_sample_rate * PCM_SAMPLE_WIDTH_BYTES

        chunk_size_sec = self.chunked_streaming_config["chunk_size_sec"]
        if not math.isfinite(chunk_size_sec) or chunk_size_sec <= 0:
            raise ValueError("realtime ASR chunk_size_sec must be finite and positive")
        else:
            self.chunk_size_bytes = int(chunk_size_sec * pcm_bytes_per_second)
        if self.chunk_size_bytes <= 0:
            raise ValueError(
                "realtime ASR chunk_size_sec is shorter than one PCM sample"
            )
        elif self.chunk_size_bytes % PCM_SAMPLE_WIDTH_BYTES:
            raise ValueError(
                "realtime ASR chunk_size_sec must resolve to whole PCM samples"
            )
        else:
            self.max_buffer_bytes = (
                get_serving().asr_max_buffer_seconds * pcm_bytes_per_second
            )
        self.decoder_streaming = bool(get_serving().enable_asr_decoder_streaming)

        routed_dp_rank = (
            encoder_window.pinned_dp_rank(session_id)
            if encoder_window is not None and session_id
            else None
        )
        if encoder_window is not None:
            self.window_mode = EncoderWindowMode(
                tokenizer_manager,
                adapter,
                encoder_window=encoder_window,
                chunk_size_bytes=self.chunk_size_bytes,
                activation_threshold_bytes=encoder_window.activation_threshold_bytes(
                    self.chunk_size_bytes, chunk_size_sec
                ),
                routed_dp_rank=routed_dp_rank,
            )
        else:
            self.window_mode = None
        self.cumulative_mode = CumulativeMode(
            tokenizer_manager, adapter, routed_dp_rank=routed_dp_rank
        )

    def create_state(self) -> RealtimeTranscriptionState:
        return RealtimeTranscriptionState(
            audio=AudioBuffer(),
            mode_state=CumulativeState(
                transcript=StreamingASRState(**self.chunked_streaming_config)
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
        sampling_params: Dict[str, JsonValue],
        on_transcript_delta: TranscriptDeltaCallback,
    ) -> None:
        """Run one mode, publish its text, then apply the accepted outcome."""
        audio = state.audio
        # Window handoff needs the inference chunk cadence. Ordinary cumulative
        # requests and final requests cover all received audio, as on main.
        end_offset_bytes = (
            audio.received_bytes
            if is_last or self.window_mode is None
            else min(
                audio.received_bytes,
                audio.last_attempted_offset_bytes + self.chunk_size_bytes,
            )
        )
        if self.window_mode is not None and self.window_mode.can_activate(
            state, end_offset_bytes=end_offset_bytes, is_last=is_last
        ):
            mode = self.window_mode
        else:
            mode = self.active_mode(state)
        step = mode.build_step(
            state, end_offset_bytes=end_offset_bytes, is_last=is_last
        )

        async def _publish_candidate(candidate: str) -> None:
            await self.publish_candidate(
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
                _publish_candidate if self.decoder_streaming and not is_last else None
            ),
        )
        await self.publish_candidate(
            state,
            outcome.delta,
            emitted_text_before=step.emitted_text,
            on_transcript_delta=on_transcript_delta,
            conflict_error="completed ASR decode revised already streamed text",
        )
        self.commit_outcome(state, step, outcome)

    async def flush_pending_transcript(
        self,
        state: RealtimeTranscriptionState,
        on_transcript_delta: TranscriptDeltaCallback,
    ) -> None:
        """Publish held text and accept its state without advancing audio."""
        emitted_text_before = state.emitted_text
        mode = self.active_mode(state)
        outcome = mode.flush_pending(state)
        await self.publish_candidate(
            state,
            outcome.delta,
            emitted_text_before=emitted_text_before,
            on_transcript_delta=on_transcript_delta,
            conflict_error="ASR flush revised already published text",
        )
        self.commit_outcome(state, None, outcome)

    async def publish_candidate(
        self,
        state: RealtimeTranscriptionState,
        candidate: str,
        *,
        emitted_text_before: str,
        on_transcript_delta: TranscriptDeltaCallback,
        conflict_error: Optional[str] = None,
    ) -> None:
        """Skip revised previews; reject revised finals before accepting state."""
        # Only this consumer appends text, so the step's historical prefix is fixed.
        published_in_step = state.emitted_text[len(emitted_text_before) :]
        if not candidate.startswith(published_in_step):
            if conflict_error is not None:
                raise RuntimeError(conflict_error)
            else:
                return
        else:
            delta = candidate[len(published_in_step) :]
        if delta:
            await on_transcript_delta(delta)
            state.emitted_text += delta
        else:
            return

    def active_mode(self, state: RealtimeTranscriptionState) -> TranscriptionMode:
        if state.encoder_window_active:
            assert self.window_mode is not None
            return self.window_mode
        else:
            return self.cumulative_mode

    def commit_outcome(
        self,
        state: RealtimeTranscriptionState,
        step: Optional[TranscriptionStep],
        outcome: TranscriptionOutcome,
    ) -> None:
        """Accept candidate state and audio progress after successful publication."""
        audio = state.audio
        state.mode_state = outcome.next_mode_state
        audio.last_attempted_offset_bytes = (
            step.end_offset_bytes
            if step is not None
            else audio.last_attempted_offset_bytes
        )
        audio.last_processed_offset_bytes = (
            step.end_offset_bytes
            if step is not None and outcome.audio_covered
            else audio.last_processed_offset_bytes
        )
        if outcome.discard_before_bytes is not None:
            audio.discard_before(outcome.discard_before_bytes)
        else:
            return
