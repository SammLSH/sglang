from __future__ import annotations

from typing import Awaitable, Callable, Dict, Optional

from pydantic import JsonValue

from sglang.srt.entrypoints.openai.realtime.audio_transcription.audio_buffer import (
    PCM_SAMPLE_WIDTH_BYTES,
    AudioBuffer,
)
from sglang.srt.entrypoints.openai.realtime.audio_transcription.cumulative_transcription import (
    CumulativeTranscriptionMode,
)
from sglang.srt.entrypoints.openai.realtime.audio_transcription.transcription_mode import (
    TranscriptionMode,
    TranscriptionOutcome,
    TranscriptionRequest,
)
from sglang.srt.entrypoints.openai.realtime.audio_transcription.transcription_state import (
    CumulativeTranscriptionState,
    RealtimeTranscriptionState,
)
from sglang.srt.entrypoints.openai.realtime.audio_transcription.windowed_transcription import (
    ResolvedEncoderWindowPolicy,
    WindowedTranscriptionMode,
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
    """Transcribe audio, send new text, then save progress for the next request."""

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        adapter: TranscriptionAdapter,
        *,
        encoder_window: Optional[ResolvedEncoderWindowPolicy] = None,
    ) -> None:
        # Copy defaults so every audio item starts with the same parameters.
        self.chunked_streaming_config = dict(adapter.chunked_streaming_config)
        pcm_bytes_per_second = adapter.model_sample_rate * PCM_SAMPLE_WIDTH_BYTES

        chunk_size_sec = self.chunked_streaming_config["chunk_size_sec"]
        self.chunk_size_bytes = int(chunk_size_sec * pcm_bytes_per_second)
        serving_config = get_serving()
        self.max_buffer_bytes = (
            serving_config.asr_max_buffer_seconds * pcm_bytes_per_second
        )
        self.decoder_streaming = serving_config.enable_asr_decoder_streaming

        if encoder_window is not None:
            self.window_mode = WindowedTranscriptionMode(
                tokenizer_manager,
                adapter,
                encoder_window=encoder_window,
                chunk_size_bytes=self.chunk_size_bytes,
                activation_threshold_bytes=encoder_window.activation_threshold_bytes(
                    self.chunk_size_bytes, chunk_size_sec
                ),
            )
        else:
            self.window_mode = None
        self.cumulative_mode = CumulativeTranscriptionMode(tokenizer_manager, adapter)

    def create_transcription_state(self) -> RealtimeTranscriptionState:
        return RealtimeTranscriptionState(
            audio=AudioBuffer(),
            mode_state=CumulativeTranscriptionState(
                transcript=StreamingASRState(**self.chunked_streaming_config)
            ),
        )

    def is_audio_chunk_ready(self, state: RealtimeTranscriptionState) -> bool:
        """Require another full audio chunk so retries wait for new input."""
        return (
            state.audio.received_bytes - state.audio.last_attempted_offset_bytes
            >= self.chunk_size_bytes
        )

    async def process_transcription(
        self,
        state: RealtimeTranscriptionState,
        *,
        is_last: bool,
        sampling_params: Dict[str, JsonValue],
        on_transcript_delta: TranscriptDeltaCallback,
    ) -> None:
        """Save transcription progress only after sending the new text succeeds."""
        audio = state.audio
        # Process one chunk at a time so a large append cannot skip the point
        # where windowed transcription starts. Final requests include all audio.
        end_offset_bytes = (
            audio.received_bytes
            if is_last or self.window_mode is None
            else min(
                audio.received_bytes,
                audio.last_attempted_offset_bytes + self.chunk_size_bytes,
            )
        )
        mode_state = state.mode_state
        if (
            self.window_mode is not None
            and isinstance(mode_state, CumulativeTranscriptionState)
            and not mode_state.windowing_disabled
            and not is_last
            and end_offset_bytes > self.window_mode.activation_threshold_bytes
        ):
            mode = self.window_mode
        else:
            mode = self.get_transcription_mode(state)
        request = mode.build_transcription_request(
            state, end_offset_bytes=end_offset_bytes, is_last=is_last
        )

        async def _send_transcript_candidate(transcript_candidate: str) -> None:
            await self.send_transcript_candidate(
                state,
                transcript_candidate,
                previous_emitted_text=request.emitted_text,
                on_transcript_delta=on_transcript_delta,
            )

        outcome = await mode.transcribe_audio(
            state,
            request,
            sampling_params=sampling_params,
            on_transcript_candidate=(
                _send_transcript_candidate
                if self.decoder_streaming and not is_last
                else None
            ),
        )
        await self.send_transcript_candidate(
            state,
            outcome.delta,
            previous_emitted_text=request.emitted_text,
            on_transcript_delta=on_transcript_delta,
            conflict_error="completed ASR decode revised already streamed text",
        )
        self.apply_transcription_outcome(state, request, outcome)

    async def flush_pending_transcript(
        self,
        state: RealtimeTranscriptionState,
        on_transcript_delta: TranscriptDeltaCallback,
    ) -> None:
        """Send pending transcript text without transcribing the audio again."""
        previous_emitted_text = state.emitted_text
        mode = self.get_transcription_mode(state)
        outcome = mode.flush_pending_transcript(state)
        await self.send_transcript_candidate(
            state,
            outcome.delta,
            previous_emitted_text=previous_emitted_text,
            on_transcript_delta=on_transcript_delta,
            conflict_error="ASR flush revised already published text",
        )
        self.apply_transcription_outcome(state, None, outcome)

    async def send_transcript_candidate(
        self,
        state: RealtimeTranscriptionState,
        transcript_candidate: str,
        *,
        previous_emitted_text: str,
        on_transcript_delta: TranscriptDeltaCallback,
        conflict_error: Optional[str] = None,
    ) -> None:
        """Prevent partial or final results from rewriting text already sent."""
        # Callbacks include earlier text from this request; send only the new part.
        emitted_text_in_request = state.emitted_text[len(previous_emitted_text) :]
        if not transcript_candidate.startswith(emitted_text_in_request):
            if conflict_error is not None:
                raise RuntimeError(conflict_error)
            return
        delta = transcript_candidate[len(emitted_text_in_request) :]
        if delta:
            await on_transcript_delta(delta)
            state.emitted_text += delta

    def get_transcription_mode(
        self, state: RealtimeTranscriptionState
    ) -> TranscriptionMode:
        if state.encoder_window_active:
            assert self.window_mode is not None
            return self.window_mode
        return self.cumulative_mode

    def apply_transcription_outcome(
        self,
        state: RealtimeTranscriptionState,
        request: Optional[TranscriptionRequest],
        outcome: TranscriptionOutcome,
    ) -> None:
        """Save state and trim audio only after sending the transcript succeeds."""
        audio = state.audio
        state.mode_state = outcome.next_mode_state
        if request is not None:
            audio.last_attempted_offset_bytes = request.end_offset_bytes
            if outcome.audio_processed:
                audio.last_processed_offset_bytes = request.end_offset_bytes
        if outcome.trim_buffer_offset_bytes is not None:
            audio.trim_buffer(outcome.trim_buffer_offset_bytes)
