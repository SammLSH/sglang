"""Per-item audio progress, publication, and the active transcription mode state."""

from __future__ import annotations

import msgspec

from sglang.srt.entrypoints.openai.realtime.audio_transcription.audio_buffer import (
    AudioBuffer,
)
from sglang.srt.entrypoints.openai.realtime.audio_transcription.transcription_suffix import (
    TranscriptionSuffixState,
)
from sglang.srt.entrypoints.openai.streaming_transcription import (
    CumulativeTranscriptState,
)


class CumulativeState(msgspec.Struct, frozen=True, tag="cumulative"):
    """Cumulative candidates and recovery before a window handoff commits."""

    transcript: CumulativeTranscriptState
    window_disabled: bool = False
    handoff_failures: int = 0
    # An empty first continuation is retried without activating windowing.
    deferred_empty_continuation: bool = False


class WindowedState(msgspec.Struct, frozen=True, tag="windowed"):
    """Continuation candidates and the audio retained for their confirmation."""

    suffix: TranscriptionSuffixState
    consecutive_failures: int = 0
    deferred_empty_continuation: bool = False
    # Keep the request start fixed while decoded text is still unconfirmed.
    unconfirmed_start: int | None = None


ModeState = CumulativeState | WindowedState


class RealtimeTranscriptionState(msgspec.Struct):
    """One audio item's shared facts and exactly one accepted mode state.

    Only the processor appends emitted_text after successful sends.
    Modes work on snapshots; the processor replaces mode_state at commit.
    """

    audio: AudioBuffer
    mode_state: ModeState
    emitted_text: str = ""

    @property
    def has_audio(self) -> bool:
        return self.audio.received_bytes > 0

    @property
    def has_transcript(self) -> bool:
        if isinstance(self.mode_state, CumulativeState):
            return bool(self.mode_state.transcript.full_transcript)
        return bool(self.emitted_text or self.mode_state.suffix.pending)

    @property
    def has_new_audio(self) -> bool:
        return self.audio.received_bytes > self.audio.last_processed_offset_bytes

    @property
    def encoder_window_active(self) -> bool:
        return isinstance(self.mode_state, WindowedState)
