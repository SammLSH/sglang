from __future__ import annotations

import msgspec

from sglang.srt.entrypoints.openai.realtime.audio_transcription.audio_buffer import (
    AudioBuffer,
)
from sglang.srt.entrypoints.openai.realtime.audio_transcription.transcription_suffix import (
    TranscriptionSuffixState,
)
from sglang.srt.entrypoints.openai.streaming_asr import (
    StreamingASRState,
)


class WindowedTranscriptionState(msgspec.Struct, frozen=True, tag="windowed"):
    """Keep unsent text and its audio until later transcriptions confirm the text."""

    suffix: TranscriptionSuffixState
    consecutive_failures: int = 0
    retrying_empty_text: bool = False
    # Reuse this audio until later transcriptions confirm its text.
    unconfirmed_audio_start_offset_bytes: int | None = None


TranscriptionModeState = StreamingASRState | WindowedTranscriptionState


class RealtimeTranscriptionState(msgspec.Struct):
    """Store one audio item's buffered audio, sent transcript, and transcription state."""

    audio: AudioBuffer
    # The state type stays fixed; only accepted candidates replace its contents.
    mode_state: TranscriptionModeState
    emitted_text: str = ""

    @property
    def has_audio(self) -> bool:
        return self.audio.received_bytes > 0

    @property
    def has_transcript(self) -> bool:
        if isinstance(self.mode_state, StreamingASRState):
            return bool(self.mode_state.full_transcript)
        else:
            return bool(self.emitted_text or self.mode_state.suffix.pending_text)

    @property
    def has_unprocessed_audio(self) -> bool:
        return self.audio.received_bytes > self.audio.last_processed_offset_bytes
