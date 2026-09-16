from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Dict, Optional

import msgspec
from pydantic import JsonValue

from sglang.srt.entrypoints.openai.realtime.audio_transcription.transcription_state import (
    RealtimeTranscriptionState,
    TranscriptionModeState,
)
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager

# Each candidate includes all new text for this request, including text already
# sent by earlier callbacks, so the caller must remove that part before sending.
TranscriptCandidateCallback = Callable[[str], Awaitable[None]]


class TranscriptionRequest(msgspec.Struct, frozen=True):
    """Keep the starting audio range and text state fixed while transcription runs."""

    is_last: bool
    start_offset_bytes: int
    end_offset_bytes: int
    last_attempted_offset_bytes: int
    last_processed_offset_bytes: int
    emitted_text: str
    mode_state: TranscriptionModeState
    # Keep preceding samples so feature extraction can compute the first frame.
    leading_context_bytes: int = 0
    decoder_prefix: str = ""
    # Preserve unsent text when switching from cumulative to windowed transcription.
    pending_transcript: str = ""


class TranscriptionOutcome(msgspec.Struct, frozen=True):
    """Apply these changes only after the transcript delta is sent successfully."""

    next_mode_state: TranscriptionModeState
    # False keeps this audio in the next request so transcription can retry it.
    audio_processed: bool
    # Include all new text since the request started, even if some was streamed.
    delta: str = ""
    trim_buffer_offset_bytes: int | None = None


class TranscriptionMode(ABC):
    """Transcription results for the processor."""

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        adapter: TranscriptionAdapter,
    ) -> None:
        self.tokenizer_manager = tokenizer_manager
        self.adapter = adapter

    @abstractmethod
    def build_transcription_request(
        self,
        state: RealtimeTranscriptionState,
        *,
        end_offset_bytes: int,
        is_last: bool,
    ) -> TranscriptionRequest:
        """Prepare the audio range and text state for one audio transcription request."""
        ...

    @abstractmethod
    async def transcribe_audio(
        self,
        state: RealtimeTranscriptionState,
        request: TranscriptionRequest,
        *,
        sampling_params: Dict[str, JsonValue],
        on_transcript_candidate: Optional[TranscriptCandidateCallback],
    ) -> TranscriptionOutcome:
        """Run one audio transcription request and return its text and next state."""
        ...

    @abstractmethod
    def flush_pending_transcript(
        self, state: RealtimeTranscriptionState
    ) -> TranscriptionOutcome:
        """Prepare unsent transcript text without running the model again."""
        ...
